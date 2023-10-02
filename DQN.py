import gym
import random
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from collections import deque

from control import Chessboard

class Buffer:
    def __init__(self):
        self.poses = []

    def addin(self, str_pos, end_pos):
        find = False
        for idx, (strposes, endposes) in enumerate(self.poses):
            if str_pos == endposes:
                self.poses[idx] = (str_pos,end_pos)
                find = True
                break
        if find == False:
            self.poses.append((str_pos,end_pos))

    def sampling(self):
        return self.poses

# 定义一个简单的Q网络
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 512)
        self.fc7 = nn.Linear(512, 1024)
        self.fc8 = nn.Linear(1024, 386)
        self.fc2 = nn.Linear(386, hidden_size)
        
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, 586),
            nn.ReLU(),
            nn.Linear(586, 1536)
        )

        self.fc3 = nn.Linear(1536, 987)
        self.fc4 = nn.Sequential(
            nn.Linear(987, 1024),
            nn.LeakyReLU(negative_slope=0.01),  # 使用Leaky ReLU函数，斜率为0.01
            nn.Linear(1024, 854)
        )
        self.fc5 = nn.Linear(854, 625)
        self.fc6 = nn.Linear(625, action_size)

    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.elu(self.fc7(x))
        x = F.elu(self.fc8(x))
        x = F.elu(self.fc2(x))
        state_value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        advantage = F.relu(self.fc3(advantage))
        advantage = self.fc4(advantage)
        advantage = F.elu(self.fc5(advantage))
        advantage = F.relu(self.fc6(advantage))
        if len(advantage.shape) == 2:
            advantage_mean = advantage.mean(dim=1, keepdim=True)
        if len(advantage.shape) == 1:
            advantage_mean = advantage.mean(dim=0, keepdim=True)
        # print(advantage,"advantage")
        # print(advantage_mean)
        q_values = state_value + (advantage - advantage_mean)
        return q_values

# 定义DQN算法
class DQN:
    def __init__(self, state_size, action_size, hidden_size, learning_rate, gamma, epsilon_decay):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = 1.0       #  ε-貪婪策略中的探索率
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        self.memory = deque(maxlen=8000)  # 经验回放缓存
        self.q_network = QNetwork(state_size, action_size, hidden_size)
        self.target_network = QNetwork(state_size, action_size, hidden_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate, weight_decay=0.0002)
        self.scheduler = StepLR(self.optimizer, step_size=70, gamma=self.gamma)
        self.loss_values = []  # 初始化损失值列表
        # 進行正規化
    
    def select_action(self, state, valid_actions):
        # print(self.epsilon, "epsilon")
        if random.uniform(0, 1) > self.epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                state_tensor = state_tensor.view(-1)
                # print(state_tensor)
                q_values = self.q_network(state_tensor)
                valid_q_values = q_values[valid_actions]  # 只選擇合法動作的 Q 值
                action = valid_q_values.argmax().item()
                action = valid_actions[action]
        else:
            action = random.choice(valid_actions)
        return action
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def returnloss(self):
        return self.loss_values
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        if len(self.memory) > 4*batch_size:
            batch = random.sample(self.memory, batch_size*4)
            batch = sorted(batch, key=lambda x: x[2], reverse=True)  # 按照奖励从大到小排序
            selected_batch = batch[:60]  # 选择前60个样本
            batch = random.sample(selected_batch, batch_size)
        else:
            batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states_tensor = torch.FloatTensor(states)
        states_tensor = states_tensor.view(states_tensor.shape[0], -1)
        next_states_tensor = torch.FloatTensor(next_states)
        next_states_tensor = next_states_tensor.view(next_states_tensor.shape[0], -1)
        rewards_tensor = torch.FloatTensor(rewards)
        actions_tensor = torch.LongTensor(actions)
        dones_tensor = torch.FloatTensor(dones)

        # 添加噪声，例如在states_tensor中加入一些随机扰动
        # print(states_tensor,"state")
        states_tensor += torch.FloatTensor(np.random.normal(0, 0.05, size=states_tensor.shape))  # 标准差为0.05的正态分布中生成随机噪声
        # print(states_tensor,"state")
        
        current_q_values = self.q_network(states_tensor).gather(1, actions_tensor.view(-1, 1))

        # 使用目標網絡估計下一狀態的動作值   
        # 雙網絡 DQN
        next_q_values_online = self.q_network(next_states_tensor)
        next_q_values_target = self.target_network(next_states_tensor)
        next_actions = next_q_values_online.argmax(dim=1, keepdim=True)
        next_q_values = next_q_values_target.gather(1, next_actions).detach()

        # next_q_values = self.target_network(next_states_tensor).max(1)[0].detach()
        target_q_values = rewards_tensor + self.gamma * next_q_values * (1 - dones_tensor)
        
        loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))

        self.loss_values.append(loss.item())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新學習率
        self.scheduler.step()
        
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def delloss(self):
        self.loss_values = []

# Define a function to preprocess the state
def preprocess_state(chess):
    # Convert the chessboard to a flattened representation
    state = [[0 , 0 , 0 , 0] for _ in range(90)]
    index = -1
    for i in chess.chessboard:
        for j in i:
            index = index + 1
            if j == 0:
                continue
            x,y = j.position()
            if j.name == '帥':
                if j.red == True:
                    state[index] = [1 , 10 , x , y] # 顏色 棋子類別 上一次位置
                else:
                    state[index] = [-1 , 10 , x , y] # 顏色 棋子類別 上一次位置
            if j.name == '士':
                if j.red == True:
                    state[index] = [1 , 20 , x , y] # 顏色 棋子類別 上一次位置
                else:
                    state[index] = [-1 , 20 , x , y] # 顏色 棋子類別 上一次位置
            if j.name == '象':
                if j.red == True:
                    state[index] = [1 , 30 , x , y] # 顏色 棋子類別 上一次位置
                else:
                    state[index] = [-1 , 30 , x , y] # 顏色 棋子類別 上一次位置
            if j.name == '車':
                if j.red == True:
                    state[index] = [1 , 40 , x , y] # 顏色 棋子類別 上一次位置
                else:
                    state[index] = [-1 , 40 , x , y] # 顏色 棋子類別 上一次位置
            if j.name == '馬':
                if j.red == True:
                    state[index] = [1 , 50 , x , y] # 顏色 棋子類別 上一次位置
                else:
                    state[index] = [-1 , 50 , x , y] # 顏色 棋子類別 上一次位置
            if j.name == '炮':
                if j.red == True:
                    state[index] = [1 , 60 , x , y] # 顏色 棋子類別 上一次位置
                else:
                    state[index] = [-1 , 60 , x , y] # 顏色 棋子類別 上一次位置
            if j.name == '兵':
                if j.red == True:
                    state[index] = [1 , 70 , x , y] # 顏色 棋子類別 上一次位置
                else:
                    state[index] = [-1 , 70 , x , y] # 顏色 棋子類別 上一次位置

    return state

def stateoripos(poses, state):
    for (strposes,endposes) in poses:
        x,y = endposes
        index = 10 * x + y
        temp = state[index]
        (temp[2],temp[3]) = strposes
        state[index] = temp

    return state

def nextstate(action, origin, chess, marks, temp):
    reward = 0
    done = False
    marking = []
    x,y = origin
    color = chess.chessboard[x][y].red
    chess.move(origin,action)
    for mark in marks:
        if mark[3] == origin:
            marking.append((mark[0],mark[1],mark[2],action))
        elif mark[3] != action:
            marking.append(mark)

    # 计算奖励
    reward = evaluate_state(chess, color, temp, action, x, y) - evaluate_state(chess, not color, temp, action,x ,y)
    reward = reward * 0.25  # 缩放奖励的大小，可以根据实际情况调整
    reward = round(reward, 5)

    # 检查游戏是否结束
    done = chess.is_game_over()

    return reward, done, marking

def evaluate_state(chess , color, temp, action, x, y):
    restchess = []
    c =[]
    che =[]
    pao =[]
    chessx, chessy =action
    value = 0.0
    count = 0
    for i in chess.chessboard:
        for j in i:
            if j == 0: continue
            if j != 0: count = count + 1
            if j.name == "帥"  and  (not j.red == color):
                restchess.append((j.name, j.position(), j.red))
                shuaicanmove = j.try_move(j.position(), chess.chessboard)
            if j.name == "車"  and  j.red == color:
                che.append(j)
            if j.name == "炮"  and  j.red == color:
                pao.append(j)

    if len(shuaicanmove) < 2 and (restchess[0][1] != (4,0) or restchess[0][1] != (4,9)):
        value = value - 0.1
    # print(restchess[0][1], restchess[0][0])

    thischess = chess.value(chessx , chessy)

    if thischess.name == "士":
        dicount = 0.96
    if thischess.name == "象":
        dicount = 0.96
    if thischess.name == "車":
        dicount = 1.86
        if chessx == x:
            for i in range(chessx + 1,9):
                tempvalue = chess.value(i,chessy)
                if tempvalue == 0: continue
                if tempvalue.red == color:
                    break
                if tempvalue.red == (not color):
                    value = value + 0.04
                    break
            for i in range(chessx - 1, -1, -1):
                tempvalue = chess.value(i,chessy)
                if tempvalue == 0: continue
                if tempvalue.red == color:
                    break
                if tempvalue.red == (not color):
                    value = value + 0.04
                    break
        if chessy == y:
            for i in range(chessy + 1,10):
                tempvalue = chess.value(chessx,i)
                if tempvalue == 0: continue
                if tempvalue.red == color:
                    break
                if tempvalue.red == (not color):
                    value = value + 0.04
                    break
            for i in range(chessy - 1, -1, -1):
                tempvalue = chess.value(chessx,i)
                if tempvalue == 0: continue
                if tempvalue.red == color:
                    break
                if tempvalue.red == (not color):
                    value = value + 0.04
                    break
        if color == True:
            if chessx == 0 and 3 <= chessy <= 9 :
                block1 = chess.value(chessx , chessy-1)
                block2 = chess.value(chessx+1 , chessy)
                if block1 != 0 and block2 !=0:
                    if block1.red == color and block2.red == color:
                        value = value - 0.05
            if chessx == 8 and 3 <= chessy <= 9 :
                block1 = chess.value(chessx , chessy-1)
                block2 = chess.value(chessx-1 , chessy)
                if block1 != 0 and block2 !=0:
                    if block1.red == color and block2.red == color:
                        value = value - 0.05
        if color == False:
            if chessx == 0 and chessy <= 6 :
                block1 = chess.value(chessx , chessy+1)
                block2 = chess.value(chessx+1 , chessy)
                if block1 != 0 and block2 !=0:
                    if block1.red == color and block2.red == color:
                        value = value - 0.05
            if chessx == 8 and chessy <= 6 :
                block1 = chess.value(chessx , chessy+1)
                block2 = chess.value(chessx-1 , chessy)
                if block1 != 0 and block2 !=0:
                    if block1.red == color and block2.red == color:
                        value = value - 0.05
    if thischess.name == "馬":
        dicount = 1.46
        for enemyx, enemyy in thischess.try_move(thischess.position(),chess.chessboard):
            enemy = chess.value(enemyx,enemyy)
            if enemy == 0: continue
            if enemy.red == (not color):
                value = value + 0.02
            if len(che) < 2 and thischess.red == True:
                if chessy <= 4:
                    value = value + 0.04
                    if len(pao) < 2:
                        value = value + 0.03
            if len(che) < 2 and thischess.red == False:
                if chessy > 4:
                    value = value + 0.04
                    if len(pao) < 2:
                        value = value + 0.03
    if thischess.name == "炮":
        dicount = 1.46
        c = []
        for i in range(0,chessx + 1):
            tempvalue = chess.value(i,chessy)
            if tempvalue == 0: continue
            c.append(tempvalue)
        c = c[::-1]
        if len(c) >= 3:
            if c[0].red != c[2].red:
                value = value + 0.03
        c = []
        for i in range(chessx, 9):
            tempvalue = chess.value(i,chessy)
            if tempvalue == 0: continue
            c.append(tempvalue)
        if len(c) >= 3:
            if c[0].red != c[2].red:
                value = value + 0.03
        c = []
        for i in range(0,chessy + 1):
            tempvalue = chess.value(chessx,i)
            if tempvalue == 0: continue
            c.append(tempvalue)
        c = c[::-1]
        if len(c) >= 3:
            if c[0].red != c[2].red:
                value = value + 0.03
        c = []
        for i in range(chessy,10):
            tempvalue = chess.value(chessx,i)
            if tempvalue == 0: continue
            c.append(tempvalue)
        if len(c) >= 3:
            if c[0].red != c[2].red:
                value = value + 0.03
    if thischess.name == "兵":
        dicount = 0.57

    shuaix , shuaiy = restchess[0][1]
    for i in chess.chessboard:
        for j in i:
            if j == 0: continue
            if j.red == color:
                x,y = j.position()
                if j.name == "士":
                    value = value + 0.9
                if j.name == "象":
                    value = value + 0.9
                    if abs(x - 4) == 2 or abs(x - 4) == 0:
                        value = value + 0.05
                    if 0 <= y <= 2 or  7 <= y <= 9:
                        value = value + 0.02
                if j.name == "車":
                    value = value + 1.8
                    if abs(y - shuaiy) <= 4:
                        value = value + 0.04
                    if count < 10 and (abs(x - shuaix) <= 2 or abs(x - shuaix) <= 2):
                        value = value + 0.1 
                if j.name == "馬":
                    value = value + 1.4
                    if abs(y - shuaiy) <= 4:
                        value = value + 0.01
                    if abs(x - shuaix) <= 4:
                        value = value + 0.01
                if j.name == "炮":
                    value = value + 1.4
                if j.name == "兵":
                    value = value + 0.55
                if j.can_move(j.position(), (shuaix,shuaiy), chess.chessboard):
                    value = value + 0.4
                    if len(temp) > 1:
                        if temp[0][2] == color and j.position == temp[0][1]:
                            value = value - 0.44
                        if temp[1][2] == color and j.position == temp[1][1]:
                            value = value - 0.44
            if j.red == (not color):
                if j.can_move(j.position(), (chessx,chessy), chess.chessboard):
                    value = value - dicount

    return value

def get_move(actions, str_poses, end_poses, valid_actions):
    origin = None
    action = None

    if len(valid_actions) > 0:
        for index , value in enumerate(valid_actions):
            if value == actions:
                origin = str_poses[index]
                action = end_poses[index]
                break
    
    return origin, action

def validaction(valid_moves,chess):
    valid_actions =[]
    str_pos = []
    end_pos = []
    all_moves = list(valid_moves.keys())

    for move in all_moves:
        x, y = move[1]
        name = move[3]
        for (posx,posy) in valid_moves[move]:
            movingx, movingy = (posx - x,posy - y)
            if name == '車':   # 0 - 67
                if abs(movingx) == 0:
                    value = movingy + 9
                    if movingy > 0:
                        value = value - 1
                if abs(movingy) == 0:
                    value = movingx + 8 + 18
                    if movingx > 0:
                        value = value - 1
                if move[2] == 2:
                    value = value + 34
            if name == '炮':   # 68 - 135
                if abs(movingx) == 0:
                    value = movingy + 9 + 68
                    if movingy > 0:
                        value = value - 1
                if abs(movingy) == 0:
                    value = movingx + 8 + 18 + 68
                    if movingx > 0:
                        value = value - 1
                if move[2] == 2:
                    value = value + 34
            if name == '象':   # 136 - 143
                if movingx == 2:
                    value = 137 + (movingy // 2)
                if movingx == -2:
                    value = 138 + (movingy // 2)
                if move[2] == 2:
                    value = value + 4
            if name == '士':   # 144 - 151
                if movingx == 1:
                    value = 145 + movingy
                if movingx == -1:
                    value = 146 + movingy
                if move[2] == 2:
                    value = value + 4
            if name == '帥':   # 152 - 155
                if movingx == 0:
                    value = 153 + movingy
                if movingy == 0:
                    value = 154 + movingx
            if name == '兵':   # 156 - 170
                if movingy == 0:
                    value = 157 + movingx
                else:
                    value = 157
                value = value + (move[2] - 1) * 3
            if name == '馬':   # 171 - 186
                if abs(movingx) == 1:
                    value = (movingx + movingy + 3) // 2 + 171
                if abs(movingx) == 2:
                    value = (movingx + movingy + 3) // 2 + 175
                if move[2] == 2:
                    value = value + 8
            valid_actions.append(value)
            str_pos.append(move[1])
            end_pos.append((posx,posy))

    return valid_actions, str_pos, end_pos

def mark(chess):
    marking = []
    index = 1
    for i in chess.chessboard:
        for j in i:
            if j == 0: continue
            x,y = j.position()
            if y == 3 or y == 6:
                index = 1 + x // 2
            if y == 0 or y == 9:
                if x < 4:
                    index = 1
                if x > 4:
                    index = 2
            marking.append((index,(j.red),(j.name),(j.position())))

    return marking

def combine(valid_moves, marking):
    all_moves = list(valid_moves.keys())

    for marks in marking:
        for move in all_moves:
            if marks[3] == move[1]:
                valid_moves[(move[0],move[1],marks[0],marks[2])] = valid_moves[move]
                del valid_moves[move]

    return valid_moves

def justmove(end_pos, chess, temparr):
    pastmove = temparr
    x, y = end_pos
    temp = chess.value(x , y)
    if len(pastmove) < 2:
        pastmove.append((temp.name, temp.position(), temp.red))
    else:
        if temp.red == pastmove[1][2]:
            pastmove[1] = (temp.name, temp.position(), temp.red)
        else:
            pastmove[0] = (temp.name, temp.position(), temp.red)
    return pastmove


# 主训练循环
def main():
    state_size = 90*4
    action_size = 187
    hidden_size = 192
    learning_rate = 0.001
    gamma = 0.99
    epsilon_decay = 0.99
    batch_size = 25
    episodes = 100000
    max_steps = 600
    save_interval = 10  # 設定保存模型的間隔


    dqn_agent = DQN(state_size, action_size, hidden_size, learning_rate, gamma, epsilon_decay)

    # 載入已經儲存的模型狀態
    # saved_model_path = 'DQNmodel.pth'
    # checkpoint = torch.load(saved_model_path)
    # dqn_agent.q_network.load_state_dict(checkpoint['model_state_dict'])
    # dqn_agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    for episode in range(episodes):
        # 創建 Chessboard 實例
        chess = Chessboard()
        state = preprocess_state(chess)
        # state = torch.tensor(state, dtype=torch.float32)
        # state = state.view(-1)
        marking = mark(chess)
        total_reward = 0
        temp = []
        done = False
        buffer = Buffer()
        
        for step in range(max_steps):
            print("迭代次數",step)
            if done:
                break

            valid_moves = chess.get_valid_moves()
            if len(valid_moves) == 0:
                done = True
                break

            valid_moves = combine(valid_moves, marking)
            valid_actions, str_poses, end_poses = validaction(valid_moves, chess)
            # print(valid_actions)

            actions = dqn_agent.select_action(state, valid_actions)
            # print(actions)
            origin, action = get_move(actions, str_poses, end_poses, valid_actions)
            # print(origin,action)

            # 執行遊戲行動，並取得下一個遊戲狀態、獎勵和結束標誌
            reward, done, marking = nextstate(action, origin, chess, marking, temp) 
            buffer.addin(origin,action)
            temp = justmove(action, chess, temp)

            next_state = preprocess_state(chess)  # 更新state为新的棋盘状态
            poses = buffer.sampling()
            next_state = stateoripos(poses, next_state)
            # print(next_state)
            # next_state = torch.tensor(next_state, dtype=torch.float32)
            # next_state = next_state.view(-1)
            
            if reward > -0.95:
                dqn_agent.remember(state, actions, reward, next_state, done)
            total_reward += reward
            state = next_state
            
            if len(dqn_agent.memory) > batch_size:
                dqn_agent.replay(batch_size)
        
        loss_values = dqn_agent.returnloss()
        print(f"Episode {episode + 1} - Total Reward: {total_reward:.2f}")
        print(f"Episode {episode + 1}, Loss: {np.mean(loss_values)}")
        if total_reward > -25 or step > 250:
            dqn_agent.update_target_network()

        if episode % save_interval == 0:
            torch.save({
                'model_state_dict': dqn_agent.q_network.state_dict(),
                'optimizer_state_dict': dqn_agent.optimizer.state_dict(),
            }, f'DQNorimodel.pth')

        # 保存每個 episode 的訓練結果（總獎勵）
        with open('training_results_ori.txt', 'a') as f:
            f.write(f"Episode {episode + 1} - Total Reward: {total_reward:.2f}\n")
            f.write(f"Episode {episode + 1} - loss: {np.mean(loss_values):.2f}\n")

        dqn_agent.delloss()

if __name__ == "__main__":
    main()