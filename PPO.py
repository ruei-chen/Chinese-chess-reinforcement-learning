import sys
import math
import random
# import os

# import gym
import numpy as np
import copy
# import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from control import Chessboard

class Buffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.masks = []
        self.poses = []

    def add(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.masks.append(1 - done)

    def addin(self, str_pos, end_pos):
        find = False
        for idx, (strposes, endposes) in enumerate(self.poses):
            if str_pos == endposes:
                self.poses[idx] = (str_pos,end_pos)
                find = True
                break
        if find == False:
            self.poses.append((str_pos,end_pos))


    def sample(self, batch_size):
        indices = torch.randint(len(self.states), size=(batch_size,))
        states = [self.states[i] for i in indices]  # 选择对应索引的数据
        actions = torch.tensor(self.actions, dtype=torch.long)[indices]
        rewards = torch.tensor(self.rewards, dtype=torch.float32)[indices]
        next_states = torch.stack(self.next_states)[indices]
        masks = torch.tensor(self.masks, dtype=torch.float32)[indices]

        return states, actions, rewards, next_states, masks
    
    def sampling(self):
        return self.poses

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, feature_dim, action_dim, hidden_size, num_transformer_blocks, num_heads):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 2260)
        self.fc2 = nn.Linear(2260, 4058)
        self.fc8 = nn.Linear(4058, 3192)
        self.fc3 = nn.Linear(3192, 1832)
        self.fc4 = nn.Linear(1832, 1024)
        self.fc5 = nn.Linear(1024, hidden_size)
        self.fc6 = nn.Linear(hidden_size, hidden_size)
        self.fc7 = nn.Linear(hidden_size, 352)
        self.fc9 = nn.Linear(352, action_dim)

    def forward(self, state):
        if len(state.shape) == 3:
            state = state.view(state.shape[0], -1)
        else:
            state = state.view(-1)
        
        state = state.clone().detach().requires_grad_(True)
            
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.elu(self.fc8(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.elu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.leaky_relu(self.fc7(x))
        action_probs = F.softmax(self.fc9(x), dim=-1)
        return action_probs

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, feature_dim, hidden_size, num_transformer_blocks, num_heads):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 1260)
        self.fc2 = nn.Linear(1260, 512)
        self.fc3 = nn.Linear(512, 208)
        self.fc4 = nn.Linear(208, 1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, state):
        if len(state.shape) == 3:
            state = state.view(state.shape[0], -1)
        else:
            state = state.view(-1)
        
        state = state.clone().detach().requires_grad_(True)
        
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        state_value = self.fc4(x)
        return state_value

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

def nextstate(action_str, ori_position, chess, temp, selected_indexes):
    rewards = []
    next_states = []
    dones = []
    actions = []
    indexes = []
    best_next_state = None
    best_reward = None
    best_action = None
    ori = None
    best_indexes = None
    done = None

    for index, value in enumerate(ori_position):
        x , y = value[1]
        color = chess.chessboard[x][y].red
        action = action_str[index]
        new_chess = copy.deepcopy(chess)  # 创建一个新的Chessboard实例
        next_state, reward, done = execute_action(action, value, new_chess, color, temp)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(done)
        indexes.append(selected_indexes[index])

    # 选择奖励最大的动作
    if len(rewards) > 0:
        best_indices = np.where(rewards == np.max(rewards))[0]  # Find the indices of maximum rewards
        if len(best_indices) > 1:
            best_index = random.choice(best_indices)  # Randomly select one of the best indices
        else:
            best_index = best_indices[0]  # Use the single best index
        best_action = actions[best_index]
        best_reward = rewards[best_index]
        best_next_state = next_states[best_index]
        done = dones[best_index]
        best_indexes = indexes[best_index]
        ori = ori_position[best_index][1]

    return best_next_state, best_reward, done, best_action, ori, best_indexes

def execute_action(action, ori_position, chess, color, temp):
    with torch.no_grad():
    # 执行行动并更新棋盘状态
        chess.move(ori_position[1], action)
        x, y = ori_position[1]
        # print(chess.chessboard[x][y].position(),chess.chessboard[x][y].name)

        # 计算奖励
        reward = evaluate_state(chess, color, temp, action, x, y) - evaluate_state(chess, not color, temp, action,x ,y)
        reward = reward * 0.3  # 缩放奖励的大小，可以根据实际情况调整
        reward = round(reward, 5)
        print(reward)

        # 检查游戏是否结束
        done = chess.is_game_over()

        # 获取新的状态并返回
        next_state = preprocess_state(chess)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        next_state = next_state.view(-1)

        chess.move(action,ori_position[1])

    return next_state, reward, done

def evaluate_state(chess , color, temp, action, x, y):
    restchess = []
    c =[]
    chessx, chessy =action
    value = 0.0
    count = 0
    for i in chess.chessboard:
        for j in i:
            if j == 0: continue
            if j != 0: count = count + 1
            if j.name == "帥"  and  (not j.red == color):
                restchess.append((j.name, j.position(), j.red))

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
                    value = value + 0.03
                    break
            for i in range(chessx - 1, -1, -1):
                tempvalue = chess.value(i,chessy)
                if tempvalue == 0: continue
                if tempvalue.red == color:
                    break
                if tempvalue.red == (not color):
                    value = value + 0.03
                    break
        if chessy == y:
            for i in range(chessy + 1,10):
                tempvalue = chess.value(chessx,i)
                if tempvalue == 0: continue
                if tempvalue.red == color:
                    break
                if tempvalue.red == (not color):
                    value = value + 0.03
                    break
            for i in range(chessy - 1, -1, -1):
                tempvalue = chess.value(chessx,i)
                if tempvalue == 0: continue
                if tempvalue.red == color:
                    break
                if tempvalue.red == (not color):
                    value = value + 0.03
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
    if thischess.name == "炮":
        dicount = 1.46
        c = []
        for i in range(0,chessx + 1):
            tempvalue = chess.value(i,chessy)
            if tempvalue == 0: continue
            c.append(tempvalue)
        c = c[::-1]
        if len(c) > 3:
            if c[0].red != c[2].red:
                value = value + 0.03
        c = []
        for i in range(chessx, 9):
            tempvalue = chess.value(i,chessy)
            if tempvalue == 0: continue
            c.append(tempvalue)
        if len(c) > 3:
            if c[0].red != c[2].red:
                value = value + 0.03
        c = []
        for i in range(0,chessy + 1):
            tempvalue = chess.value(chessx,i)
            if tempvalue == 0: continue
            c.append(tempvalue)
        c = c[::-1]
        if len(c) > 3:
            if c[0].red != c[2].red:
                value = value + 0.03
        c = []
        for i in range(chessy,10):
            tempvalue = chess.value(chessx,i)
            if tempvalue == 0: continue
            c.append(tempvalue)
        if len(c) > 3:
            if c[0].red != c[2].red:
                value = value + 0.03
        if chessx != 0 and chessx != 8:
            if chess.chessboard[chessx-1][chessy] != 0 and chess.chessboard[chessx+1][chessy] != 0:
                value = value - 0.025

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
                        value = value + 0.06
                    if 0 <= y <= 2 or  7 <= y <= 9:
                        value = value + 0.01
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

# Define a function to convert action probabilities to chess moves
def convert_action(action_probs, chess, marking):
    valid_moves = chess.get_valid_moves()
    # print(valid_moves)
    action_prob = action_probs.detach().numpy()
    all_moves = []
    if len(valid_moves) > 0:
        valid_moves = combine(valid_moves, marking)

        all_moves = list(valid_moves.keys())
        done = False

        all_probs = []
        all_positions = []
        possible_move = []
        indexes = []
        # print(all_moves)


        # 車90 馬90 象7 士5 帥9 炮90 兵55  346*2
        for move in all_moves:
            x, y = move[1]
            name = move[3]
            for (posx,posy) in valid_moves[move]:
                # print(posx,posy,"pos")
                movingx, movingy = (posx - x,posy - y)
                if name == '車':   # 0 - 67
                    if abs(movingx) == 0:
                        index = movingy + 9
                        if movingy > 0:
                            index = index - 1
                    if abs(movingy) == 0:
                        index = movingx + 8 + 18
                        if movingx > 0:
                            index = index - 1
                    if move[2] == 2:
                        index = index + 34
                if name == '炮':   # 68 - 135
                    if abs(movingx) == 0:
                        index = movingy + 9 + 68
                        if movingy > 0:
                            index = index - 1
                    if abs(movingy) == 0:
                        index = movingx + 8 + 18 + 68
                        if movingx > 0:
                            index = index - 1
                    if move[2] == 2:
                        index = index + 34
                if name == '象':   # 136 - 143
                    if movingx == 2:
                        index = 137 + (movingy // 2)
                    if movingx == -2:
                        index = 138 + (movingy // 2)
                    if move[2] == 2:
                        index = index + 4
                if name == '士':   # 144 - 151
                    if movingx == 1:
                        index = 145 + movingy
                    if movingx == -1:
                        index = 146 + movingy
                    if move[2] == 2:
                        index = index + 4
                if name == '帥':   # 152 - 155
                    if movingx == 0:
                        index = 153 + movingy
                    if movingy == 0:
                        index = 154 + movingx
                if name == '兵':   # 156 - 170
                    if movingy == 0:
                        index = 157 + movingx
                    else:
                        index = 157
                    index = index + (move[2] - 1) * 3
                if name == '馬':   # 171 - 186
                    if abs(movingx) == 1:
                        index = (movingx + movingy + 3) // 2 + 171
                    if abs(movingx) == 2:
                        index = (movingx + movingy + 3) // 2 + 175
                    if move[2] == 2:
                        index = index + 8
                # valid_actions.append(value)
                # str_pos.append(move[1])
                # end_pos.append((posx,posy))
                prob = action_prob[index]
                indexes.append(index)
                all_probs.append(prob)
                all_positions.append((move[0],move[1]))
                possible_move.append((posx,posy))

        # print(all_positions)
        # print(possible_move)

        all_probs_tensor = torch.tensor(all_probs, dtype=torch.float32)
        # print(all_probs_tensor)

        if len(all_probs_tensor) > 0 and torch.all(all_probs_tensor == 0):
            selected_index = torch.randint(0, len(all_probs_tensor), (3,))
            selected_position = [all_positions[idx.item()] for idx in selected_index]
            selected_move = [possible_move[idx.item()] for idx in selected_index]
            selected_indexes = [indexes[idx.item()] for idx in selected_index]


        if (all_probs_tensor.numel() > 0 and torch.any(all_probs_tensor > 0)):
            # Normalize the probabilities
            all_probs_tensor = all_probs_tensor / torch.sum(all_probs_tensor)

            # Sample an action based on the probabilities
            if len(all_probs_tensor) > 5:
                selected_index = torch.multinomial(all_probs_tensor, 3)
            elif len(all_probs_tensor) > 3:
                selected_index = torch.multinomial(all_probs_tensor, 2)
            else:
                selected_index = torch.multinomial(all_probs_tensor, 1)
            selected_position = [all_positions[idx.item()] for idx in selected_index]
            selected_move = [possible_move[idx.item()] for idx in selected_index]
            selected_indexes = [indexes[idx.item()] for idx in selected_index]
        # for i in range(len(all_moves)):
        #     if len(valid_moves[selected_position]) > 0:
        #         break
        #     selected_position = all_moves[i]
        #     selected_move = valid_moves[selected_position]
    # else:
        # selected_position = random.choice(all_moves)
        # selected_move = valid_moves[selected_position]
        # for i in range(len(all_moves)):
        #     selected_position = all_moves[i]
        #     selected_move = valid_moves[selected_position]
        #     if len(valid_moves[selected_position]) > 0:
        #         break
    if len(all_moves) ==0:
        selected_move = []
        selected_position = []
        selected_indexes = []
        done = True
    # print(selected_index,"selected_index")
    # print(selected_indexes,"selected_indexes")
    # print(selected_move,"selected_move")
    # print(selected_position,"selected_position")
    
    return selected_move, selected_position, done, selected_indexes

def markingupdate(action, origin, marks):
    marking = []
    for mark in marks:
        if mark[3] == origin:
            marking.append((mark[0],mark[1],mark[2],action))
        elif mark[3] != action:
            marking.append(mark)

    return marking

def ppo_train(env, policy_net, value_net, num_episodes, max_steps, epsilon, gamma, lamda, epochs, batch_size):
    win = 0

    optimizer_policy = optim.Adam(policy_net.parameters(), lr=1e-4)
    optimizer_value = optim.Adam(value_net.parameters(), lr=1e-4)

    for episode in range(num_episodes):
        chessboard = Chessboard()  # 创建一个新的棋盘实例，或者使用现有的棋盘实例，根据需求选择
        state = preprocess_state(chessboard)
        states = torch.tensor(state, dtype=torch.float32)
        # state = state.view(-1)
        log_probs = []
        values = []
        rewards = []
        masks = []
        next_value = []
        temp = []
        policyloss = []
        valueloss = []
        buffer = Buffer()
        marking = mark(chessboard)
        entropy = 0

        for step in range(max_steps):
            print("迭代次數",step)
            action_probs = policy_net(states)
            action_dist = Categorical(action_probs)
            # print(action_dist)
            # action = action_dist.sample()
            # print(action,"action")
            action_str, ori_position, done, selected_indexes = convert_action(action_probs, chessboard, marking)
            # print (action_str , ori_position , "辨識")
            if done:
                # buffer.add(state, action, reward, next_state, done)
                # log_probs.append(log_prob)
                # values.append(value_net(state))
                # rewards.append(reward)  # 存储reward
                # masks.append(torch.tensor(1 - done, dtype=torch.float32))
                # entropy = entropy + action_dist.entropy().mean()
                win = win + 1

                break

            next_state, reward, done, actions, ori, action =  nextstate(action_str, ori_position, chessboard, temp, selected_indexes)

            # print(ori, actions)
            action = torch.tensor(action)
            # print(action)
            chessboard.move(ori, actions)
            marking = markingupdate(actions, ori, marking)
            temp = justmove(actions, chessboard, temp)
            # print(ori,actions)

            buffer.add(state, action, reward, next_state, done)
            buffer.addin(ori,actions)

            log_prob = action_dist.log_prob(action)
            entropy = entropy + action_dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value_net(states).squeeze())
            rewards.append(reward)  # 存储reward

            masks.append(torch.tensor(1 - done, dtype=torch.float32))

            state = preprocess_state(chessboard)  # 更新state为新的棋盘状态
            poses = buffer.sampling()
            state = stateoripos(poses, state)
            states = torch.tensor(state, dtype=torch.float32)
            # state = state.view(-1)
            # print(state)
            next_value.append(states)

            if done:
                win = win + 1
                break

        # next_state_tensor = state.clone().detach()
        # next_state_tensor = next_state_tensor.to(torch.float32)  # Ensure the data type is float32 (if needed)
        # next_value = value_net(next_state_tensor).detach()
        # print(next_value)

        returns = calculate_returns(rewards, next_value, masks, gamma, lamda)
        advantages = calculate_advantages(rewards, values, next_value, masks, gamma, lamda)

        # 將 log_probs 轉換成 PyTorch 張量
        log_probs = torch.tensor(log_probs, dtype=torch.float32)
        values = torch.stack(values, dim=0)  # Stack values into a tensor
        return_clone = returns.clone().detach()
        return_clone = return_clone.to(torch.float32)
        entropymean = torch.tensor(entropy.item(), requires_grad=True)

        for _ in range(epochs):
            state, action_tensor, _ , _ , _ = buffer.sample(batch_size)
            state = torch.tensor(state, dtype=torch.float32)
            # print(state.shape)
            # state = torch.tensor(states, dtype=torch.float32)
            # action_tensor = torch.tensor(actions, dtype=torch.long)
            # reward_tensor = torch.tensor(rewards, dtype=torch.float32)
            # next_state_tensor = torch.tensor(next_states, dtype=torch.float32)
            # mask_tensor = torch.tensor(masks, dtype=torch.float32)
            action_probs = policy_net(state)
            action_dist = Categorical(action_probs)
            new_log_probs = action_dist.log_prob(action_tensor)

            # print(len(new_log_probs) , len(log_probs))
            if len(new_log_probs) < len(log_probs) :
                log_probs_subset = log_probs[:len(new_log_probs)]
                ratio = torch.exp(new_log_probs - log_probs_subset)
                advantages = advantages[:len(new_log_probs)]
            else:
                new_log_probs = new_log_probs[:len(advantages)]
                log_probs = log_probs[:len(advantages)]
                ratio = torch.exp(new_log_probs - log_probs)

            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
            policy_loss = -torch.min(surrogate1, surrogate2).mean()
            policyloss.append(policy_loss)

            value_copy =  values.clone().detach()
            value_copy.requires_grad = True  # 確保 value_copy 需要計算梯度
            value_loss = F.mse_loss(value_copy, return_clone)
            valueloss.append(value_loss)
            print("haahhahahahahahahaaaaaaaaaahhhahahahhahaha")

            entropy_loss = -entropymean

            optimizer_policy.zero_grad()
            optimizer_value.zero_grad()
            policy_loss.backward()
            value_loss.backward()
            entropy_loss.backward(retain_graph=True)
            gradients = []
            for param in policy_net.parameters():
                gradients.append(param.grad.view(-1))

            # print(gradients)

            # 将所有梯度拼接成一个向量
            all_gradients = torch.cat(gradients)

            # 计算梯度范数
            gradient_norm = torch.norm(all_gradients, p=2)  # 使用p=2表示欧几里得范数

            print("Gradient Norm:", gradient_norm.item())
            # torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)  # 这里使用梯度范数裁剪
            optimizer_policy.step()
            optimizer_value.step()

        # 输出每个episode的奖励
        total_reward = sum(rewards)
        avg_reward = total_reward / len(rewards)
        print(f"Episode {episode + 1} - Total Reward: {total_reward}, Average Reward: {avg_reward}")
        print(f"Episode {episode + 1} - policy loss: {torch.mean(torch.stack(policyloss)).item():.2f}, value loss: {torch.mean(torch.stack(valueloss)).item():.2f}")
        # 保存模型參數
        torch.save(policy_net.state_dict(), "ppo_training_results/policy_net.pth")
        torch.save(value_net.state_dict(), "ppo_training_results/value_net.pth")

        with open('ppo_training_results/ori_training_results.txt', 'a') as f:
            f.write(f"Episode {episode + 1} - policy loss: {torch.mean(torch.stack(policyloss)).item():.2f}, value loss: {torch.mean(torch.stack(valueloss)).item():.2f}\n")

    print("win time : ", win)

    return policy_net, value_net

def calculate_returns(rewards, next_value, masks, gamma, lamda):
    returns = []
    for step in reversed(range(len(rewards))):
        temp = next_value[step]
        next_state_tensor = temp.clone().detach()
        next_state_tensor = next_state_tensor.to(torch.float32)  # Ensure the data type is float32 (if needed)
        next_values = value_net(next_state_tensor).detach()
        R = next_values
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32)

def calculate_advantages(rewards, values, next_value, masks, gamma, lamda):
    advantages = []
    for step in reversed(range(len(rewards)-1)):
        temp = next_value[step]
        next_state_tensor = temp.clone().detach()
        next_state_tensor = next_state_tensor.to(torch.float32)  # Ensure the data type is float32 (if needed)
        next_values = value_net(next_state_tensor).detach()
        R = next_values
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        R = delta + gamma * lamda * R * masks[step]
        advantages.append(R)
    advantages = torch.tensor(advantages, dtype=torch.float32)
    return advantages


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    
    state_dim = 90*4  # 棋盘狀態的维度
    feature_dim = 4   # 棋盘特徵的维度
    action_dim = 187  # 象棋的动作维度，假设有187个合法动作
    hidden_size = 512
    num_episodes = 100000  # 训练的episode数
    max_steps = 400  # 每个episode的最大步数
    epsilon = 0.2  # PPO算法的超参数，用于clip surrogate函数
    gamma = 0.99  # 折扣因子
    lamda = 0.95  # GAE-Lambda参数
    epochs = 10  # 每个更新步骤中PPO的循环次数
    batch_size = 24
    num_transformer_blocks = 1
    num_heads = 2

    # 创建策略网络和价值网络实例
    policy_net = PolicyNetwork(state_dim, feature_dim, action_dim, hidden_size, num_transformer_blocks, num_heads)
    value_net = ValueNetwork(state_dim,  feature_dim, hidden_size, num_transformer_blocks, num_heads)

    # 加載先前訓練的參數
    policy_net.load_state_dict(torch.load('ppo_training_results/policy_net.pth'))
    value_net.load_state_dict(torch.load('ppo_training_results/value_net.pth'))

    # 开始训练
    policy_net, value_net = ppo_train(env=None, policy_net=policy_net, value_net=value_net, num_episodes=num_episodes,
                                    max_steps=max_steps, epsilon=epsilon, gamma=gamma, lamda=lamda, epochs=epochs, batch_size=batch_size)
    
    # 假設已經訓練好的 policy_net 和 value_net
    # 保存模型參數
    torch.save(policy_net.state_dict(), "ppo_training_results/policy_net.pth")
    torch.save(value_net.state_dict(), "ppo_training_results/value_net.pth")

    # 調整神經網絡，優化PPOtrain下半部

# https://keras.io/examples/rl/ppo_cartpole/
# https://github.com/higgsfield/RL-Adventure-2/blob/master/3.ppo.ipynb
# https://github.com/openai/baselines
# https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_colab.ipynb
# https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html  
# https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
# https://ir.nctu.edu.tw/bitstream/11536/39543/1/559101.pdf