import os
import torch
from PPOTransformer import PolicyNetwork , ValueNetwork , Buffer , stateoripos ,  preprocess_state , convert_action , nextstate , justmove
from DQN import DQN , preprocess_state as pre , Buffer as B , combine , nextstate as nstate , validaction , get_move , justmove as jmove , stateoripos as stateori , mark
from torch.distributions import Categorical
import pygame
from pygame.locals import *
import pygame.mixer

from control import Chessboard

if not pygame.font: print('fonts disable')
if not pygame.mixer: print('mixer disable')

# 主路径
main_dir = os.path.split(os.path.abspath(__file__))[0]
# 资源文件路径
data_dir = os.path.join(main_dir, 'source')


# 創建policy_net和value_net實例
policy_net = PolicyNetwork(90 * 4, 4, 692, 128, 1, 2)
value_net = ValueNetwork(90 * 4, 4, 128, 1, 2)

# 加載先前訓練的參數
policy_net.load_state_dict(torch.load('ppo_training_results/policy_Transformer_net.pth'))
value_net.load_state_dict(torch.load('ppo_training_results/value_Transformer_net.pth'))

dqn_agent = DQN(90*4, 187, 192, 0.001, 0.99, 0.99)

# 載入已經儲存的模型狀態
saved_model_path = 'DQNmodel.pth'
checkpoint = torch.load(saved_model_path)
dqn_agent.q_network.load_state_dict(checkpoint['model_state_dict'])
dqn_agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# 設置為評估模式
dqn_agent.q_network.eval()

# 設置為評估模式
policy_net.eval()
value_net.eval()

red_win_count = 0
black_win_count = 0

def load_image(name):
    "loads an image, prepares it for play"
    filename = os.path.join(data_dir, name)
    try:
        surface = pygame.image.load(filename)
    except pygame.error:
        raise SystemExit('Could not load image "%s" %s'%(filename, pygame.get_error()))
    return surface.convert(), surface.get_rect()

def load_sound():
    class NoneSound:
        def play(self): pass

    if not pygame.mixer or not pygame.mixer.get_init():
        return NoneSound()
    pygame.mixer.music.load(os.path.join(data_dir, "music.mid"))

class ChessBoard(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        
        self.image, self.rect = load_image('boardchess.gif')
        self.image = pygame.transform.scale(self.image, (720, 800))
        self.rect = self.image.get_rect()


# 棋子类,position为棋子坐标
class Piece(pygame.sprite.Sprite):
    def __init__(self, name, position, *groups):
        pygame.sprite.Sprite.__init__(self)

        self.image, self.rect = load_image(name)

        self.position = position
        x, y = self.position
        # 通过改变矩形的中心来改变位置
        self.rect.center = (x * 80 + 40, y * 80 + 40)

    def move(self, position):
        x, y = position
        self.rect.center = (x * 80 + 40, y * 80 + 40)
        self.position = position

class arrowdown(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)

        original_image, rect = load_image('箭頭下.gif')
        scaled_image = pygame.transform.scale(original_image, (30, 40))
        self.image = scaled_image
        self.rect = self.image.get_rect()

        self.rect.x = x
        self.rect.y = y

class arrowup(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)

        original_image, rect = load_image('箭頭上.gif')
        scaled_image = pygame.transform.scale(original_image, (30, 40))
        self.image = scaled_image
        self.rect = self.image.get_rect()

        self.rect.x = x
        self.rect.y = y

# 鼠标相对于棋盘的位置
# 返回位置坐标x,y
def mouse_pos(position):
    x,y = position
    ex = x // 80
    ey = y // 80
    return ex,ey

def chess_main():
    pygame.init()
    load_sound()
    pygame.mixer.music.set_volume(0.2)  # 设置音量大小，可根据需要调整
    pygame.mixer.music.play(-1)  # -1表示无限循环播放

    screen = pygame.display.set_mode((720,800))
    pygame.display.set_caption('chess')

    clock = pygame.time.Clock()

    chessboard = ChessBoard()
    Arrowdown = arrowdown(0,380)
    Arrowup = arrowup(0,380)

    # init group
    Arrows = pygame.sprite.Group()
    board = pygame.sprite.Group()
    chesses = pygame.sprite.Group()

    board.add((chessboard, ))
    Arrows.add((Arrowdown, ))
    Arrows.add((Arrowup, ))
    show_arrow_down = True  # 初始显示箭头下
    clock = pygame.time.Clock()
    # add chess
    chess = Chessboard()
    for i in chess.chessboard:
        for y in i:
            if not y == 0:
                temp = Piece(y.picture(), y.position())
                chesses.add((temp,))

    red_box_pos = None
    box_width = 80
    box_height = 80
    going = True
    freq = 0
    # 一会要测试一下这里
    global start_pos
    global end_pos
    global red_win_count
    global black_win_count

    start_pos = end_pos = (-1,-1)
    while going:
        clock.tick(10)

        for event in pygame.event.get():
            if event.type == QUIT:
                going = False
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                going = False
            elif event.type == KEYDOWN:
                if event.key == pygame.K_s:
                    chess.save()
            elif event.type == MOUSEBUTTONDOWN:
                # 鼠标按下 記的position 怪
                pos = pygame.mouse.get_pos()
                end_pos = mouse_pos(pos)
                box_x = (end_pos[0] * 80 + 40) - box_width // 2
                box_y = (end_pos[1] * 80 + 40) - box_height // 2
                red_box_pos = (box_x, box_y)
                valid = chess.get_valid_moves()
                print(valid)
                # 沒步可走時
                if len(valid) == 0:
                    if show_arrow_down:
                        chess.black_win_count = 1
                    if not show_arrow_down:
                        chess.red_win_count = 1

                if chess.move(start_pos,end_pos):
                    for spr in chesses:
                        if spr.position == end_pos:
                            spr.kill()
                        if spr.position == start_pos:
                            spr.move(end_pos)
                    end_pos = start_pos = (-1,-1)
                    show_arrow_down = not show_arrow_down  # 切换显示状态
                    freq = freq + 1
                elif start_pos == (-1,-1):
                    start_pos = end_pos
                else:
                    end_pos = start_pos = (-1,-1)
                
                if start_pos == (-1, -1) and end_pos == (-1, -1):
                    if chess.red_win_count > 0 or chess.black_win_count > 0:
                        if chess.red_win_count > 0:
                            object = "red wins"
                        if chess.black_win_count > 0:
                            object = "black wins"
                        red_win_count = red_win_count + chess.get_red_win_count()
                        black_win_count = black_win_count + chess.get_black_win_count()
                        print("Red wins:", red_win_count)
                        print("Black wins:", black_win_count)
                        going = False
                        end_screen(object,red_win_count,black_win_count)  # 重新开始游戏
                        pygame.quit()
        if freq > 250:
            object = "It's tie!"
            going = False
            end_screen(object,red_win_count,black_win_count)  # 重新开始游戏
            pygame.quit()

        if going:
            board.draw(screen)
            chesses.draw(screen)

            if red_box_pos:
                for spr in chesses:
                    if spr.position == end_pos:
                        pygame.draw.rect(screen, (255, 0, 0), (red_box_pos[0], red_box_pos[1], box_width, box_height), 2)
                else:
                    pass

            if show_arrow_down:
                Arrowdown.rect.x = 0
                Arrowup.rect.x = -1000
                Arrows.draw(screen)  # 显示箭头下
            else:
                Arrowdown.rect.x = -1000
                Arrowup.rect.x = 0
                Arrows.draw(screen)  # 显示箭头上

            pygame.display.flip()

    pygame.quit()

def chess_mainai():
    pygame.init()
    load_sound()
    pygame.mixer.music.set_volume(0.2)  # 设置音量大小，可根据需要调整
    pygame.mixer.music.play(-1)  # -1表示无限循环播放

    screen = pygame.display.set_mode((720,800))
    pygame.display.set_caption('chess')

    clock = pygame.time.Clock()

    chessboard = ChessBoard()
    Arrowdown = arrowdown(0,380)
    Arrowup = arrowup(0,380)

    # init group
    Arrows = pygame.sprite.Group()
    board = pygame.sprite.Group()
    chesses = pygame.sprite.Group()

    board.add((chessboard, ))
    Arrows.add((Arrowdown, ))
    Arrows.add((Arrowup, ))
    show_arrow_down = True  # 初始显示箭头下
    clock = pygame.time.Clock()
    # add chess
    chess = Chessboard()
    for i in chess.chessboard:
        for y in i:
            if not y == 0:
                temp = Piece(y.picture(), y.position())
                chesses.add((temp,))

    red_box_pos = None
    box_width = 80
    box_height = 80
    going = True
    freq = 0
    # 一会要测试一下这里
    global start_pos
    global end_pos
    global red_win_count
    global black_win_count
    temparray =[]
    buffer = Buffer()

    start_pos = end_pos = (-1,-1)
    while going:
        clock.tick(10)

        if not show_arrow_down:
            pygame.time.delay(1000)
            state = preprocess_state(chess)
            poses = buffer.sampling()
            if len(poses) > 0:
                state = stateoripos(poses, state)
            state = torch.tensor(state, dtype=torch.float32)
            # state = state.view(-1)
            action_probs = policy_net(state)
            action_dist = Categorical(action_probs)
            action = action_dist.sample()
            action_str, ori_position, done, selected_indexes = convert_action(action_probs, chess)
            next_state, reward, done, actions, ori, action =  nextstate(action_str, ori_position, chess, temparray, selected_indexes)
            start_pos = ori
            end_pos = actions
            if start_pos ==  None and end_pos == None:
                start_pos = (-1, -1)
                end_pos = (-1, -1)
                chess.red_win_count = 1

            if chess.move(start_pos,end_pos):
                print(start_pos,end_pos)
                for spr in chesses:
                    if spr.position == end_pos:
                        spr.kill()
                    if spr.position == start_pos:
                        spr.move(end_pos)
                end_pos = start_pos = (-1,-1)
                show_arrow_down = not show_arrow_down  # 切换显示状态
                freq = freq + 1
                temparray = justmove(actions, chess, temparray)
            elif start_pos == (-1,-1):
                start_pos = end_pos
            else:
                end_pos = start_pos = (-1,-1)

            if start_pos == (-1, -1) and end_pos == (-1, -1):
                if len(action_str) == 0:
                    if chess.red_win_count > 0:
                        red_win_count = red_win_count + chess.get_red_win_count()
                        object = "red wins"
                        print("Red wins:", red_win_count)
                        going = False
                        end_screen(object,red_win_count,black_win_count)  # 重新开始游戏
                        pygame.quit()

        if show_arrow_down:
            for event in pygame.event.get():
                if event.type == QUIT:
                    going = False
                elif event.type == KEYDOWN and event.key == K_ESCAPE:
                    going = False
                elif event.type == KEYDOWN:
                    if event.key == pygame.K_s:
                        chess.save()
                elif event.type == MOUSEBUTTONDOWN:
                    # 鼠标按下 記的position 怪
                    pos = pygame.mouse.get_pos()
                    end_pos = mouse_pos(pos)
                    box_x = (end_pos[0] * 80 + 40) - box_width // 2
                    box_y = (end_pos[1] * 80 + 40) - box_height // 2
                    red_box_pos = (box_x, box_y)
                    valid = chess.get_valid_moves()
                    print(valid)
                    if len(valid) == 0:
                        chess.black_win_count = 1

                    if chess.move(start_pos,end_pos):
                        print(start_pos,end_pos)
                        for spr in chesses:
                            if spr.position == end_pos:
                                spr.kill()
                            if spr.position == start_pos:
                                spr.move(end_pos)
                        end_pos = start_pos = (-1,-1)
                        show_arrow_down = not show_arrow_down  # 切换显示状态
                        freq = freq + 1
                        # temparray = justmove(end_pos, chess, temparray)
                    elif start_pos == (-1,-1):
                        start_pos = end_pos
                    else:
                        end_pos = start_pos = (-1,-1)
                    
                    if start_pos == (-1, -1) and end_pos == (-1, -1):
                        if chess.red_win_count > 0 or chess.black_win_count > 0:
                            if chess.red_win_count > 0:
                                object = "red wins"
                            if chess.black_win_count > 0:
                                object = "black wins"
                            red_win_count = red_win_count + chess.get_red_win_count()
                            black_win_count = black_win_count + chess.get_black_win_count()
                            print("Red wins:", red_win_count)
                            print("Black wins:", black_win_count)
                            going = False
                            end_screen(object,red_win_count,black_win_count)  # 重新开始游戏
                            pygame.quit()
        if freq > 250:
            object = "It's tie!"
            going = False
            end_screen(object,red_win_count,black_win_count)  # 重新开始游戏
            pygame.quit()

        if going:
            board.draw(screen)
            chesses.draw(screen)

            if red_box_pos:
                for spr in chesses:
                    if spr.position == end_pos:
                        pygame.draw.rect(screen, (255, 0, 0), (red_box_pos[0], red_box_pos[1], box_width, box_height), 2)
                else:
                    pass

            if show_arrow_down:
                Arrowdown.rect.x = 0
                Arrowup.rect.x = -1000
                Arrows.draw(screen)  # 显示箭头下
            else:
                Arrowdown.rect.x = -1000
                Arrowup.rect.x = 0
                Arrows.draw(screen)  # 显示箭头上

            pygame.display.flip()

    pygame.quit()

def chess_mainaiai():
    pygame.init()
    # load_sound()
    # pygame.mixer.music.set_volume(0.2)  # 设置音量大小，可根据需要调整
    # pygame.mixer.music.play(-1)  # -1表示无限循环播放

    screen = pygame.display.set_mode((720,800))
    pygame.display.set_caption('chess')

    clock = pygame.time.Clock()

    chessboard = ChessBoard()
    Arrowdown = arrowdown(0,380)
    Arrowup = arrowup(0,380)

    # init group
    Arrows = pygame.sprite.Group()
    board = pygame.sprite.Group()
    chesses = pygame.sprite.Group()

    board.add((chessboard, ))
    Arrows.add((Arrowdown, ))
    Arrows.add((Arrowup, ))
    show_arrow_down = True  # 初始显示箭头下
    clock = pygame.time.Clock()
    # add chess
    chess = Chessboard()
    for i in chess.chessboard:
        for y in i:
            if not y == 0:
                temp = Piece(y.picture(), y.position())
                chesses.add((temp,))

    going = True
    freq = 0
    # 一会要测试一下这里
    global start_pos
    global end_pos
    global red_win_count
    global black_win_count
    temparray =[]
    temparray1 = []
    buffer = Buffer()
    buffer1 = B()

    start_pos = end_pos = (-1,-1)

    marking = mark(chess)
    
    while going:
        clock.tick(10)
        pygame.time.delay(10)

        if not show_arrow_down:
            Arrowdown.rect.x = -1000
            Arrowup.rect.x = 0
            Arrows.draw(screen)  # 显示箭头上
            pygame.display.flip()

        if not show_arrow_down:
            state = preprocess_state(chess)
            poses = buffer.sampling()
            if len(poses) > 0:
                state = stateoripos(poses, state)
            state = torch.tensor(state, dtype=torch.float32)
            # state = state.view(-1)
            action_probs = policy_net(state)
            action_dist = Categorical(action_probs)
            action = action_dist.sample()
            action_str, ori_position, done, selected_indexes = convert_action(action_probs, chess)
            next_state, reward, done, actions, ori, action =  nextstate(action_str, ori_position, chess, temparray, selected_indexes)
            start_pos = ori
            end_pos = actions
            if start_pos ==  None and end_pos == None:
                start_pos = (-1, -1)
                end_pos = (-1, -1)
                chess.red_win_count = 1

            if chess.move(start_pos,end_pos):
                print(start_pos,end_pos, "PPO")
                for spr in chesses:
                    if spr.position == end_pos:
                        spr.kill()
                    if spr.position == start_pos:
                        spr.move(end_pos)
                end_pos = start_pos = (-1,-1)
                show_arrow_down = not show_arrow_down  # 切换显示状态
                freq = freq + 1
                temparray = justmove(actions, chess, temparray)
            elif start_pos == (-1,-1):
                start_pos = end_pos
            else:
                end_pos = start_pos = (-1,-1)

            if start_pos == (-1, -1) and end_pos == (-1, -1):
                if len(action_str) == 0:
                    if chess.red_win_count > 0:
                        red_win_count = red_win_count + chess.get_red_win_count()
                        object = "red wins"
                        print("Red wins:", red_win_count)
                        going = False
                        end_screen(object,red_win_count,black_win_count)  # 重新开始游戏
                        pygame.quit()

        pygame.time.delay(10)


        if show_arrow_down:
            Arrowdown.rect.x = 0
            Arrowup.rect.x = -1000
            Arrows.draw(screen)  # 显示箭头下
            pygame.display.flip()

        if show_arrow_down:
            state = pre(chess)
            poses = buffer1.sampling()
            if len(poses) > 0:
                state = stateori(poses, state)
            valid_moves = chess.get_valid_moves()
            # if len(valid_moves) == 0:
            #     done = True
            #     break
            print(valid_moves)

            if len(valid_moves.keys()) > 0:
                valid_moves = combine(valid_moves, marking)
                valid_actions, str_poses, end_poses = validaction(valid_moves, chess)

                actions = dqn_agent.select_action(state, valid_actions)
                origin, action = get_move(actions, str_poses, end_poses, valid_actions)

                buffer1.addin(origin,action)

                start_pos = origin
                end_pos = action
                if start_pos ==  None and end_pos == None:
                    start_pos = (-1, -1)
                    end_pos = (-1, -1)
            else:
                start_pos = (-1, -1)
                end_pos = (-1, -1)
                chess.black_win_count = 1

            if chess.judgemove(start_pos,end_pos):
                # 執行遊戲行動，並取得下一個遊戲狀態、獎勵和結束標誌
                reward, done, marking = nstate(end_pos, start_pos, chess, marking, temparray1) 
                print(start_pos,end_pos,"DQN")
                for spr in chesses:
                    if spr.position == end_pos:
                        spr.kill()
                    if spr.position == start_pos:
                        spr.move(end_pos)
                end_pos = start_pos = (-1,-1)
                show_arrow_down = not show_arrow_down  # 切换显示状态
                freq = freq + 1
                temparray1 = jmove(action, chess, temparray1)
            elif start_pos == (-1,-1):
                start_pos = end_pos
            else:
                end_pos = start_pos = (-1,-1)

            if start_pos == (-1, -1) and end_pos == (-1, -1):
                if len(valid_moves) == 0:
                    if chess.black_win_count > 0:
                        print("挖屋")
                        pygame.time.delay(10000)
                        black_win_count = black_win_count + chess.get_black_win_count()
                        object = "black wins"
                        print("black wins:", black_win_count)
                        going = False
                        end_screen(object,red_win_count,black_win_count)  # 重新开始游戏
                        pygame.quit()
        if freq > 250:
            object = "It's tie!"
            going = False
            end_screen(object,red_win_count,black_win_count)  # 重新开始游戏
            pygame.quit()

        if going:
            board.draw(screen)
            chesses.draw(screen)

            pygame.display.flip()

    pygame.quit()

def start_screen(red_win_count,black_win_count):
    pygame.init()

    screen = pygame.display.set_mode((720, 800))
    pygame.display.set_caption('Chess')
    clock = pygame.time.Clock()

    string1 = "red wins     :  " + str(red_win_count)
    string2 = "black wins :  " + str(black_win_count)

    font = pygame.font.Font(None, 44)  # 创建字体对象
    font1 = pygame.font.Font(None, 24)
    number1 = font1.render(string1, True, (255, 255, 255))
    number2 = font1.render(string2, True, (255, 255, 255))
    number_rect1 = number1.get_rect(center=(70, 710))
    number_rect2 = number2.get_rect(center=(70, 750))
    text1 = font.render("h vs h", True, (255, 255, 255))
    text2 = font.render("h vs h", True, (0, 0, 0))
    text2.set_alpha(0)
    text3 = font.render("h vs AI", True, (255, 255, 255))
    text4 = font.render("h vs AI", True, (0, 0, 0))
    text4.set_alpha(0)
    text5 = font.render("AI vs AI", True, (255, 255, 255))
    text6 = font.render("AI vs AI", True, (0, 0, 0))
    text6.set_alpha(0)
    current_text2 = text5
    current_text1 = text4
    current_text = text1  # 当前显示的文本
    blink_timer = 0  # 闪烁计时器
    blink_interval = 500  # 闪烁间隔（毫秒）

    background_image , background_rect = load_image('start_background.jpg')
    background_image = pygame.transform.scale(background_image, (720, 800))

    running = True
    while running:
        dt = clock.tick(60) 
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:  # 检查鼠标左键点击
                    if text_rect.collidepoint(event.pos):  # 点击按钮
                        running = False  # 结束开始界面，进入游戏主循环
                        chess_main()
                        pygame.quit()
                if event.button == 1:  # 检查鼠标左键点击
                    if text_rect1.collidepoint(event.pos):  # 点击按钮
                        running = False  # 结束开始界面，进入游戏主循环
                        chess_mainai()
                        pygame.quit()
                if event.button == 1:  # 检查鼠标左键点击
                    if text_rect2.collidepoint(event.pos):  # 点击按钮
                        running = False  # 结束开始界面，进入游戏主循环
                        chess_mainaiai()
                        pygame.quit()
        if running:
            blink_timer += dt
            if blink_timer >= blink_interval:
                blink_timer = 0  # 重置计时器
                # 交替改变颜色
                if current_text == text1:
                    current_text2 = text6
                    current_text1 = text3
                    current_text = text2
                else:
                    current_text2 = text5
                    current_text1 = text4
                    current_text = text1

            screen.blit(background_image, (0, 0))
            text_rect = current_text.get_rect(center=(360, 600))
            text_rect1 = current_text1.get_rect(center=(360,520))
            text_rect2 = current_text2.get_rect(center=(360,440))
            screen.blit(current_text, text_rect)
            screen.blit(current_text1, text_rect1)
            screen.blit(current_text2, text_rect2)

            screen.blit(number1, number_rect1)
            screen.blit(number2, number_rect2)
            pygame.display.flip()

    pygame.quit()

def end_screen(object,red_win_count,black_win_count):
    pygame.init()

    screen = pygame.display.set_mode((720, 800))
    pygame.display.set_caption('Chess')

    font = pygame.font.Font(None, 80)  # 创建字体对象
    text = font.render(object, True, (255, 255, 255))

    background_image , background_rect = load_image('win.jpg')
    background_image = pygame.transform.scale(background_image, (720, 800))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

        if running:
            screen.blit(background_image, (0, 0))
            text_rect = text.get_rect(center=(360, 200))
            screen.blit(text, text_rect)
            pygame.display.flip()
            pygame.time.delay(5000)  # 延迟50豪秒
        running = False  # 结束开始界面，进入游戏主循环
        start_screen(red_win_count,black_win_count)
        pygame.quit()

    pygame.quit()

if __name__ == '__main__':
    start_screen(red_win_count,black_win_count)


    # https://github.com/suragnair/alpha-zero-general
    # https://github.com/gavindst/chess
    # https://github.com/mm12432/MyChess
    # https://github.com/PacktPublishing/Reinforcement-Learning-Algorithms-with-Python/blob/master/Chapter05/DQN_Atari.py