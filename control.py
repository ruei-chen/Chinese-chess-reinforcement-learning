import json
import copy

from mode.bing import Bing
from mode.pao import Pao
from mode.che import Che
from mode.ma import Ma
from mode.xiang import Xiang
from mode.shi import Shi
from mode.shuai import Shuai

class Chessboard():

    def __init__(self):
        # 初始化
        board = [0] * 9
        for i in range(9):
            board[i] = [0] * 10
        self.chessboard = board
        # 初始化棋盘
        # 黑棋
        # x为横坐标,y为纵坐标,图像的左上角为坐标原点

        self.chessboard[0][3] = Bing(0, 3, red = False) 
        self.chessboard[2][3] = Bing(2, 3, red = False) 
        self.chessboard[4][3] = Bing(4, 3, red = False) 
        self.chessboard[6][3] = Bing(6, 3, red = False) 
        self.chessboard[8][3] = Bing(8, 3, red = False) 
        self.chessboard[1][2] = Pao(1, 2, red = False) 
        self.chessboard[7][2] = Pao(7, 2, red = False) 
        self.chessboard[0][0] = Che(0, 0, red = False) 
        self.chessboard[8][0] = Che(8, 0, red = False) 
        self.chessboard[1][0] = Ma(1, 0, red = False) 
        self.chessboard[7][0] = Ma(7, 0, red = False) 
        self.chessboard[2][0] = Xiang(2, 0, red = False) 
        self.chessboard[6][0] = Xiang(6, 0, red = False) 
        self.chessboard[3][0] = Shi(3, 0, red = False) 
        self.chessboard[5][0] = Shi(5, 0, red = False) 
        self.chessboard[4][0] = Shuai(4, 0, red = False) 

        # # 红棋
        self.chessboard[0][6] = Bing(0, 6) 
        self.chessboard[2][6] = Bing(2, 6) 
        self.chessboard[4][6] = Bing(4, 6) 
        self.chessboard[6][6] = Bing(6, 6) 
        self.chessboard[8][6] = Bing(8, 6) 
        self.chessboard[1][7] = Pao(1, 7) 
        self.chessboard[7][7] = Pao(7, 7) 
        self.chessboard[0][9] = Che(0, 9) 
        self.chessboard[8][9] = Che(8, 9) 
        self.chessboard[1][9] = Ma(1, 9) 
        self.chessboard[7][9] = Ma(7, 9) 
        self.chessboard[2][9] = Xiang(2, 9) 
        self.chessboard[6][9] = Xiang(6, 9) 
        self.chessboard[3][9] = Shi(3, 9) 
        self.chessboard[5][9] = Shi(5, 9) 
        self.chessboard[4][9] = Shuai(4, 9) 

        self.red_move = True
        self.black_move = False
        self.red_win_count = 0
        self.black_win_count = 0
        self.game_over = False

        # 将军的标志
        self.jiangjun = False
        # 记录走子
        self.memory = []

    # 判断坐标是否合法
    def check(self, position):
        x,y = position
        if x < 0 or y < 0 or x > 8 or y > 9:
            return False
        return True
    
    def value(self,x,y):
        temp = copy.deepcopy(self.chessboard[x][y])
        return temp
    
    def get_valid_moves(self):
        valid_moves = {}  # 创建一个空的valid_moves字典
        for i in self.chessboard:
            for y in i:
                if y == 0: continue
                trymove = y.try_move(y.position(), self.chessboard)

                # 直接将新键值对添加到字典中，如果键不存在，则会自动创建
                if y.red == True:
                    valid_moves[(1, y.position())] = trymove
                else:
                    valid_moves[(-1, y.position())] = trymove

        # print(valid_moves)

        # 删除 valid_moves 字典中的特定元组
        for key, value in valid_moves.items():
            valid_moves[key] = [tup for tup in value if self.judgemove(key[1],tup)]

        valid_move = {key: value for key, value in valid_moves.items() if value}

        # print(valid_move)
        return valid_move
    
    def judgemove(self, start_position, end_position):  # 帥和將的判別
        x,y = start_position
        ex,ey = end_position
        temp = self.value(x,y)
        tempe = self.value(ex,ey)
        value = True

        if not self.check(start_position):
            return False
        if not self.check(end_position):
            return False

        # 要走的位置没棋子
        if self.chessboard[x][y] == 0:
            return False

        # 添加走棋规则,一人一步,红方先走 黑紅交替有問題 在沒被將軍時
        if self.red_move and not self.chessboard[x][y].red:
            self.chessboard[x][y] = copy.deepcopy(temp)
            self.chessboard[ex][ey] = copy.deepcopy(tempe)
            return False 
        if self.black_move and self.chessboard[x][y].red:
            self.chessboard[x][y] = copy.deepcopy(temp)
            self.chessboard[ex][ey] = copy.deepcopy(tempe)
            return False 
        self.red_move = not self.red_move
        self.black_move = not self.black_move

        self.chessboard[ex][ey] = self.chessboard[x][y]
        self.chessboard[ex][ey].set_position(end_position)
        self.chessboard[x][y] = 0

            # 添加判断,走棋方走棋后是否被将军,是则不能走棋 !!!!!!!!!!!
            # if not self.jiangjun:
        if self.is_jiang(temp.red, self.chessboard):
            # self.chessboard[x][y] = copy.deepcopy(temp)
            # self.chessboard[ex][ey] = copy.deepcopy(tempe)
            # self.red_move = not self.red_move
            # self.black_move = not self.black_move
            print('被將軍')
            value = False
        
        # print(self.chessboard[ex][ey].name,self.chessboard[ex][ey].position())        
        self.chessboard[x][y] = copy.deepcopy(temp)
        self.chessboard[ex][ey] = copy.deepcopy(tempe)
        self.red_move = not self.red_move
        self.black_move = not self.black_move
        # print("幹",temp.position())

        return value

    def move(self, start_position, end_position):
        x,y = start_position
        ex,ey = end_position
        temp = self.value(x,y)
        tempe = self.value(ex,ey)

        if not self.check(start_position):
            return False
        if not self.check(end_position):
            return False

        # 要走的位置没棋子
        if self.chessboard[x][y] == 0:
            return False

        if not self.chessboard[x][y].move(start_position, end_position, self.chessboard):
            return False
        # 添加走棋规则,一人一步,红方先走 黑紅交替有問題 在沒被將軍時
        if self.red_move and not self.chessboard[x][y].red:
            self.chessboard[x][y] = copy.deepcopy(temp)
            self.chessboard[ex][ey] = copy.deepcopy(tempe)
            return False 
        if self.black_move and self.chessboard[x][y].red:
            self.chessboard[x][y] = copy.deepcopy(temp)
            self.chessboard[ex][ey] = copy.deepcopy(tempe)
            return False 
        self.red_move = not self.red_move
        self.black_move = not self.black_move

        self.chessboard[ex][ey] = self.chessboard[x][y]
        self.chessboard[x][y] = 0

            
            # 添加判断,走棋方走棋后是否被将军,是则不能走棋 !!!!!!!!!!!
            # if not self.jiangjun:
        if self.is_jiang(temp.red, self.chessboard):
            self.chessboard[x][y] = copy.deepcopy(temp)
            self.chessboard[ex][ey] = copy.deepcopy(tempe)
            self.red_move = not self.red_move
            self.black_move = not self.black_move
            print('被將軍')
            return False
                
            # 添加判断,走棋方走棋后是否将对面军,是则更改将军标志
        if self.is_jiang((not temp.red), self.chessboard):
            self.jiangjun = True
            print('將軍')
            if self.is_win(temp.red , self.chessboard):
                self.game_over = True
                if self.red_move:
                    color = 0
                    self.restart_chess(color)
                else:
                    color = 1
                    self.restart_chess(color)
        else:
            self.jiangjun = False

        # 保存走子
        self.history(start_position,end_position)
        return True

    def set_selected(self, position):
        x,y = position
        self.chessboard[x][y].set_selected(True)
    def rm_selected(self, position):
        x,y = position
        self.chessboard[x][y].set_selected(False)

    # 判断死棋
    # 判断胜利
    # red 为应将方
    # 应该成功返回True
    def is_win(self, red, chessboard):
        # 没有想到好方法,只能遍历己方所有棋子的所有走法,看能否应将
        # 存储己方所有棋子
        chesses = []
        # 应将成功标志
        yy = True

        for i in self.chessboard:
            for y in i:
                if y == 0: 
                    continue
                if y.red == (not red): 
                    chesses.append(y)
        break_flag = False

        # for i in chesses:
        #     # 所有的走法
        #     a = []
        #     x1 , y1 = i.position()
        #     temp = chessboard[x1][y1]
        #     a = i.try_move(i.position(), chessboard)
        #     for y in a:
        #         tempe = chessboard[y[0]][y[1]]
        #         chessboard[y[0]][y[1]] = chessboard[x1][y1]
        #         chessboard[x1][y1] = 0
        #         if not self.is_jiang(temp.red, chessboard):
        #             yy = False
        #             chessboard[x1][y1] = copy.deepcopy(temp)
        #             chessboard[y[0]][y[1]] = copy.deepcopy(tempe)
        #             break_flag = True
        #             break
        #         else:
        #             chessboard[x1][y1] = copy.deepcopy(temp)
        #             chessboard[y[0]][y[1]] = copy.deepcopy(tempe)

        for i in chesses:
            # 所有的走法
            a = []
            a = i.try_move(i.position(), chessboard)
            for y in a:
                # if i.can_move(i.position(), y, self.chessboard):
                fa,fb = i.position()
                ea, eb = y

                temp = self.chessboard[fa][fb]
                tempe = self.chessboard[ea][eb]

                # 改变棋子的位置属性,棋盘中棋子位置
                i.set_position(y)
                self.chessboard[ea][eb] = self.chessboard[fa][fb]
                self.chessboard[fa][fb] = 0
        
                if not self.is_jiang(temp.red, self.chessboard):
                    yy = False
                    break_flag = True
                        # 应将成功

                # 回滚
                self.chessboard[ea][eb].set_position((fa,fb))
                self.chessboard[fa][fb] = temp
                self.chessboard[ea][eb] = tempe

                if break_flag:
                    break

            if break_flag:
                break
                    # if yy: break
        print(yy)
        return yy
                
    # 判断将军,每一步走棋都可能会将军,无论是本方还是对方
    # red 为被将军的那一方
    # 被将军返回True
    def is_jiang(self, red, chessboard):
        # 只需要判断走棋后,本方棋子下一步能不能走到对方将的位置
        # 将两个位置输入到对应棋子的move函数判断
        a = []
        b = []
        c = []
        found_shuai = False
        for i in chessboard:
            if found_shuai:
                break
            for y in i:
                if y == 0:
                    continue
                if y.name == '帥':
                    found_shuai = True
                if found_shuai == True:
                    c.append(y)

        if len(c) > 1:
            if c[0].name == c[1].name:
                return True

        for i in chessboard:
            for y in i:
                if y == 0: continue
                if y.name == '帥' and y.red == red:
                    a.append(y)
                    continue
                if y.name == '炮' and y.red == (not red):
                    b.append(y)
                    continue
                if y.name == '馬' and y.red == (not red):
                    b.append(y)
                    continue
                if y.name == '車' and y.red == (not red):
                    b.append(y)
                    continue
                if y.name == '兵' and y.red == (not red):
                    b.append(y)
                    continue
        if len(a) != 0:
            x,y = a[0].position()
            # print('---------------------------------')
            # print(a[0].position(), a[0].name)
            # # if self.check((x,y+1)) and not chessboard[x][y+1] == 0:
            # #     if chessboard[x][y+1].can_move((x,y+1), (x,y), chessboard):
            # #         return True
            # # if self.check((x,y-1)) and not chessboard[x][y-1] == 0:
            # #     if chessboard[x][y-1].can_move((x,y-1), (x,y), chessboard):
            # #         return True
            # # if self.check((x+1,y)) and not chessboard[x+1][y] == 0:
            # #     if chessboard[x+1][y].can_move((x+1,y), (x,y), chessboard):
            # #         return True
            # # if self.check((x-1,y)) and not chessboard[x-1][y] == 0:
            # #     if chessboard[x-1][y].can_move((x-1,y), (x,y), chessboard):
            # #         return True
        
            # print('---------------------------------')
            for i in b:
                # 结果将棋子自身属性位置改变成了对方将的位置
                # 棋子在棋盘上的位置没变,所以移动吃子没有问题
                # print(i.position(), i.name)
                if i.can_move(i.position(), (x,y), chessboard): 
                    return True

        return False
    

    # 保存历史记录用来悔棋
    def history(self,start_position, end_position):
        # 保存移动前的位置
        # 移动后的位置
        # 如果移动后的位置有棋子,保存棋子
        # 可选保存棋子名称

        # 以上所有保存在元组中,所有元组保存在列表中
        # [(x,y),(x,y)]
        c = []
        c.append(start_position)
        c.append(end_position)
        self.memory.append(c)
    
    # 保存记录
    def save(self):
        with open('memory','w') as f:
            json.dump(self.memory, f)
        

    # 重新开始游戏
    def restart_chess(self,color):
        if color == 0:
            print("black win")
            self.black_win_count = self.black_win_count + 1
        if color == 1:
            print("red win")
            self.red_win_count = self.red_win_count + 1
        if color != 0 and color != 1:
            print("ended in a tie")
    
    def get_red_win_count(self):
        return self.red_win_count

    def get_black_win_count(self):
        return self.black_win_count
    
    def is_game_over(self):
        return self.game_over

    # 设置对弈模式,人人,人机,AI会有的
    def setting_mode(self):
        pass