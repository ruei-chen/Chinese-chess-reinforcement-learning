from mode.chess import Chess

class Bing(Chess):
    # 如果使用父类的构造函数,这里会自动调用不需要重写
    def __init__(self, x, y, red=True):
        super(Bing, self).__init__(x, y, red=True, selected=False)
        self.name = '兵'
        self.red = red

    def picture(self):
        if self.red:
            pic = self.redpic()
        else:
            pic = self.blackpic()
        return pic

    def redpic(self):
        pic = 'red_pawn.gif'
        return pic
    def blackpic(self):
        pic = 'black_pawn.gif'
        return pic

    # 兵的规则要改一下,要考虑上下两方的坐标
    # 改了一下不知道对不对
    def rule(self, dx, dy):
        # 把它想成坐标系
        if not abs(dx) + abs(dy) == 1:
            return False

        self.bool_myplace = self.is_myplace(self.y)
        # 判断兵是否到达对面
        if self.bool_myplace:
            # 未到
            if self.red and dy == -1:
                return True
            if not self.red and dy == 1:
                return True
            return False
        else:
            if self.red and dy == 1:
                return False
            if not self.red and dy == -1:
                return False
            return True

    def can_move(self, start_position, end_position, chessboard):
        x,y = end_position
        dx = end_position[0] - start_position[0]
        dy = end_position[1] - start_position[1]
        if self.rule(dx, dy):
            if not self.position_has_chess(end_position, chessboard):
                return True
            else:
                # 是否本方棋子
                if self.chess_is_my(self.red, chessboard[x][y]):
                    return False
                else:
                    # 吃子
                    return True
        else:
            return False

    # 检查是否越界,越界则返回False
    def checkp(self, pos):
        x,y = pos
        if x < 0 or y < 0 or x > 8 or y > 9:
            return False
        return True
    # 返回棋子能够移动的所有位置
    # position 棋子位置, chessboard棋盘二维列表
    # 返回列表,列表元素为元组
    def try_move(self, position, chessboard):
        x,y = position
        a = [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
        b = []

        for i in a:
            if not self.checkp(i):
                continue
            if self.can_move(position, i, chessboard):
                b.append(i)
        
        return b
