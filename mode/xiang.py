from mode.chess import Chess

class Xiang(Chess):
    # 如果使用父类的构造函数,这里会自动调用不需要重写
    def __init__(self, x, y, red=True, selected=False):
        super(Xiang, self).__init__(x, y, red=True, selected=False)
        self.name = '象'
        self.red = red

    def picture(self):
        if self.red:
            pic = self.redpic()
        else:
            pic = self.blackpic()
        return pic

    def redpic(self):
        pic = 'red_elephant.gif'
        return pic
    def blackpic(self):
        pic = 'black_elephant.gif'
        # if self.selected:
        #     return pic
        # else:
        return pic

    def rule(self, dx, dy):
        if abs(dx) == 2 and abs(dy) == 2:
            return True
        return False

    def can_move(self, start_position, end_position, chessboard):
        x, y = end_position
        dx = end_position[0] - start_position[0]
        dy = end_position[1] - start_position[1]
        if self.rule(dx, dy):
            # 移动
            if not self.position_has_chess(end_position, chessboard):
                if self.xiangti(start_position, dx, dy, chessboard):
                    return False
                if not self.is_out_range(y):
                    return False
                return True
            else:
                # 吃子
                # 是否本方棋子
                if self.chess_is_my(self.red, chessboard[x][y]):
                    return False
                else:
                    # 吃子
                    if self.xiangti(start_position, dx, dy, chessboard):
                        return False
                    if not self.is_out_range(y):
                        return False
                    return True
        else:
            return False
    # 是否绊象蹄
    # 是 返回 True
    def xiangti(self, start_position, dx, dy, chessboard):
        x, y = start_position
        x += dx/2
        y += dy/2
        x = int(x)
        y = int(y)
        if chessboard[x][y] == 0:
            return False
        return True

    # 判断是否过界
    # 过界 返回 False
    def is_out_range(self, y):
        if self.red and y < 5:
            # out range
            return False
        if not self.red and y > 4:
            return False
        return True

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
        a = [(x+2,y+2),(x-2,y+2),(x-2,y-2),(x+2,y-2)]
        b = []

        for i in a:
            if not self.checkp(i):
                continue
            if self.can_move(position, i, chessboard):
                b.append(i)
        return b
