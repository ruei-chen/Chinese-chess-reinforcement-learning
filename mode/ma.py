from mode.chess import Chess

class Ma(Chess):
    # 如果使用父类的构造函数,这里会自动调用不需要重写
    def __init__(self, x, y, red=True, selected=False):
        super(Ma, self).__init__(x, y, red=True, selected=False)
        self.name = '馬'
        self.red = red

    def picture(self):
        if self.red:
            pic = self.redpic()
        else:
            pic = self.blackpic()
        return pic

    def redpic(self):
        pic = 'red_knight.gif'
        return pic
    def blackpic(self):
        pic = 'black_knight.gif'
        return pic

    def rule(self, dx, dy):
        if dx == 0 or dy == 0:
            return False
        if abs(dx) + abs(dy) == 3:
            return True
        
    def can_move(self, start_position, end_position, chessboard):
        dx = end_position[0] - start_position[0]
        dy = end_position[1] - start_position[1]
        x = end_position[0]
        y = end_position[1]
        if self.rule(dx, dy):
            # 移动
            if not self.position_has_chess(end_position, chessboard):
                if self.mati(start_position, dx, dy, chessboard):
                    return False
                return True
            else:
                # 吃子
                # 是否本方棋子
                if self.chess_is_my(self.red, chessboard[x][y]):
                    return False
                else:
                    # 吃子
                    if self.mati(start_position, dx, dy, chessboard):
                        return False
                    return True
        else:
            return False
    # 是否绊马蹄
    # 是 返回 True
    def mati(self, start_position, dx, dy, chessboard):
        x,y = start_position
        if abs(dx) == 1:
            y = y + dy/2
            y = int(y)
            if chessboard[x][y] == 0:
                return False
            else:
                return True
        if abs(dy) == 1:
            x = x + dx/2
            x = int(x)
            if chessboard[x][y] == 0:
                return False
            else:
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
        a = [(x+1,y+2),(x+2,y+1),(x-1,y+2),(x-2,y+1),
                (x-1,y-2),(x-2,y-1),(x+1,y-2),(x+2,y-1)
            ]
        b = []

        for i in a:
            if not self.checkp(i):
                continue
            if self.can_move(position, i, chessboard):
                b.append(i)
        return b
