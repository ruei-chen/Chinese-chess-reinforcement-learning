from mode.chess import Chess

class Pao(Chess):
    # 如果使用父类的构造函数,这里会自动调用不需要重写
    def __init__(self, x, y, red=True, selected=False):
        super(Pao, self).__init__(x, y, red=True, selected=False)
        self.name = '炮'
        self.red = red

    def picture(self):
        if self.red:
            pic = self.redpic()
        else:
            pic = self.blackpic()
        return pic

    def redpic(self):
        pic = 'red_cannon.gif'
        return pic
    def blackpic(self):
        pic = 'black_cannon.gif'
        return pic


    def rule(self, dx, dy):
        if not (dx == 0 and dy == 0):
            if dx == 0 or dy == 0:
                return True
        return False
        
    def can_move(self, start_position, end_position, chessboard):
        dx = end_position[0] - start_position[0]
        dy = end_position[1] - start_position[1]
        x = end_position[0]
        y = end_position[1]
        if self.rule(dx, dy):
            # 炮移动
            num = self.count_chess(start_position, end_position, chessboard)
            if not num:
                return True
            # 炮吃子
            if num == 2 and self.position_has_chess(end_position, chessboard):
                if not self.chess_is_my(self.red, chessboard[x][y]):
                    return True
            return False

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
        a = [(x+1,y),(x+2,y),(x+3,y),(x+4,y),(x+5,y),(x+6,y),(x+7,y),(x+8,y),
                (x-1,y),(x-2,y),(x-3,y),(x-4,y),(x-5,y),(x-6,y),(x-7,y),(x-8,y),
                (x,y+1),(x,y+2),(x,y+3),(x,y+4),(x,y+5),(x,y+6),(x,y+7),(x,y+8),(x,y+9),
                (x,y-1),(x,y-2),(x,y-3),(x,y-4),(x,y-5),(x,y-6),(x,y-7),(x,y-8),(x,y-9)
            ]
        b = []

        for i in a:
            if not self.checkp(i):
                continue
            if self.can_move(position, i, chessboard):
                b.append(i)
        return b
