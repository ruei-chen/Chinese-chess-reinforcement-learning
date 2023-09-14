class Chess():
    """
    棋子的基类
    """

    def __init__(self, x, y, red=True, selected=False):
        #棋子是否选中
        self.selected = selected 
        # 黑方或者红方,红方为True
        self.red = red
        self.x = x
        self.y = y
        # 是否在自己的地盘
        self.bool_myplace = self.is_myplace(self.y)
        
        self.name = '棋'

    #棋子的图片
    # def picture(self):
    #     # 返回棋子对应的图片名字
    #     name = "RS.gif"
    #     return name

    # 楚河汉界,判断是否在自己地盘
    # 在返回 True
    def is_myplace(self, y):
        if self.red and y > 4:
            return True
        elif not self.red and y < 5:
            return True
        return False

    # 返回棋子的颜色
    def is_red(self):
        red = self.red
        return red

    #返回棋子的位置==列表的索引
    def position(self):
        pos = (self.x,self.y)
        #返回元组
        return pos

    # 设置棋子位置
    def set_position(self, pos):
        self.x, self.y = pos

    # 棋子移动
    def move(self, start_position, end_position, chessboard):
        # 棋子能否移动
        # 更新棋子状态
        if self.can_move(start_position, end_position, chessboard):
            # 棋子位置,选中,是否在自己的地盘
            self.x = end_position[0]
            self.y = end_position[1]
            self.selected = False
            self.bool_myplace = self.is_myplace(self.x)
            # 先不更新chessboard,因为这是棋子,不要加进棋盘
            return True
        return False
            

    #棋子的走法及相应的规则
    # dx, dy为相对于之前位置的位移量
    def rule(self, dx, dy):
        return False

    #棋子是否能够移动到相应位置
    # chessboard是棋盘的数组
    # position为元组,chessboard为列表
    def can_move(self, start_position, end_position, chessboard):
        # 先调用rule看看能不能这么走,不能直接返回false
        # 判断落点有无棋子,是否己方,能否吃子
        # 判断能否到达落点
        #能移动则返回True
        return True

    # 判断落点有无棋子
    # 有返回True
    def position_has_chess(self, position, chessboard):
        ax = position[0]
        ay = position[1]

        if chessboard[ax][ay] == 0:
            # 对应点无棋子
            return False
        return True

    # 判断棋子是否是己方的
    # 是则返回True
    # red 己方棋子的颜色
    # chess 要判断的棋子对象
    # 是同一方的棋子返回True
    def chess_is_my(self, red, chess):
        black = chess.is_red()
        
        if red == black:
            # 棋子颜色相同,是己方的棋子
            return True
        return False

    # 能否移动到落点
    def move_position(self, start_position, end_position, chessboard):
        return False

    # 计算起点和终点直线之间的棋子,只适用于炮和車
    def count_chess(self, start_position, end_position ,chessboard):
        x = start_position[0]
        y = start_position[1]
        ex = end_position[0]
        ey = end_position[1]

        dx = end_position[0] - x
        dy = end_position[1] - y

        # 方向
        sx = dx/abs(dx) if dx != 0 else 0
        sy = dy/abs(dy) if dy != 0 else 0
        
        num = 0
        while x != ex or y != ey:
            x += sx
            y += sy
            x = int(x)
            y = int(y)
            if not chessboard[x][y] == 0:
                num += 1
        return num
