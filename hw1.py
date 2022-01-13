import numpy as np

import scipy.optimize as op


class Solver:
    def __init__(self, m):
        """
        随机初始化m个点,3为参数
        """
        self.p = None
        self.y = None
        self.m = m
        self.x = np.random.normal(size=(m, 3))
        self.c = np.ones(m)
        #     这是min的参数,eye是对角矩阵
        self.A_ub = np.ones((m, m)) - np.identity(m) * m
        #     这是构建不等式约束的参数，对角都是1-m
        self.B_ub = []
        self.res = None

    def f(self):
        """
        函数 = x1**2-x2**2+x3**2
        :param X:
        :return:
        """
        y = np.square(self.x)
        y[1] *= -1

        self.y = np.sum(y)

    def jac(self):
        """
        返回x的梯度，手动计算梯度
        :param x:
        :return:

        """
        y = np.empty_like(self.x)
        y[0] = 2 * self.x[0]
        y[1] = 2 * self.x[1]
        y[2] = 2 * self.x[2]
        self.p = y*-1

    def B_c(self):
        """
        进行不等式右边的参数构建,首先是二阶范式，计算每个bi，bi形式都是一样的，就是中心乘上1-m
        axis=-1 == ndims-1维度-1

        :return:
        """
        for i in range(self.m):
            # print(np.square(self.x[i:i + 1] - self.p))
            b = np.sqrt(np.sum(np.square(self.x[i:i + 1] - self.p), axis=-1))
            # print(b)
            b[i] *= (1 - self.m)
            # 最后进行求和
            self.B_ub.append(np.sum(b))
        self.B_ub = np.array(self.B_ub)

    def ans(self):
        self.res = op.linprog(self.c, A_ub=self.A_ub, b_ub=self.B_ub, bounds=[0, None], method='revised simplex')

    def run(self):
        self.f()
        self.jac()
        # print(self.x[1:2])
        self.B_c()
        self.ans()
        print(self.res)


if __name__ == '__main__':
    so = Solver(100)
    so.run()
