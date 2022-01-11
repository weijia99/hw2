from PIL import Image
import numpy as np
# import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

# sk 进行线性回归
from scipy.optimize import root


# 优化包
class solver():
    def __init__(self, m, path):
        self.lo = 0
        self.m = m
        self.path = path
        # 放入随机生成的点云
        self.lists = []
        self.rec = {}
        #     记录每个点，所包含在那个领域
        self.superface = {}

        self.edge = {}

    # 1.读取文件，创建数组,并且灰度化文件
    def read_image(self):

        # I.show()
        # I.save('./save.png')
        I_array = np.array(Image.open(self.path).convert('L'), 'f')
        print(I_array.shape)
        self.I_array = I_array
        # I =Image.fromarray(np.uint8(I_array))
        # I.show()

    def create(self):
        '''
        随机生成m个点云
        :return:
        '''
        for i in range(self.m):
            self.lists.append((0, i))

    def init_m(self):
        """
        对m个点的所有值，初始化
        :return:
        """
        for i in range(self.m):
            point = self.lists[i]

            self.rec[point] = []
            #         初始化是01 02 03 0 4，使用set可以去重
            self.edge[point] = {}

    def cut(self):
        '''
        分割图像，到m个点上
        :return:
        '''
        for i in range(self.I_array.shape[0]):
            for j in range(self.I_array.shape[1]):
                # 设置第一个是最小的，下面进行更新
                minP = (self.lists[0][0] - i) ** 2 + (self.lists[0][1] - j) ** 2
                x = 0
                y = 0

                for point in self.lists:
                    if (point[0] - i) ** 2 + (point[1] - j) ** 2 < minP:
                        minP = (point[0] - i) ** 2 + (point[1] - j) ** 2
                        x = point[0]
                        y = point[1]
                # 这是第一轮，最小的，看还有没有一样小的
                self.rec[(x, y)].append(([i, j]))

                for point in self.lists:
                    if (point[0] - i) ** 2 + (point[1] - j) ** 2 == minP:
                        if point[0] != x or point[1] != y:
                            self.rec[point].append(([i, j]))

    def find_edge(self):
        """
        找出当前点的附近点
        :return:
        """
        for i in range(self.m):
            point = self.lists[i]
            x = self.rec[point]
            # 恶臭代码。当前值往四处走，看是不是在其他领域
            for index in x:
                index_1 = (index[0] + 1, index[1])
                index_2 = (index[0] - 1, index[1])
                index_3 = (index[0], index[1] + 1)
                index_4 = (index[0], index[1] - 1)

                for j in range(self.m):
                    if index_1 in self.rec[self.lists[j]] and i != j:
                        self.edge[point].add(self.lists[j])
                    elif index_2 in self.rec[self.lists[j]] and i != j:
                        self.edge[point].add(self.lists[j])
                    elif index_3 in self.rec[self.lists[j]] and i != j:
                        self.edge[point].add(self.lists[j])
                    elif index_4 in self.rec[self.lists[j]] and i != j:
                        self.edge[point].add(self.lists[j])

    def least_square(self):
        for i in range(self.m):
            point = self.lists[i]
            x = self.rec[point]
            y = []
            for index in x:
                y.append(self.I_array[index[0], index[1]])
            #         加入到numpy

            x = np.array(x)
            # x归一化
            # x[0]=(x[0]-x.shape[0])/x.shape[0]
            # x[1]=(x[1]-x.shape[1])/x.shape[1]
            # print(x.shape)
            y = np.array(y)
            # 将 y 分别增加一个轴，以满足 sklearn 中回归模型认可的数据
            # 此时由于 x 是多元变量，则不用添加新的轴了
            y = y[:, np.newaxis]
            # y = (y[:, np.newaxis]-128)/128

            model = LinearRegression()  # 构建线性模型
            model.fit(x, y)  # 自变量在前，因变量在后
            predicts = model.predict(x)  # 预测值
            R2 = model.score(x, y)  # 拟合程度 R2
            print('R2 = %.3f' % R2)  # 输出 R2
            coef = model.coef_  # 斜率
            intercept = model.intercept_  # 截距
            #         w =np.dot(np.dot(np.linalg.inv(np.dot(x.T,x)),x.T),y)
            #
            # #         这是w的参数
            #         进行水平拼接
            intercept = np.array(intercept).reshape((-1, 1))
            # print(model.coef_, intercept)  # 输出斜率和截距

            self.superface[point] = np.hstack((coef, intercept)).reshape((-1, 1))

    def loss(self):
        """
        计算loss，通过与真实值
        # todo：
        :return:
        """
        total = 0.0
        # 找出所有的点
        for i in range(self.m):
            point = self.lists[i]
            x = self.rec[point]
            y = []
            for index in x:
                y.append(self.I_array[index[0], index[1]])
            #         加入到numpy

            x = np.array(x)
            one = np.ones((x.shape[0], 1))
            x = np.hstack((x, one))
            # 通过dot进行计算
            y = np.array(y)
            y_pre = np.dot(x, self.superface[point])
            lo = (y - y_pre) ** 2
            total += np.sum(lo)
        #         消灭行

        self.lo = total

    def backward(self):
        """
        反向传播
        :return:
        """
        pass

    def forward(self):
        '''
        前项传播
        :return:
        '''

    def run(self):
        self.read_image()
        self.create()
        self.init_m()
        self.cut()
        self.find_edge()
        print(self.edge)
        # self.least_square()
        # print(self.superface[(0,0)])
        # self.loss()
        # print(self.lo)


if __name__ == '__main__':
    s = solver(4, "下载.jpg")
    s.run()
