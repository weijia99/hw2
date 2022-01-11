from PIL import Image
import numpy as np
# import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

# sk 进行线性回归
from scipy.optimize import root
# 优化包
class solver():
    def __init__(self,m,path):
        self.m=m
        self.path = path
        # 放入随机生成的点云
        self.lists =[]
        self.rec = {}
    #     记录每个点，所包含在那个领域
        self.superface = {}


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
            self.lists.append((0,i))
            self.rec[(0,i)] = []



    def cut(self):
        '''
        分割图像，到m个点上
        :return:
        '''
        for i in range(self.I_array.shape[0]):
            for j in range(self.I_array.shape[1]):
                minP = (self.lists[0][0]-i)**2+(self.lists[0][1]-j)**2
                x=0
                y=0

                for point in self.lists:
                    if (point[0] - i)**2+(point[1] - j)**2<minP:
                        minP = (point[0] - i)**2+(point[1] - j)**2
                        x=point[0]
                        y=point[1]

                self.rec[(x,y)].append(([i,j]))



    def least_square(self):
        for i in range(self.m):
            point = self.lists[i]
            x = self.rec[point]
            y =[]
            for index in x:
                y.append(self.I_array[index[0],index[1]])
    #         加入到numpy

            x = np.array(x)

            y =np.array(y)
            # 将 y 分别增加一个轴，以满足 sklearn 中回归模型认可的数据
            # 此时由于 x 是多元变量，则不用添加新的轴了
            y = y[:, np.newaxis]

            model = LinearRegression()  # 构建线性模型
            model.fit(x, y)  # 自变量在前，因变量在后
            predicts = model.predict(x)  # 预测值
            R2 = model.score(x, y)  # 拟合程度 R2
            # print('R2 = %.3f' % R2)  # 输出 R2
            coef = model.coef_  # 斜率
            intercept = model.intercept_  # 截距
            # print(model.coef_.shape, model.intercept_.shape)  # 输出斜率和截距
    #         w =np.dot(np.dot(np.linalg.inv(np.dot(x.T,x)),x.T),y)
    #
    # #         这是w的参数
    #         进行水平拼接
            intercept=intercept[:, np.newaxis]
            self.superface[point] =  np.vstack((coef,intercept))









    def run(self):
        self.read_image()
        self.create()
        self.cut()
        self.least_square()
        # print(self.superface[(0,0)])




if __name__ == '__main__':
    s =solver(4,"下载.jpg")
    s.run()
 



