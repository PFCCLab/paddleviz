import os
os.environ['GLOG_v'] = '6'

import paddle
import paddle.nn as nn

from paddleviz.viz import make_graph


class Testmodel(nn.Layer):
    def __init__(self):
        super(Testmodel, self).__init__()
        self.conv = nn.Conv2D(3, 6, 3, 1)
        self.pool = nn.MaxPool2D(2, 2)

    def forward(self, x):
        y = x ** 2
        y = x + y
        # x = self.conv(x)
        # x = self.pool(x)
        return y 

class LeNet(nn.Layer):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(3, 6, 3, 1), # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2D(2, 2), # kernel_size, stride
            nn.Conv2D(6, 16, 3, 1),
            nn.Sigmoid(),
            nn.MaxPool2D(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.reshape([img.shape[0], -1]))
        return output



if __name__ == '__main__':
    model = Testmodel()
    # model = LeNet()
    x = paddle.randn([1, 1, 4, 4])
    x.stop_gradient = False 
    
    y = model(x)

    print(":Hello-------")

    # 先输出日志信息
    y.backward()

    dot = make_graph(y)
    dot.render('viz-result.gv', format='png', view=True)
    # unittest.main()
    

# os.environ['GLOG_v'] = '11'

# import paddle
# import paddle.nn as nn

# class Testmodel(nn.Layer):
#     def __init__(self):
#         super().__init__()
#     def forward(self, x):
#         y = x**2
#         y = x + y
#         return y

# tst_model = Testmodel()
# x = paddle.randn([2, 2])
# x.stop_gradient = False
# y = tst_model(x)

# print(y)
# y.backward()