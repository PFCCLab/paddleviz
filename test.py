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

class Model(nn.Layer):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(3, 6, 3, 1), # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2D(2, 2), # kernel_size, stride
            nn.Conv2D(6, 16, 3, 1),
            nn.Sigmoid(),
            nn.MaxPool2D(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*6*6, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        print('Shape---:', feature.shape)
        output = self.fc(feature.reshape([img.shape[0], -1]))
        return output


class MyTransformer(nn.Layer):
    def __init__(self) -> None:
        super(MyTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(128, 2, 512)
        self.encoder = nn.TransformerEncoder(encoder_layer, 2)
        

    def forward(self, img):
        img = self.encoder(img)
        return img


if __name__ == '__main__':
    # model = Testmodel()
    # model = Model()
    # x = paddle.randn([1, 3, 32, 32])

    model = MyTransformer()
    x = paddle.randn([2, 4, 128])

    y = model(x)


    # import paddle

    # # 随机生成输入张量，形状为(批次大小, 序列长度, 输入特征数)
    # inputs = paddle.rand((4, 3, 8))

    # # 随机生成初始隐藏状态，形状为(层数, 批次大小, 隐藏状态特征数)
    # prev_h = paddle.randn((5, 4, 8))

    # # 创建一个5层的RNN层，输入特征为8，隐藏状态特征为8
    # rnn = paddle.nn.RNN(8, 8, num_layers=5)

    # # 将输入张量和初始隐藏状态传入RNN层并获得输出张量和最终隐藏状态
    # outputs, final_states = rnn(inputs, prev_h)

    # # 打印输出张量和最终隐藏状态张量的形状
    # print(outputs.shape)
    # print(final_states.shape)

    # import paddle

    # rnn = paddle.nn.LSTM(4, 4, 2)

    # x = paddle.randn((1, 3, 4))
    # prev_h = paddle.randn((2, 1, 4))
    # prev_c = paddle.randn((2, 1, 4))
    # y, (h, c) = rnn(x, (prev_h, prev_c))


    # 先输出日志信息
    y.sum().backward()

    dot = make_graph(y)
    dot.render('viz-result.gv', format='png', view=True)
    