import os
os.environ['GLOG_v'] = '6'

import paddle
import paddle.nn as nn

from paddleviz.viz import make_graph

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
        output = self.fc(feature.reshape([img.shape[0], -1]))
        return output

if __name__ == '__main__':
    
    # 定义网络
    model = Model()
    x = paddle.randn([1, 3, 32, 32])

    # 正向推理
    y = model(x)

    # 反向推理
    y.sum().backward()

    # 可视化网络反向图，dpi 代表分辨率，默认为600，如果网络较大，可以改为更大的分辨率
    dot = make_graph(y, dpi="600")

    # 绘制保存反向图
    dot.render('viz-result.gv', format='png', view=False)

