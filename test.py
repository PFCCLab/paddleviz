import paddle
import paddle.nn as nn

from paddleviz.viz import make_graph

class Testmodel(nn.Layer):
    def __init__(self):
        super(Testmodel, self).__init__()
        self.conv = nn.Conv2D(3, 6, 3, 1)
        self.pool = nn.MaxPool2D(2, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x 

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
    x = paddle.randn([1, 3, 24, 24])
    y = model(x)
    dot = make_graph(y)
    dot.render('viz-result.gv', format='png', view=True)