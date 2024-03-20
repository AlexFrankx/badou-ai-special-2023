# -*- coding: utf-8 -*-
# @Author : Ruby
# @File : w9-2_Mnist_pytorch.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class Model:
    def __init__(self, net, cost, optimist):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimist)
        pass

    def create_cost(self, cost):
        """指定损失函数"""
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }
        return support_cost[cost]

    def create_optimizer(self, optimist, **rests):
        """指定优化器"""
        support_optim = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.001, **rests),
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }

        return support_optim[optimist]

    def train(self, train_loader, epoches=3):
        for epoch in range(epoches):
            running_loss = 0.0
            running_acc = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data

                self.optimizer.zero_grad()  # 清空过往梯度

                # forward + backward + optimize
                outputs = self.net(inputs)  # 网络输出
                loss = self.cost(outputs, labels)  # 计算网络输出和真实值之间的差距
                loss.backward()  # 反向传播，计算当前梯度
                self.optimizer.step()  # 根据梯度更新网络参数

                running_loss += loss.item()
                running_acc += (outputs.argmax(1) == labels).type(torch.float).sum().item()
                if i % 100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f acc: %.1f%%' %
                          (epoch + 1, (i + 1) * 1. / len(train_loader) * 100, running_loss / 100, running_acc / 32))
                    running_loss = 0.0
                    running_acc = 0.0

        print('Finished Training')

    def evaluate(self, test_loader):
        print('Evaluating ...')
        correct = 0
        total = 0
        with torch.no_grad():  # no grad when test and predict,测试时模型参数不用更新
            for data in test_loader:
                images, labels = data

                outputs = self.net(images)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


def mnist_load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0, ], [1, ])])  # 逐channel的对图像进行标准化（均值变为0，标准差变为1），可以加快模型的收敛

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)
    return trainloader, testloader


class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


if __name__ == '__main__':
    # train for mnist
    net = MnistNet()
    model = Model(net, 'CROSS_ENTROPY', 'ADAM')
    train_loader, test_loader = mnist_load_data()
    model.train(train_loader)
    model.evaluate(test_loader)
