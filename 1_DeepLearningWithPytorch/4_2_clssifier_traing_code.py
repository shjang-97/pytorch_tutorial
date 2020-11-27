import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.optim as optim

from model import Net

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def dataloader():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    return trainloader, testloader


def training(trainloader, net, PATH='./ckpt/cifar_net.pth', epochs=2):
    # loss & optimizer 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # 신경망 학습
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()  # gradient 매개변수를 0으로 만듦
            outputs = net(inputs)  # forward
            loss = criterion(outputs, labels)
            loss.backward()  # backward
            optimizer.step()  # TODO??? 최적화 하는 곳??

            running_loss += loss.item()
            if i % 2000 == 1999:  # 2000 번 마다 loss 평균 출력
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')

    # 학습 모델 저장
    torch.save(net.state_dict(), PATH)


def test(testloader, net, PATH='./ckpt/cifar_net.pth'):
    net.load_state_dict(torch.load(PATH))  # checkpoint 불러오기

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            output = net(images)
            _, predicted = torch.max(output.data, 1)  # 10개의 클래스 중에서 가장 높은 값을 갖는 인덱스를 출력
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


def test_class(testloader, net, PATH='./ckpt/cifar_net.pth'):
    '''
    test data의 각 클래스별 정확도 출력
    '''

    classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net.load_state_dict(torch.load(PATH))  # checkpoint 불러오기

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            output = net(images)
            _, predicted = torch.max(output.data, 1)  # 10개의 클래스 중에서 가장 높은 값을 갖는 인덱스를 출력
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

if __name__ == '__main__':
        trainloader, testloader = dataloader()

        PATH = './ckpt/cifar_net.pth'

        # 모델 정의
        net = Net().to(device)

        # 모델 학습
        # training(trainloader, net, PATH)

        # 모델 테스트
        # test(testloader, net, PATH)
        test_class(testloader, net, PATH)
