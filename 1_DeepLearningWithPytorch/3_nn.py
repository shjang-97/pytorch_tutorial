import torch
import torch.nn as nn
import torch.nn.functional as F



class Net(nn.Module):
    def __init__(self, use_cuda):
        super(Net, self).__init__()
        # convolution kernel 정의 (kernel size: 3x3)
        conv1 = nn.Conv2d(1, 6, 3)  # input: 이미지 채널 1개(흑백)
        conv2 = nn.Conv2d(6, 16, 3)
        pool = nn.MaxPool2d(2)  # 2x2 윈도우 크기로 맥스 풀링

        # affine 연산: y = Wx + b
        fc1 = nn.Linear(16*6*6, 120) # output: 16 (6x6 차원)
        fc2 = nn.Linear(120, 84)
        fc3 = nn.Linear(84, 10)

        # conv module 정의
        self.conv_module = nn.Sequential(
            conv1,
            nn.ReLU(),
            pool,
            conv2,
            nn.ReLU(),
            pool
        )

        # fc 모듈 정의
        self.fc_module = nn.Sequential(
            fc1,
            nn.ReLU(),
            fc2,
            nn.ReLU(),
            fc3
        )
        '''
        함수들을 엮어 하나의 모듈로 구성한 이유는 gpu 할당 때문임
        그래프를 선언할 때 gpu option을 주면 그 안에서 선언한 함수는 모두 gpu에 할당되는 텐서플로우와는 다르게
        torch는 직접 .cuda()를 통해 할당시켜줘야 함
        번거로움을 최소화하기 위해 모듈로 구성
        '''
        if use_cuda:
            # gpu 사용이 가능하면 gpu로 할당
            self.conv_module = self.conv_module.cuda()
            self.fc_module = self.fc_module.cuda()

    def forward(self, x):
        x = self.conv_module(x)
        x = x.view(-1, self.num_flat_features(x))  # make linear
        x = self.fc_module(x)  # TODO: 원래 딥러닝에서 마지막에는 activation function 안 넣었나?
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        print(size)
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == '__main__': # 스크립트 파일이 메인 프로그램으로 사용될 때와 모듈로 사용될 때를 구분하기 위한 용도
    use_cuda = torch.cuda.is_available()
    net = Net(use_cuda)
    print(net)

    params = list(net.parameters())
    print(len(params))  # 10 ( #TODO 0: conv1의 weight, 1: conv1의 bias인가? [6]
    print(params[0].size())  # conv1의 weight [6, 1, 3, 3]

    input = torch.randn(1, 1, 32, 32) # nSamples x nChannels x Height x Width
    out = net(input)
    print(out)

    net.zero_grad()  # gradient beffer를 0으로 설정
    out.backward(torch.randn(1, 10))

    '''
    torch.nn은 미니배치만 지원
    torch.nn은 하나의 샘플이 아닌 미니 배치를 입력으로 받음
    예) nnConv2d: nSamples x nChannels x Height x Width의 4차원 tensor를 입력으로 받음
    만약 하나의 샘플만 있다면 input.unsqeeze(0)을 사용해서 0번째에 가상의 차원(1) 추가
    '''

    target = torch.randn(10)  # 예시를 위한 임의의 정답
    target = target.view(1, -1)  # 출력과 같은 shape으로 만듦
    criterion = nn.MSELoss()

    loss = criterion(out, target)
    print(loss)

    import torch.optim as optim

    optimizer = optim.SGD(net.parameters(), lr = 0.01)  # optimizer 생성

    # 학습 과정
    optimizer.zero_grad()  # 그라디언트 버퍼를 0으로 설정
    output = net(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()  # update 진행