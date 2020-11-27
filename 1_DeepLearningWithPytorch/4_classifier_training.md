# 분류기(classifier) 학습하기
## 데이터 처리
일반적으로 이미지나 텍스트, 오디오, 비디오 데이터를 다룰 때는 표준 python 패키지를 이용하여 numpy 배열로 불러오면 된다.  
그 후에 그 배열을 `torch.*Tensor`로 변환
- 이미지: Pillow나 OpenCV 
- 오디오: SciPy, LibROSA 
- 텍스트: Python, Cython, NLTK, SpaCy

> ####`torchvision`패키지

- 영상 분야를 위해 만들어짐
- Imagenet이나 CIFAR10, MNIST 등과 같이 일반적으로 사용하는 dataset을 위한 data loader
- `torchvision.datasets`과 이미지용 data transformer, 즉 `torch.utils.data.DataLoader`가 포함되어 있음

위의 기능은 엄청 편리하고, 유사한 코드를 매번 반복해서 작성하는 것을 피할 수 있음

본 튜토리얼에서는 `CIFAR10` dataset을 사용
- 10개 class
    - 비행기, 자동차, 새, 고양이, 사슴, 개, 개구리, 말, 배, 트럭
- Img size: 3x32x32 (3개 채널, 32x32 픽셀 크기 이미지)
![](https://tutorials.pytorch.kr/_images/cifar10.png)

## 이미지 분류기 학습하기
단계:  
1. `torchvision`을 사용하여 CIFAR10의 training/test dataset을 불러오고, 정규화(normalizaing) 함
1. 합성곱 신경망(Convolution Neural Network) 정의
1. 손실함수 정의
1. training dataset을 사용하여 신경망 학습
1. test dataset을 사용하여 신경망을 검사

> #### 1. CIFAR 10을 불러오고 정규화하기
`torchvision`을 사용하여 매우 쉽게 CIFAR10을 불러올 수 있음
```buildoutcfg
import torch
import torchvision
import torchvision.transforms as transforms
```
torchvision 데이터셋의 출력(output)은 [0, 1] 범위를 갖는 PILImage임  
이를 [-1, 1]의 범위로 정규화된 Tensor로 변환
```buildoutcfg
>> 4_1_data_loading.py
```
- data 다운, 정규화, 한 배치 내의 이미지 보여주기

> #### 2. 합성곱 신경망(Convolution neural network 정의)
신경망 섹션에서 배웠던 신경망을 그대로 사용  
(기존에는 1채널의 이미지만 처리하도록 정의된 것을 3채널 이미지를 처리할 수 있도록 수정)
> #### 3. 손실 함수와 optimizer 정의
Cross-entropy loss와 momentum 값을 갖는 SGD를 사용   
-> Momentum???
```buildoutcfg
import torch.nn as nn
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum=0.9)
```
> #### 4. 신경망 학습
데이터를 반복해서 신경망에 입력으로 제공하고, 최적화(optimize)하면 됨
```
for epoch in range(2):   # 데이터셋을 수차례 반복합니다.

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
        inputs, labels = data

        # 변화도(Gradient) 매개변수를 0으로 만들고
        optimizer.zero_grad()

        # 순전파 + 역전파 + 최적화를 한 후
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 통계를 출력합니다.
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
```

학습된 모델 저장
```buildoutcfg
    PATH = './ckpt/cifar_net.pth'
    torch.save(net.state_dict(), PATH)
```

> ### 5. test dataset으로 신경망 검사
training dataset을 epochs만큼 반복하며 신경망을 학습시켰음  
이를 이제 test dataset을 이용해 accuracy를 판별해보자!

신경망이 예측한 출력과 진짜 정답(ground-truth)를 비교하는 방식으로 확인  
만약 예측이 맞다면 샘플은 '맞은 예측값(correct predictions' 목록에 넣음

시험용 데이터 출력
```buildoutcfg
dataiter = iter(testloader)
images, labels = dataiter.next()

# 이미지를 출력합니다.
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
```

![](https://tutorials.pytorch.kr/_images/sphx_glr_cifar10_tutorial_002.png)

Out:
```buildoutcfg
GroundTruth:    cat  ship  ship plane
```

저장한 모델 불러오기
```buildoutcfg
net = Net()
net.load_state_dict(torch.load(PATH))
```

테스트 데이터 신경망 예측  
출력은 10개 분류 각각에 대한 값으로 나타남, 어떤 분류에 대해서 더 높은 값이 나타난다는 것은 신경망이 그 이미지가 해당 분류에 더 가깝다고 생각하는 것과 같음. 따라서, 가장 높은 값을 갖는 인덱스를 출력하자!
```buildoutcfg
outputs = net(images)

_, predicted = torch.max(outputs, 1)  # 가장 높은 값을 갖는 인덱스 출력
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))
```

Out:
```buildoutcfg
Predicted:    cat truck truck plane
```

전체 데이터셋 동작 확인:
```buildoutcfg
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
```
Out:
```buildoutcfg
accuracy of the network on the 10000 test images: 54 %
```
10 클래스 중 랜덤으로 찍었을 떄의 확률인 10%보다 정확도가 높은 걸 보니 신경망이 무언가를 학습한 거 같다!
그럼 이제 어떤 것들을 잘 분류하고 못 분류했는지 알아보자!
```buildoutcfg
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
```
Out:
```buildoutcfg
accuracy of plane : 72 %
accuracy of   car : 68 %
accuracy of  bird : 30 %
accuracy of   cat : 38 %
accuracy of  deer : 38 %
accuracy of   dog : 44 %
accuracy of  frog : 68 %
accuracy of horse : 64 %
accuracy of  ship : 67 %
accuracy of truck : 52 %
```
##GPU에서 학습하기
```buildoutcfg
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
inputs, labels = data[0].to(device), data[1].to(device)
```