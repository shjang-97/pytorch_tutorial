# 신경망
신경망은 *torch.nn* package를 사용하여 생성   
*nn*은 모델을 정의하는 데 사용, *autograd*는 미분하는 데 사용   
*nn.Module*은 layer와 *output*을 반환하는 *forward(input)* 메서드를 포함   
>torch 패키지
```buildoutcfg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
```
- nn: 딥러닝 모델에 필요한 모듈이 있는 패키지 (클래스)  
ex) nn.Linear(128,128), nn.ReLU()
- F: nn과 같은 모듈이 모아져 있지만, 함수의 input으로 반드시 연산되어야 되는 값을 받음 
ex) F.linear(x, 128,128), F.rulu(x)
    - nn과 nn.functional(F)은 같은 결과를 내며 차이가 없음   
    pytorch 공식 문서에 torch.nn.CrossEntropyLoss 설명을 보면 파라미터로 weight는 각 class에 대한 가중치 정보다.  
    예를들어 10개의 class로 분류하는 문제에 손실 함수로 cross entropy loss를 사용하고 특정 클래스(7번)를 더 잘 찾고 싶다면, 7번 class 부분의 loss에 큰 값의 가중치를 곱해서 더 잘 학습시키게 만들 수 있다.  
    이를 위해서 처음 CrossEntropyLoss 클래스를 인스턴스화 할 때 weight 값을 인자로 보내 한 번만 설정하면 그 뒤로 학습시킬 때 계속 적용이 된다.  
    반면 torch.nn.functional.cross_entropy 함수에도 파라미터로 weight가 동일하게 있지만 이 weight 를 적용시켜서 loss 계산을 하고싶을 때는 매 번 함수를 호출할 때마다 인자로 weight 값을 넣어줘야 한다.   
    출처: https://cvml.tistory.com/10 [Computer Vision :)]
- optim: 학습에 관련된 optimazing method가 있는 패키지
- data_utils: batch generator 등 학습 데이터와 관련된 패키지


> 신경망의 일반적인 학습 과정
1. 학습 가능한 매개변수(or 가중치(weight))를 갖는 신경망 정의
1. 데이터셋 입력을 반복
1. 입력을 신경망에서 전파(process) (--->)
1. 손실(loss; 출력이 정답으로부터 얼마나 떨어져 있는지)을 계산
1. 변화도(gradient)를 신경망의 매개변수들에 역으로 전파
1. 신경망의 가중치를 갱신  
new weight = weight - learning rate * gradient

## 신경망 정의하기
> 다룰 내용
- 신경망 정의
- 입력을 처리하고 backward를 호출하는 것

> 숫자 이미지 분류 신경망 예제 

```buildoutcfg
>> 3_nn.code.py
```
* 요약:
    * `torch.Tensor`: `backward`와 같은 *autograd* 연산을 지원하는 **다차원 배열**, 또한 tensro에 대한 *gradient*를 가지고 있음
    - `nn.Module` : 신경망 모듈. 매개변수를 캡슐화(encapsulation)하는 간단한 방법으로 gpu로 이동, 내보내기(exporting), 불러오기(loading) 등의 작업을 위한 헬퍼(helper) 제공
    - `nn.Parameter` : Tensor의 한 종류로 `Module`에 속성으로 할당될 때 자동으로 매개변수로 등록
    - `autograd.Function` : autograd 연산의 전방향과 역방향 정의를 구현, 모든 `Tensor` 연산은 하나 이상의 `Function` 노드를 생성하며 각 노드는 `Tensor`를 생성하고 이력(history)를 부호화하는 함수들과 연결함
    
## 손실 함수(Loss function)
> 다룰 내용
- 손실을 계산하는 것  

손실함수는 (output, target)을 한 쌍(pair)의 입력으로 받아, 출력(output)이 정답(target)으로부터 얼마나 멀리 떨어져있는지 추정하는 값을 계산함   

nn 패키지에는 여러가지의 손실함수들이 존재  
ex) `nn.MSEloss`: 평균제곱오차(mean-squared error)

```buildoutcfg
output = net(input)
target = torch.randn(10)  # 예시를 위한 임의의 정답
target = target.view(1, -1)  # 출력과 같은 shape으로 만듦
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)
```
Out:
```buildoutcfg
tensor(1.7904, grad_fn=<MseLossBackward>)
```

이제 `.grad_fn` 속성을 사용하여 `loss`를 역방향에서 따라가다보면, 이러한 모습의 연산 그래프를 볼 수 있음
```buildoutcfg
input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
      -> view -> linear -> relu -> linear -> relu -> linear
      -> MSELoss
      -> loss
```

따라서 `loss.backward()`를 실행할 때, 전체 그래프는 손실(loss)에 대해 미분되며, 그래프 내의 `requires_grad = True`인 모든 Tensor는 변화도(gradient)가 누적된 `.grad` Tensor를 갖게 됨
```buildoutcfg
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU
```
Out:
```buildoutcfg
<MseLossBackward object at 0x7f520fa2ad30>
<AddmmBackward object at 0x7f520fa2ada0>
<AccumulateGrad object at 0x7f520fa2ada0>
```


## 역전파(backprop)
오차(error)를 역전파하기 위해서는 `loss.backward()`만 해주면 됨  
기존 변화도를 없애는 작업이 필요한데, 그렇지 않으면 변화도가 기존의 것에 누적되기 때문이다   

이제 `loss.backward()`를 호출하여 역전파 전과 후에 conv1의 bias gradient를 살펴보자!
```buildoutcfg
net.zero_grad()     # 모든 매개변수의 변화도 버퍼를 0으로 만듦

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
```

Out:
```buildoutcfg
conv1.bias.grad before backward
tensor([0., 0., 0., 0., 0., 0.])
conv1.bias.grad after backward
tensor([ 0.0086,  0.0034, -0.0079, -0.0114, -0.0105,  0.0007])
```

## 가중치 갱신
> 다룰 내용
- 신경망의 가중치를 갱신  

실제로 많이 사용되는 가장 단순한 update 방법은 SGD(Stochastic Gradient Descent) 방식임  
`new weight = weigth - learning rate * gradient`

간단한 python 코드:
```buildoutcfg
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_*f.grad.data * learning_rate
```

신경망을 구성할 때 SGD, Adam, RMSProp 등과 같은 다양한 update 방법을 사용하기 위해서는 `torch.optim`이라는 패키지를 사용하면 됨
```buildoutcfg
import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr = 0.01)  # optimizer 생성

# 학습 과정
optimizer.zero_grad()  # 그라디언트 버퍼를 0으로 설정
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()  # update 진행

```

 - `optimizer.zero_grad()`를 사용하여 수동으로 gradient buffer를 0으로 설정한 것 주의  
 이는 역전파에서 설명한 것처럼 변화도가 누적되기 때문
 