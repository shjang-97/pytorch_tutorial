#  예제로 배우는 파이토치   

토치의 2가지 주요 특징
- numpy와 유사하지만 gpu 상에서 실행 가능한 n차원 텐서
- neural network를 구성하고 학습하면서 자동미분(autograd)

### 예제
fully connected ReLU 신경망 예제   
< 구조 >
- 하나의 은닉층
- 예측값과 정답값의 유클라디안 거리를 최소화하는 방법으로 gradient descent 사용

## Numpy   

numpy는 n차원 배열 객체를 다루기 위해 다양한 함수를 제공 (과학적 분야의 연산을 위한 포괄적 framework)   
그러나 넘파이는 연산 그래프, 딥러닝, 변화도에 대해서는 알지 못함   
numpy를 사용하여 forward와 barward를 직접 구현하여 2-layer를 갖는 신경망을 구현해보자!

```angular2html
>> 1_1_Numpy_warm_up.py
```

numpy는 훌륭한 프레임워크지만, gpu를 사용하여 연산을 가속화할 수가 없음   
gpu로 연산하면 속도가 50배 혹은 그 이상 오를 수 있음   
따라서 딥러닝 연산은 gpu를 사용하는 것이 좋고, numpy는 적합하지 않음   

## Tensors
pytorch tensor는 개념적으로 numpy 배열과 동일   
tensor는 N차원 배열이며, pytorch도 tensor 연산을 위한 다양한 함수를 제공함   
numpy와 마찬가지로 torch tensor도 딥러닝이나 연산 그래프, gradient를 알지 못하며, 과학적 분야의 연산을 위한 포괄적인 도구일 뿐이다.   

**그러나 numpy와 달리 tensor는 gpu를 사용하여 수치 연산을 가속화할 수 있음**   
gpu에서 tensor를 사용하려면 새로운 자료형으로 변환(cast)해주기만 하면 됨  

예제 (numpy 예제와 같음, 단지 torch tensor로 구현했을 뿐이다)
```angular2html
>> 1_2_Tensors.py
```

## Autograd
forward와 backward를 직접 구현하는 것은 대규모의 복잡한 신경망에서는 별로 좋지 않음   
** -> torch의 autograd 패키지를 사용하여 신경망에서 backward 연산을 자동화하자!**  

- 신경망의 forward 단계는 연산 그래프를 정의  
    - 이 그래프의 노드는 tensor, 엣지는 입력 tensor로부터 출력 tensor를 만들어내는 함수가 됨
    - 이 그래프를 통해 역전파를 하게 되면 gradient를 쉽게 계산이 가능

각 Tensor는 연산 그래프에서 노드로 표현됨   
만약 x가 `x.requires_grad=True` 인 x.grad는 어떤 스칼라 값에 대한 x의 변화도를 갖는 또 다른 tensor임   

### 예제
autograd를 사용한 2 layer neural network   
(역전파를 직접 구현 X)
```
>> 1_3_autograd.py
```

## 새 autograd 함수 정의
내부적으로 autograd의 기본 연산자는 실제로 tensor를 조작하는 2개의 함수로 이루어져 있음   
- forward: 입력 tensor로부터 출력 tensor를 계산하는 부분
- backward: 어떤 스칼라 값에 대한 출력 tensor의 변화도를 전달받고, 동일한 스칼라 값에 대한 입력 tensor의 변화도를 계싼

PyTorch에서 `torch.autograd.Function` 의 서브클래스(subclass)를 정의하고 forward 와 backward 함수를 구현함으로써 사용자 정의 autograd 연산자를 손쉽게 정의할 수 있음  
그 후, 인스턴스(instance)를 생성하고 이를 함수처럼 호출하여 입력 데이터를 갖는 Tensor를 전달하는 식으로 새로운 autograd 연산자를 사용할 수 있음

### 예제
ReLU로 비선형적으로 동작하는 사용자 정의 autograd 함수 정의  
2 layers neural network에 적용
```angular2html
>> 1_4_new_autograd_function.py
```

## Tensorflow : 정적 그래프