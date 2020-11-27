# What is PyTorch?
- 2017년 초에 공개된 딥러닝 프레임워크  
- 개발자들과 연구자들이 쉽게 GPU를 활용하여 인공 신경망 모델을 만들고 학습시킬 수 있게 도와줌  
- 파이토치의 전신이라고 할 수 있는 토치(torch)는 루아 프로그래밍 언어로 되어 있었지만, 파이토치는 파이썬으로 작성되어 파이썬의 언어 특징을 많이 가지고 있음   

>#### 다른 프레임워크와 비교
1. Numpy VS PyTorch  
    - x,y,z 세 변수에 대해 학습하고자 할 때, 기울기를 계산하기 위해 연산 그래프를 쭉 따라서 미분해야 됨
        - Numpy: Numpy만을 사용하면 모든 미분 식을 직접 계산하고 코드로 작성해야 되므로 변수 하나 당 두 줄씩 여섯 줄이 필요함
        - Pytorch: backward()라는 함수를 한 번 호출하면 이 과정을 자동으로 계산해줌
    - GPU를 통한 연산 가능 여부
        - numpy 만으로는 gpu로 값을 보내 연산을 돌리고 다시 값을 받는 게 불가능
        - 파이토치는 내부적으로 CUDA, cuDNN이라는 API를 통해 gpu 연산을 사용할 수 있고 이로 인해 생기는 연산 속도 차이가 엄청남
            - cuda: NVIDIA에서 gpu를 통한 연산을 가능하게 만든 API 모델
            - cuDNN: CUDA를 이용해 딥러닝 연산을 가속해주는 라이브러리
        - GPU를 사용하면 연산속도가 CPU의 15배 이상 빨라짐
        
    - 즉 파이토치는 심층신경망을 만들 때 함수 및 기울기 계산, gpu를 이용한 연산 가속 등의 장점이 있기 때문에 딥러닝 프레임워크로 사용하기 아주 좋음
     
1. Tensorflow VS PyTorch
    - 텐서플로우와 파이토치 모두 연산에 gpu를 사용할 수 있는 딥러닝 프레임워크임
    - 그러나 텐서플로우는 `Define and Run` 방식이고, 파이토치는 `Define by Run` 방식이다 
        - Define and Run: 연산 그래프를 미리 만들고 실제 연산할 때 값을 전달하여 결과를 얻음
        - Define by Run: 그래프를 만듦과 동시에 값이 할당 (직관적이어서 쉬움)  
    - 또한 연산 속도도 파이토치가 더 빠름
        - 그래프를 고정하고 값만 전달하는 텐서플로우가 더 빠를 수 있겠지만, 실제로 실험에 많이 사용되는 모델로 벤치마킹한 결과 파이토치가 텐서플로우보다 2.5배 빨랐다
        - 모델, 사용하는 함수 마다 차이는 있겠지만 파이토치가 속도 면에서 전반적으로 텐서플로우보다 빠르거나 밀리지 않음
         
### 간단 예제
> #### Tensor
Tensor는 Numpy의 ndarray와 유사하고, GPU를 사용한 연산 가속도가 가능함
```buildoutcfg
x = torch.empty(5, 3)  # 초기화되지 않은 5x3 행렬
x = torch.rand(5, 3)   # 무작위로 초기화된 행렬 생성
x = torch.randn(5, 3)
x = torch.zeros(5, 3, dtype=torch.long)  # dtype이 long이고 0으로 채워진 행렬 생성
x = torch.tensor([5.5, 3])  # data로부터 tensor를 직접 생성
```

기존 tensor를 바탕으로 새로운 tensor를 만듦  
이들 메소드는 사용자로부터 새로운 값을 제공받지 않는 한, 입력 tensor의 속성들(예: dtype)을 재사용
```buildoutcfg
x = x.new_ones(5, 3, dtype=torch.double)      # new_* 메소드는 크기를 받습니다
x = torch.randn_like(x, dtype=torch.float)    # dtype을 오버라이드(Override) 합니다! (기존 x와 동일한 크기)   
```

행렬의 크기를 구함
```buildoutcfg
print(x.size())
```

> #### 연산
바꿔치기(in-place): tensor의 값을 변경하는 연산 뒤에는 _를 붙임  
```buildoutcfg
y.add_(x)   # y에 x 더하기
y.copy_(x)  # x값을 y에 복사
```
크기 변경
```buildoutcfg
x = x.view(-1, 8)
```
만약 텐서에 하나의 값만 존재한다면 `.item()`을 사용하여 숫자 값을 얻을 수 있음
```buildoutcfg
x = torch.randn(1)
print(x)         # tensor([0.8994])
print(x.item())  # 0.8994463682174683
```

> #### numpy 배열을 torch tensor로 변환
```buildoutcfg
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)  # a랑 b 모두 같은 memory를 공유하고 있어서 값이 같이 변함 
```