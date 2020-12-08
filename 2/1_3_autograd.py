import torch

dtype = float
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)
# requires_grad의 default 값이 false임
# 입력과 출력에 대한 tensor 변화도를 계산할 필요는 없기 때문이다

w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)  # weight에 대한 변화도 계산

learning_rate = 1e-6
for t in range(500):
    # forward
    y_pred = x.mm(w1).clamp(min = 0).mm(w2)

    # loss
    loss = (y_pred - y).pow(2).sum()  # loss는 sum을 통해 (1, ) 형태의 tensor로 바뀌고,
    if t % 100 == 99:
        print(t, loss.item())         # .item()으로 스칼라 값으로 변경됨

    # backward
    loss.backward()  # requires_grad = True를 갖는 모든 tensor에 대한 loss gradient를 계산
    # w1.grad와 w2.grad는 w1과 w2 각각에 대한 loss 변화도를 갖는 tensor이다

    # gradient descent를 사용하여 weight 수동 갱신
    with torch.no_grad():
        # 가중치들이 requires_Grad = True이지만, autograd할 때 (update할 때) 이를 추적할 필요가 없으므로
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # 가중치 update 이후 수동으로 변화도를 0으로 만들어야 됨 (다시 변화도 계산해야 되니까)
        w1.grad.zero_()
        w2.grad.zero_()
