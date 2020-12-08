import torch

dtype = torch.float
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

N, D_in, H, D_out = 64, 1000, 100, 10

# 정규 분포를 갖는 무작위 입력, 출력 데이터 생성
x = torch.randn(N, D_in, device = device, dtype=dtype)
y = torch.randn(N, D_out, device = device, dtype=dtype)

# 무작위로 가중치 초기화
w1 = torch.randn(D_in, H, device = device, dtype=dtype)
w2 = torch.randn(H, D_out, device = device, dtype=dtype)

learnig_rate = 1e-6

for t in range(500):
    # forward
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # loss 계산 후 출력
    loss = (y_pred - y).pow(2).sum().item()  # pow(2): 제곱
    if t % 100 == 99:
        print(t, loss)

    # loss에 따른 w1, w2의 gradien를 계산하고 backward
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()  # copy
    grad_h[h<0] = 0
    grad_w1 = x.t().mm(grad_h)

    # gradient descent
    w1 -= learnig_rate * grad_w1
    w2 -= learnig_rate * grad_w2