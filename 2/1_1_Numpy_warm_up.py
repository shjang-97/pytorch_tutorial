import numpy as np

if __name__ == '__main__':
    N = 64       # batch size
    D_in = 1000  # input dim
    H = 100      # hidden layer dim
    D_out = 10   # output dim

    # 정규 분포로 무작위로 입력과 출력 데이터 생성
    x = np.random.randn(N, D_in)
    y = np.random.randn(N, D_out)

    # 정규분포로 무작위로 가중치 초기화
    w1 = np.random.randn(D_in, H)
    w2 = np.random.randn(H, D_out)

    learning_rate = 1e-6

    for t in range(500):
        h = x.dot(w1)
        h_relu = np.maximum(h, 0)  # ReLU 활성화 함수
        y_pred = h_relu.dot(w2)

        # loss 계산 (유클라디언)
        loss = np.square(y_pred - y).sum()

        # loss에 따른 w1, w2의 gradient를 계산하고 backward
        grad_y_pred = 2.0 * (y_pred - y)  # loss 미분값
        grad_w2 = h_relu.T.dot(grad_y_pred)
        grad_h_relu = grad_y_pred.dot(w2.T)
        grad_h = grad_h_relu.copy()
        grad_h[h<0] = 0  # relu의 미분은 input(h)이 0보다 크면 1(곱하니까 값 그대로), 작으면 0
        grad_w1 = x.T.dot(grad_h)

        # 가중치 갱신
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2
        print(loss)

    print(w1)
    print(w2)