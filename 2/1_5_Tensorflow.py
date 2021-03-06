import tensorflow as tf
import numpy as np

# 먼저 연산 그래프를 구성합니다:

# N은 배치 크기이며, D_in은 입력의 차원입니다;
# H는 은닉층의 차원이며, D_out은 출력 차원입니다.
N, D_in, H, D_out = 64, 1000, 100, 10

# 입력과 정답(target) 데이터를 위한 플레이스홀더(placeholder)를 생성합니다;
# 이는 우리가 그래프를 실행할 때 실제 데이터로 채워질 것입니다.
x = tf.placeholder(tf.float32, shape=(None, D_in))
y = tf.placeholder(tf.float32, shape=(None, D_out))

# 가중치를 저장하기 위한 Variable을 생성하고 무작위 데이터로 초기화합니다.
# Tensorflow의 Variable은 그래프가 실행되는 동안 그 값이 유지됩니다.
w1 = tf.Variable(tf.random_normal((D_in, H)))
w2 = tf.Variable(tf.random_normal((H, D_out)))

# 순전파 단계: Tensorflow의 Tensor 연산을 사용하여 예상되는 y 값을 계산합니다.
# 이 코드가 어떠한 수치 연산을 실제로 수행하지는 않는다는 것을 유의하세요;
# 이 단계에서는 나중에 실행할 연산 그래프를 구성하기만 합니다.
h = tf.matmul(x, w1)
h_relu = tf.maximum(h, tf.zeros(1))
y_pred = tf.matmul(h_relu, w2)

# Tensorflow의 Tensor 연산을 사용하여 손실(loss)을 계산합니다.
loss = tf.reduce_sum((y - y_pred) ** 2.0)

# w1, w2에 대한 손실의 변화도(gradient)를 계산합니다.
grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])

# 경사하강법(gradient descent)을 사용하여 가중치를 갱신합니다. 실제로 가중치를
# 갱신하기 위해서는 그래프가 실행될 때 new_w1과 new_w2 계산(evaluate)해야 합니다.
# Tensorflow에서 가중치의 값을 갱신하는 작업이 연산 그래프의 일부임을 유의하십시오;
# PyTorch에서는 이 작업이 연산 그래프의 밖에서 일어납니다.
learning_rate = 1e-6
new_w1 = w1.assign(w1 - learning_rate * grad_w1)
new_w2 = w2.assign(w2 - learning_rate * grad_w2)

# 지금까지 우리는 연산 그래프를 구성하였으므로, 실제로 그래프를 실행하기 위해 이제
# Tensorflow 세션(session)에 들어가보겠습니다.
with tf.Session() as sess:
    # 그래프를 한 번 실행하여 w1과 w2 Variable을 초기화합니다.
    sess.run(tf.global_variables_initializer())

    # 입력 데이터 x와 정답 데이터 y를 저장하기 위한 NumPy 배열을 생성합니다.
    x_value = np.random.randn(N, D_in)
    y_value = np.random.randn(N, D_out)
    for t in range(500):
        # 그래프를 여러 번 실행합니다. 매번 그래프가 실행할 때마다 feed_dict
        # 인자에 x_value를 x에, y_value를 y에 할당(bind)하도록 명시합니다.
        # 또한, 그래프를 실행할 때마다 손실과 new_w1, new_w2 값을
        # 계산하려고 합니다; 이러한 Tensor들의 값은 NumPy 배열로 반환됩니다.
        loss_value, _, _ = sess.run([loss, new_w1, new_w2],
                                    feed_dict={x: x_value, y: y_value})
        if t % 100 == 99:
            print(t, loss_value)