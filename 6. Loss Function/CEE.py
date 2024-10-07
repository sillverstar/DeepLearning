# 교차 엔트로피 오차(CEE, cross entropy error)

# (데이터 하나에 대한 CEE) = - ∑(t_k)log(y_k)
# (y_k = 신경망의 출력, t_k = 정답 레이블, k = 데이터의 차원 수, log는 밑이 e인 자연로그)

import numpy as np
import matplotlib.pyplot as plt

# 자연로그 그래프 확인
x = np.arange(0.0, 1.0, 0.01)
y = np.log(x)

plt.style.use('dark_background')
plt.plot(x, y)
plt.show()


def cross_entropy_error(y, t):
    delta = 1e-7 # 아주 작은 값 추가 Why? = np.log() 함수에 0을 입력하면 -inf가 되기 때문
    return -np.sum(t * np.log(y + delta))

y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
print(cross_entropy_error(np.array(y), np.array(t)))

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t)))