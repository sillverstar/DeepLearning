# 오차제곱합(SSE, sum of squares for error)

# (데이터 하나에 대한 SSE) = 1/2 ∑(y_k - t_k)^2
# (y_k = 신경망의 출력, t_k = 정답 레이블, k = 데이터의 차원 수)

import numpy as np

y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0] # 소프트 맥스 함수의 출력
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] # 정답 레이블(one-hot Encoding)

def sum_squares_error(y, t):
    return 0.5 * np.sum((y-t) ** 2)

print(sum_squares_error(np.array(y), np.array(t)))

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(sum_squares_error(np.array(y), np.array(t)))