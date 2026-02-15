# tao ma tran
import numpy as np

# tao ma tran chu B
B = np.array([
    [1,1,1,0,0],
    [1,0,0,1,0],
    [1,1,1,0,0],
    [1,0,0,1,0],
    [1,1,1,0,0]
])
# bien thanh mang 1 chieu
B = B.reshape(25)

C = np.array([
    [0,1,1,1,0],
    [1,0,0,0,1],
    [1,0,0,0,0],
    [1,0,0,0,1],
    [0,1,1,1,0]
]).reshape(25)


tap_huan_luyen = np.array([B,C])
answer_label = np.array([1,0]) # 1 la B, 0 la C


# khoi tao mang no ron
weights = np.random.rand(25) # 25 la so luong dac trung
learning_rate = 0.1
bias = np.random.randn() # khoi tao bias ngau nhien

# ham sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# huan luyen mang no ron
for epoch in range(1000):
    for i in range(len(tap_huan_luyen)):
        # tinh tong cong
        linear_output = np.dot(weights, tap_huan_luyen[i]) + bias
        predicted = sigmoid(linear_output)

        # tinh loi
        error = predicted - answer_label[i]

        # cap nhat trong so va bias
        weights -= learning_rate * error * tap_huan_luyen[i]
        bias -= learning_rate * error

test = B
z = np.dot(test, weights) + bias
predicted = sigmoid(z)

if predicted > 0.5:
    print("Là chữ B")
else:
    print("Không phải B")

