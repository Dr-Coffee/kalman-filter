import numpy as np
import matplotlib.pyplot as plt

tDelta = 0.1
t = np.arange(0, 5, tDelta)
C_n = len(t)
C_g = 10
sz = [2, C_n]

x = 0.5 * C_g * np.square(t)
z = x + np.sqrt(10) * np.random.randn(C_n)
plt.plot(t, x)
plt.plot(t, z)

Q = np.array([[0,0],[0,9e-1]])
R = 10

A = np.array([[1, tDelta],
              [0, 1]])
B = np.array([0.5*np.square(tDelta), tDelta])
H = np.array([1, 0])

xHat = np.zeros(sz)         #estimated value
xHatMinus = np.zeros(sz)    #predicted value
PMinus = np.zeros(sz)       #covariance matrix between predicted and real values

P = np.array([[0, 0], [0, 0]])
K = np.zeros([2, 1])
I = np.eye(2)

for k in range(9, C_n):
    #
    xHatMinus[:, k] = np.dot(A, xHat[:, k-1]) + np.dot(B, C_g)
    PMinus = np.dot(np.dot(A, P), np.matrix.transpose(A)) + Q
    #
    K = np.dot(np.dot(PMinus, np.matrix.transpose(H)),
               1./(np.dot(np.dot(H, PMinus), np.matrix.transpose(H)) + R))
    xHat[:, k] = xHatMinus[:, k] + np.dot(K, z[k]-np.dot(H, xHatMinus[:, k]))
    P = np.dot(I - np.dot(K, H), PMinus)
plt.plot(t, xHat[0,:])

plt.show()