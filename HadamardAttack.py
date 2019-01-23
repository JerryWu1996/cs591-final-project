import sys
import numpy as np
from scipy.linalg import hadamard from scipy.spatial import distance from matplotlib.pyplot import errorbar
'''
The Hadamard attack '''
def HadamardAttack(n,sigma): # Generate x
    x = np.random.choice(a=[0, 1], size=(n, 1), p=[1/2, 1/2]) # Generate Hadamard Matrix H
    H = hadamard(n)
    #tmp = H.dot(np.true_divide(x, n))
    tmp = H.dot(1 / n * x)
    # Generate Y
    Y = np.random.uniform(low = 0, high = sigma ** 2, size=(n, 1)) # Compute answer matrix a
    a = tmp + Y
    # Compute matrix z
    z = H.dot(a)
    for i in range(n):
        if z[i] < 1 / 2: z[i] = 0
    else:
        z[i] = 1

    ham = distance.hamming(x, z)
    return ham


def testAttack(n, sigma):
    count = 0
    hamList = [0] * 20
    while count < 20:
        hamList[count] = HadamardAttack(n, sigma)
        count += 1
    mean = np.mean(hamList, dtype=np.float64)
    stdDeviation = np.std(hamList, ddof=1)
    print('For n =', n, 'and sigma =', sigma, '\n')
    print('Hamming Distance for each randaom attack =', hamList, '\n')
    print('Mean =', mean, '\n')
    print('Standard Deviation =', stdDeviation, ':\n')
    print('------------------------------', '\n')
# n ∈ {128,512,2048,8192}
# sigma from 2 ^ -1 down to 2 ^ -4
# arguments: testAttack(n, sigma)
# the last sigma that isn't perfect + the sigma that we achieve a perfect attack testAttack(128, 1 / 8)
testAttack(128, 1 / 16)
testAttack(512, 1 / 16)
testAttack(512, 1 / 32)
testAttack(2048, 1 / 32)
testAttack(2048, 1 / 64)
testAttack(8192, 1 / 64)
testAttack(8192, 1 / 128)