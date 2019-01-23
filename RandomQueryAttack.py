import sys
#import statistics
import numpy as np
from scipy.spatial import distance import matplotlib.pyplot as plt
'''
The random query attack '''
def randQueryAttack(n, m, sigma):
    # when m = 1.1n, we need to round it
    m = round(m)
    int(m)
    # Generate x
    x = np.random.choice(a=[0, 1], size=(n, 1), p=[1/2, 1/2]) # Generate Matrix B
    B = np.random.randint(2, size=(m, n))
    tmp = B.dot(1 / n * x)
    # Generate Y
    Y = np.random.uniform(low=0, high = sigma ** 2, size=(m, 1)) # Compute answer matrix a
    a = tmp + Y
    # Compute matrix z
    z = np.linalg.lstsq(((1 / n) * B), a)[0]
    for i in range(n):
        if z[i] < 1 / 2:
            z[i] = 0
        else:
            z[i] = 1
    ham = distance.hamming(x, z)
    return ham

def testAttack(n, m, sigma):
    count = 0
    hamList = [0] * 20
    while count < 20:
        hamList[count] = randQueryAttack(n, m, sigma)
        count += 1
    stdDeviation = np.std(hamList, ddof=1)
    mean = np.mean(hamList, dtype=np.float64)
    print('For n =', n, 'and sigma =', sigma, '\n')
    print('Hamming Distance for each randaom attack =', hamList, '\n') print('Mean =', mean, '\n')
    print('Standard Deviation =', stdDeviation, ':\n') print('------------------------------', '\n')
    return (mean, stdDeviation)
# n ∈ {128,512,2048,8192}
# theta from 2 ^ -1 down to 2 ^ -4 # m ∈ {1.1n, 4n, 16n}
# For every sigma, display plotGraph
'''
# Plotting.

def Graph(n, mList, sigmaList, meanList, stdList):
    fig, ax = plt.subplots()
    x_pos = np.arange(len(sigmaList))
    ax.bar(sigmaList, meanList, yerr=stdList, align='center', alpha=0.5,
ecolor='black', capsize=10) 
    ax.set_ylabel('...') 
    #ax.set_xticks(sigma) 
    ax.set_title('Title') 
    ax.yaxis.grid(True)
    
# Test
nList = [128, 512, 2048, 8192]
mList = [1.1 * 128, 4 * 128, 16 * 128] 
sigmaList = [1/4, 1/8, 1/16, 1/32, 1/64] 
for k in range(5):
    meanList = [] stdList = []
    # For every n, m 
    for i in range (2):
        for j in range(2):
            mean, std = testAttack(nList[i], mList[j], sigmaList[k]) 
            meanList.append(mean)
            stdList.append(std)
    Graph(nList, mList, sigmaList, meanList, stdList) '''

testAttack(128, 1.1 * 512, 1 / 16)
testAttack(512, 4 * 512, 1 / 8)
testAttack(512, 16 * 512, 1 / 4)
testAttack(512, 1.1 * 512, 1 / 32)
testAttack(512, 4 * 512, 1 / 16)
testAttack(512, 16 * 512, 1 / 8)
testAttack(2048, 4 * 2048, 1 / 64)
testAttack(2048, 4 * 2048, 1 / 32)
testAttack(2048, 16 * 2048, 1 / 16)
# for n = 8192, it takes a long time to run. #testAttack(8192, 1.1 * 2048, 1 / 64) #testAttack(8192, 4 * 2048, 1 / 32) #testAttack(8192, 16 * 2048, 1 / 16)