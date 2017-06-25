import numpy
import re
import math
def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        #if e < 0.001:
        break
        
    return P, Q.T


def matrix_factorization_implicit(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    B = numpy.zeros(shape=(4))
    for step in range(steps):
        for i in range(len(R)):
            B[0] = B[1] = B[2] = B[3] = 0
            for j in range(len(R[i])):               
                if R[i][j] > 3 and R[i][j] <= 4:
                    B[3] = B[3] + 1
                elif R[i][j] > 2 and R[i][j] <= 3:
                    B[2] = B[2] + 1
                elif R[i][j] > 1 and R[i][j] <= 2:
                    B[1] = B[1] + 1
                else:
                    B[0] = B[0] + 1
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    if R[i][j] > 3 and R[i][j] <= 4:
                        div = B[3]
                    elif R[i][j] > 2 and R[i][j] <= 3:
                        div = B[2]
                    elif R[i][j] > 1 and R[i][j] <= 2:
                        div = B[1]
                    else:
                        div = B[0]
                    if div != 0:
                        eij = (R[i][j] - numpy.dot(P[i,:],Q[:,j]))/pow(div, 0.5)
                    else:
                        eij = (R[i][j] - numpy.dot(P[i,:],Q[:,j]))
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        if e < 0.001:
            break
    return P, Q.T

def matrix_factorization_implicit_semantic(R, S, P, Q, C, K, Cat, steps=5000, alpha=0.0002, beta=0.02, gamma = 0.00002):
    Q1 = Q
    Q = Q.T
    B = numpy.zeros(shape=(4))
    for step in range(steps):
        for i in range(len(R)):
            B[0] = B[1] = B[2] = B[3] = 0
            for j in range(len(R[i])):               
                if R[i][j] > 3 and R[i][j] <= 4:
                    B[3] = B[3] + 1
                elif R[i][j] > 2 and R[i][j] <= 3:
                    B[2] = B[2] + 1
                elif R[i][j] > 1 and R[i][j] <= 2:
                    B[1] = B[1] + 1
                else:
                    B[0] = B[0] + 1
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    if R[i][j] > 3 and R[i][j] <= 4:
                        div = B[3]
                    elif R[i][j] > 2 and R[i][j] <= 3:
                        div = B[2]
                    elif R[i][j] > 1 and R[i][j] <= 2:
                        div = B[1]
                    else:
                        div = B[0]
                    if div != 0:
                        eij = (R[i][j] - numpy.dot(P[i,:],Q[:,j]))/pow(div, 0.5)
                    else:
                        eij = (R[i][j] - numpy.dot(P[i,:],Q[:,j]))
                    val = 0
                    for s in range(Cat):
                        val = val + S[j][s] - numpy.dot(Q1[j,:],C[:,s])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] + (2 * gamma * val) - beta * Q[k][j])
        eR = numpy.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        if e < 0.001:
            break
    return P, Q.T

def matrix_factorization_implicit_semantic_tag(R, S, T, P, Q, C, K, Cat, Tag, steps=5000, alpha=0.0002, beta=0.02, gamma = 0.00002):
    Q1 = Q
    Q = Q.T
    B = numpy.zeros(shape=(4))
    for step in range(steps):
        for i in range(len(R)):
            B[0] = B[1] = B[2] = B[3] = 0
            for j in range(len(R[i])):               
                if R[i][j] > 3 and R[i][j] <= 4:
                    B[3] = B[3] + 1
                elif R[i][j] > 2 and R[i][j] <= 3:
                    B[2] = B[2] + 1
                elif R[i][j] > 1 and R[i][j] <= 2:
                    B[1] = B[1] + 1
                else:
                    B[0] = B[0] + 1
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    if R[i][j] > 3 and R[i][j] <= 4:
                        div = B[3]
                    elif R[i][j] > 2 and R[i][j] <= 3:
                        div = B[2]
                    elif R[i][j] > 1 and R[i][j] <= 2:
                        div = B[1]
                    else:
                        div = B[0]
                    if div != 0:
                        eij = (R[i][j] - numpy.dot(P[i,:],Q[:,j]))/pow(div, 0.5)
                    else:
                        eij = (R[i][j] - numpy.dot(P[i,:],Q[:,j]))
                    val = 0
                    val1 = 0
                    for s in range(Cat):
                        val = val + S[j][s] - numpy.dot(Q1[j,:],C[:,s])
                    for s in range(Tag):
                        val1 = val1 + T[j][s] - numpy.dot(Q1[j,:],G[:,s])
                        """
                        val1 = val1 + T[j][s] - numpy.dot(Q1[j,:],G[:,s])
                        for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] + (2 * gamma * val) + (2 * gamma * val1) - beta * Q[k][j])
                        """
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] + (2 * gamma * val) + (2 * gamma * val1) - beta * Q[k][j])
        eR = numpy.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        if e < 0.001:
            break
    return P, Q.T

def matrix_factorization_implicit_semantic_tag_time(R, S, T, P, Q, C, K, Cat, Tag, c, steps=5000, alpha=0.0002, beta=0.02, gamma = 0.00002):
    Q1 = Q
    Q = Q.T
    B = numpy.zeros(shape=(4))
    for step in range(steps):
        for i in range(len(R)):
            B[0] = B[1] = B[2] = B[3] = 0
            for j in range(len(R[i])):               
                if R[i][j] > 3 and R[i][j] <= 4:
                    B[3] = B[3] + 1
                elif R[i][j] > 2 and R[i][j] <= 3:
                    B[2] = B[2] + 1
                elif R[i][j] > 1 and R[i][j] <= 2:
                    B[1] = B[1] + 1
                else:
                    B[0] = B[0] + 1
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    if R[i][j] > 3 and R[i][j] <= 4:
                        div = B[3]
                    elif R[i][j] > 2 and R[i][j] <= 3:
                        div = B[2]
                    elif R[i][j] > 1 and R[i][j] <= 2:
                        div = B[1]
                    else:
                        div = B[0]
                    if div != 0:
                        eij = (R[i][j] - numpy.dot(P[i,:],Q[:,j]))/pow(div, 0.5)
                    else:
                        eij = (R[i][j] - numpy.dot(P[i,:],Q[:,j]))
                    val = 0
                    val1 = 0
                    for s in range(Cat):
                        val = val + S[j][s] - numpy.dot(Q1[j,:],C[:,s])
                    for s in range(Tag):
                        val1 = val1 + T[j][s] - numpy.dot(Q1[j,:],G[:,s])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] + (2 * gamma * val) + c[k][j] * (2 * gamma * val1) - beta * Q[k][j])
        eR = numpy.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        if e < 0.001:
            break
    return P, Q.T


R = [
     [5,3,0,1],
     [4,0,0,1],
     [1,1,0,5],
     [1,0,0,4],
     [0,1,5,4],
    ]
#count = numpy.zeros(shape=(2105,18750))
count = [
        [0,0,15,20],
        [40,10,0,10],
        [10,0,20,15],
        [10,10,0,40],
        [0,0,0,40]
        ]

S = [
    [0, 0.14, 0.22],
    [0, 0.14, 0.22],
    [0, 0.14, 0.22],
    [0, 0.14, 0.22]
    ]
T = [
    [10, 20, 20],
    [10, 20, 20],
    [10, 20, 20],
    [10, 20, 20]
    ]
K = 2
Cat = 3
Tag = 3
Ta = numpy.zeros(shape=(3))
for zz in range(len(T[0])):
    for zz1 in range(len(T)):
        Ta[zz] = Ta[zz] + T[zz1][zz]

print(Ta)
for zz in range(len(T)):
    for zz1 in range(len(T[0])):
        b = T[zz][zz1]-1
        a = Ta[zz1]
        T[zz][zz1] = round(math.log10((Tag/b))/a, 2)

'''
print(count)
che = 0
with open("user_artists.dat", "r") as f:
#lines = [line.rstrip('\n') for line in open('user_artists.dat')]
#print(lines)
    for line in f:
        if che == 1:
            final = line.rstrip('\n')
            ff = re.split(r'\t+', final.rstrip('\t'))
            a = int(ff[0])
            b = int(ff[1])
            c = int(ff[2])
            count[a][b] = c
            #print(count[a][b])
        che = 1
'''
count = numpy.array(count)
print(count)

#freq = numpy.zeros(shape=(2105,18750))
freq = numpy.zeros(shape=(5,4))
for i in range(len(count)):
    sum = 0
    for j in range(len(count[i])):
        sum = sum + count[i][j]
    for j in range(len(count[i])):
        #print(sum)
        if count[i][j] != 0.0:
            #print(count[i][j])
        #print(count[i][j])
            freq[i][j] = round(count[i][j]/sum, 2)
        #print(freq[i][j])
    arr = sorted(freq[i], reverse=True)
    #arr.sort(reverse=True)
    sum = 0
    d = dict()
    if arr[0] != 0:
        for j in range(len(arr)):
            if j != 0:
                sum = sum + arr[j-1]
            if j != 0 and arr[j] != arr[j-1]:
                d[arr[j]] = 4 * (1 - sum)
            if j == 0:
                d[arr[j]] = 4
        for j in range(len(count[i])):
            freq[i][j] = d[freq[i][j]]
            if freq[i][j] < 0:
                freq[i][j] = 0
    #print(d)
    
freq = numpy.array(freq)
print(freq)
R = numpy.array(R)



#N = len(R)
#M = len(R[0])
N = len(freq)
M = len(freq[0])

P = numpy.random.rand(N,K)
Q = numpy.random.rand(M,K)
C = numpy.random.rand(K,Cat)
G = numpy.random.rand(K, Tag)
#print(P)
#print(Q)
#nP, nQ = matrix_factorization_implicit_semantic_tag(freq, S, T, P, Q, C, K, Cat, Tag)


dummyTags = [
    [1, 1, 1, 1],
    [1, 2, 1, 2],
    [1, 3, 2, 0],
    [1, 4, 1, 3],
    [2, 2, 1, 1],
    [2, 3, 3, 1],
    [2, 4, 4, 2],
    [3, 3, 1, 0],
    [3, 4, 4, 1],
    [4, 4, 4, 2]
]
artist = [0, 1, 2, 3]
tags = [1, 2, 3, 4]
c = numpy.zeros(shape = (len(artist), len(tags)))
postScore = numpy.zeros(shape = (len(artist), len(tags)))
tagSpecificity = numpy.zeros(shape = (len(artist), len(tags)))

for i in range(len(dummyTags)):
    artist_id = dummyTags[i][1] - 1
    time = dummyTags[i][3]
    tag_id = dummyTags[i][2] - 1
    postScore[artist_id][tag_id] = postScore[artist_id][tag_id] + 0.9**time
    tagSpecificity[artist_id][tag_id] = tagSpecificity[artist_id][tag_id] + 1

for  i in range(len(tagSpecificity)):
    for j in range(len(tagSpecificity[i])):
        tagSpecificity[i][j] = math.log10(tagSpecificity[i][j] + 50)
    
for i in range(len(artist)):
    for j in range(len(tags)):
        c[i][j] = postScore[i][j] / tagSpecificity[i][j]
        print(c[i][j], end=" ")
    print()


nP, nQ = matrix_factorization_implicit_semantic_tag_time(freq, S, T, P, Q, C, K, Cat, Tag, c)

#nP, nQ = matrix_factorization_implicit_semantic(freq, S, P, Q, C, K, Cat)
#nP, nQ = matrix_factorization_implicit(freq, P, Q, K)
#nP, nQ = matrix_factorization(R, P, Q, K)
nR = numpy.dot(nP, nQ.T)
print(nR)
