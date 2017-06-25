import numpy
import re
import math
import csv
from datetime import date

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
            #print("BINNING DONE")
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
                    #print("semantic done")
                    for s in range(Tag):
                        val1 = val1 + c[j][s] * (T[j][s] - numpy.dot(Q1[j,:],G[:,s]))
                    #print("tag done")
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] + (2 * gamma * val) + (2 * gamma * val1) - beta * Q[k][j])
                    #print("Phase " + str(j) + "Completed outof " + str(len(R[i])))
            print("Phase " + str(i) + "Completed outof " + str(len(R)))
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


R = [
     [5,3,0,1],
     [4,0,0,1],
     [1,1,0,5],
     [1,0,0,4],
     [0,1,5,4],
    ]
count = numpy.zeros(shape=(2101,501))



S = numpy.zeros(shape=(501,5))
j = 0
i = 0
with open('mappedPCA5.csv') as csvfile:
    readCSV = csv.reader(csvfile,delimiter = ',')
    print(readCSV)
    j = 0
    for row in readCSV:
        for i in range(5):
            S[j][i] = row[i]
        if (j < 500):
                j = j + 1    
print(S)         
            
'''
count = [
        [0,0,15,20],
        [40,10,0,10],
        [10,0,20,15],
        [10,10,0,40],
        [0,0,0,40]
        ]
'''
'''S = [
    [0, 0.14, 0.22],
    [0, 0.14, 0.22],
    [0, 0.14, 0.22],
    [0, 0.14, 0.22]
    ]'''
'''
T = [
    [10, 20, 20],
    [10, 20, 20],
    [10, 20, 20],
    [10, 20, 20]
    ]
'''



artist_count = 501
tag_count = 101
curr_year = int(date.today().year)

tags = {}

with open("tag1.dat") as f:
    flag = 0;
    for line in f:
        if (flag == 0):
            flag = 1
            continue
        words = line.split()
        tags[int(words[0])] = 1
    #print(tags)
    #print(tags[1])


T = numpy.zeros(shape = (artist_count, tag_count))

with open("user_tag_data_new.dat") as f:
    flag = 0
    for line in f:
        words = line.split()
        if (flag == 0):
            flag = 1
            continue
        artist_id = int(words[1])
        tag_id = int(words[2])
        #print(artist_id, end=" ")
        #print(tag_id, end=" ")
        #print(time)
        if (tag_id in tags and tag_id < 101):
            tag_id = tag_id
            T[artist_id][tag_id] = T[artist_id][tag_id] + 1


print("dhruv")
print(T)


K = 2
Cat = 5
Tag = tag_count
Ta = numpy.zeros(shape=(Tag))
for zz in range(len(T[0])):
    for zz1 in range(len(T)):
        Ta[zz] = Ta[zz] + T[zz1][zz]

print(Ta)
for zz in range(len(T)):
    for zz1 in range(len(T[0])):
        b = T[zz][zz1] - 1
        a = Ta[zz1]
        #print(a)
        #print(b)
        if (a > 0 and b > 0):
            T[zz][zz1] = round(math.log10((Tag/b))/a, 2)


print(count)
che = 0
with open("user_artist_short.dat", "r") as f:
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

count = numpy.array(count)
print(count)

freq = numpy.zeros(shape=(2101,501))
#freq = numpy.zeros(shape=(5,4))
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



'''
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

artist = numpy.ndarray((artist_count, ), int)
tags = numpy.ndarray((tag_count, ), int)

for i in range(0, artist_count):
    artist [i] = i

for i in range(0, tag_count):
    tags [i] = i
'''


tags = {}

with open("tag1.dat") as f:
    flag = 0;
    for line in f:
        if (flag == 0):
            flag = 1
            continue
        words = line.split()
        tags[int(words[0])] = 1
    #print(tags)
    #print(tags[1])


c = numpy.zeros(shape = (artist_count, tag_count))
postScore = numpy.zeros(shape = (artist_count, tag_count))
tagSpecificity = numpy.zeros(shape = (artist_count, tag_count))

with open("user_tag_data_new.dat") as f:
    flag = 0
    for line in f:
        words = line.split()
        if (flag == 0):
            flag = 1
            continue
        artist_id = int(words[1])
        tag_id = int(words[2])
        time = curr_year - int(words[5])
        #print(artist_id, end=" ")
        #print(tag_id, end=" ")
        #print(time)
        if (tag_id in tags and tag_id < 101):
            tag_id = tag_id
            postScore[artist_id][tag_id] = postScore[artist_id][tag_id] + 0.9**time
            tagSpecificity[artist_id][tag_id] = tagSpecificity[artist_id][tag_id] + 1


'''
for i in range(len(dummyTags)):
    artist_id = dummyTags[i][1] - 1
    time = dummyTags[i][3]
    tag_id = dummyTags[i][2] - 1
    postScore[artist_id][tag_id] = postScore[artist_id][tag_id] + 0.9**time
    tagSpecificity[artist_id][tag_id] = tagSpecificity[artist_id][tag_id] + 1
'''

for  i in range(len(tagSpecificity)):
    for j in range(len(tagSpecificity[i])):
        tagSpecificity[i][j] = math.log10(tagSpecificity[i][j] + 50)
    
for i in range(artist_count):
    for j in range(tag_count):
        c[i][j] = postScore[i][j] / tagSpecificity[i][j]
        #print(c[i][j], end=" ")
    #print("c[i][j] calculated")

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
nP, nQ = matrix_factorization_implicit_semantic_tag_time(freq, S, T, P, Q, C, K, Cat, Tag, c)
#nP, nQ = matrix_factorization_implicit_semantic_tag(freq, S, T, P, Q, C, K, Cat, Tag)
#nP, nQ = matrix_factorization_implicit_semantic(freq, S, P, Q, C, K, Cat)
#nP, nQ = matrix_factorization_implicit(freq, P, Q, K)
#nP, nQ = matrix_factorization(R, P, Q, K)
nR = numpy.dot(nP, nQ.T)
print(nR)
