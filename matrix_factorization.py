import numpy
import re
import math
import csv
import random
from datetime import date


artist_count = 1000
tag_count = 500
curr_year = int(date.today().year)
user_count = 2100
lbd = 0.75
CNT = 0
CNT_TEST = 0
training_error = 0

def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.003, beta=0.001): #just the basic implemtation of the matrix factorization
    Q = Q.T
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
            #print("process " + str(i) + " completed of " + str(len(R)))
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


'def matrix_factorization_implicit(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02): #implentation of matrix factorization with the implicit data
    Q = Q.T
    B = numpy.zeros(shape=(4))
    for step in range(steps):
        #Binning done ahead
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
            print("process " + str(i) + " completed of " + str(len(R)))
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
    return P, Q
    
#matrix factorization with the semantic data
def matrix_factorization_implicit_semantic(G, R, S, T, P, Q, C, K, Cat, Tag, c, userMatrix, CNT, alpha = 0.003, steps=300, beta=0.001, gamma = 0.003):
    Q1 = Q
    Q = Q.T
    B = numpy.zeros(shape=(4))
    R = R / 4

    #print (CNT)
    #print (CNT_TEST)

    Q1G = numpy.zeros(shape = (artist_count, tag_count))
    Q1C = numpy.zeros(shape = (artist_count, tag_count))
    catVal = numpy.zeros(shape = (artist_count))
    tagVal = numpy.zeros(shape = (artist_count))
    


    for j in range(len(R[0])):
        for s in range(Tag):
            Q1G[j][s] = numpy.dot(Q1[j,:],G[:,s])#calculating pg eq12
            tagVal[j] = tagVal[j] + c[j][s] * (T[j][s] - Q1G[j][s]) #caluclating c(T-pg) eq12
    for j in range(len(R[0])):
        for s in range(Cat):
            Q1C[j][s] = numpy.dot(Q1[j,:],C[:,s])#caluclating pc eq6
            catVal[j] = catVal[j] + S[j][s] - Q1C[j][s]#calculating (S-pc) eq6    
        
    iterations = 1
    print("Pre calculations completed...")
    for it in range(iterations):
        for step in range(steps):
            #Binning
            check_cnt = 0
            for i in range(len(userMatrix)):
                B[0] = B[1] = B[2] = B[3] = 0
                for j in userMatrix[i]:
                    if (R[i][j] > 0):
                        if R[i][j] > 0.75 and R[i][j] <= 1:
                            B[3] = B[3] + 1
                        elif R[i][j] > 0.5 and R[i][j] <= 0.75:
                            B[2] = B[2] + 1
                        elif R[i][j] > 0.25 and R[i][j] <= 0.5:
                            B[1] = B[1] + 1
                        else:
                            B[0] = B[0] + 1
                for j in userMatrix[i]:
                    if R[i][j] > 0:
                        val = catVal[j]
                        val1 = tagVal[j]
                        if R[i][j] > 0.75 and R[i][j] <= 1:
                           div = B[3]
                        elif R[i][j] > 0.5 and R[i][j] <= 0.75:
                            div = B[2]
                        elif R[i][j] > 0.25 and R[i][j] <= 0.5:
                            div = B[1]
                        else:
                            div = B[0]
                        if div != 0:
                            #calculating error for the updation of P and Q matrix
                            eij = (R[i][j] - numpy.dot(P[i,:],Q[:,j]))/pow(div, 0.5) + (2 * gamma * val)
                        else:
                            eij = (R[i][j] - numpy.dot(P[i,:],Q[:,j])) + (2 * gamma * val)
                        
                        
                        for k in range(K):
                            #updating P
                            P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                            #updating Q
                            Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
            eR = numpy.dot(P,Q)
            e = 0
            
            for i in range(len(userMatrix)):
                for j in userMatrix[i]:
                    if R[i][j] > 0:
                        e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                        for k in range(K):
                            if (R[i][j]  != 0):
                                #calculating the RMSE error
                                e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2)) / R[i][j]
            e = e / CNT
            e = pow(e, 1/2)
            if e < 0.001:
                break
            training_error = e
            print ("completed " + str(step) + " out of " + str(steps) + " error = " + str(e))
        print ('Error on training data = '+ str(training_error))
    print("Returning...")
    return P, Q
#matrix factorization with the tagtime data
def matrix_factorization_implicit_tag_time(G, R, S, T, P, Q, C, K, Cat, Tag, c, userMatrix, CNT, alpha = 0.003, steps=300, beta=0.001, gamma = 0.003):
    Q1 = Q
    Q = Q.T
    B = numpy.zeros(shape=(4))
    R = R / 4

    Q1G = numpy.zeros(shape = (artist_count, tag_count))
    Q1C = numpy.zeros(shape = (artist_count, tag_count))
    catVal = numpy.zeros(shape = (artist_count))
    tagVal = numpy.zeros(shape = (artist_count))
    
    for j in range(len(R[0])):
        for s in range(Tag):
            Q1G[j][s] = numpy.dot(Q1[j,:],G[:,s])
            tagVal[j] = tagVal[j] + c[j][s] * (T[j][s] - Q1G[j][s])

    for j in range(len(R[0])):
        for s in range(Cat):
            Q1C[j][s] = numpy.dot(Q1[j,:],C[:,s])
            catVal[j] = catVal[j] + S[j][s] - Q1C[j][s]
            
        
    iterations = 1
    print("Pre calculations completed...")
    for it in range(iterations):
        for step in range(steps):
            check_cnt = 0
            for i in range(len(userMatrix)):
                B[0] = B[1] = B[2] = B[3] = 0
                for j in userMatrix[i]:
                    if (R[i][j] > 0):
                        if R[i][j] > 0.75 and R[i][j] <= 1:
                            B[3] = B[3] + 1
                        elif R[i][j] > 0.5 and R[i][j] <= 0.75:
                            B[2] = B[2] + 1
                        elif R[i][j] > 0.25 and R[i][j] <= 0.5:
                            B[1] = B[1] + 1
                        else:
                            B[0] = B[0] + 1
                for j in userMatrix[i]:
                    if R[i][j] > 0:
                        val = catVal[j]
                        val1 = tagVal[j]
                        if R[i][j] > 0.75 and R[i][j] <= 1:
                           div = B[3]
                        elif R[i][j] > 0.5 and R[i][j] <= 0.75:
                            div = B[2]
                        elif R[i][j] > 0.25 and R[i][j] <= 0.5:
                            div = B[1]
                        else:
                            div = B[0]
                        if div != 0:
                            eij = (R[i][j] - numpy.dot(P[i,:],Q[:,j]))/pow(div, 0.5) + (2 * gamma * val1) #added the tagtime error
                        else:
                            eij = (R[i][j] - numpy.dot(P[i,:],Q[:,j])) + (2 * gamma * val1)
                        
                        for k in range(K):
                            P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                            if (P[i][k] < 0):
                                P[i][k] = 0    
                            Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
                            if (Q[k][j] < 0):
                                Q[k][j] = 0
    
            eR = numpy.dot(P,Q)
            e = 0
            
            for i in range(len(userMatrix)):
                for j in userMatrix[i]:
                    if R[i][j] > 0:
                        e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                        #print(R[i][j] - numpy.dot(P[i,:],Q[:,j]))
                        for k in range(K):
                            if (R[i][j]  != 0):
                                e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2)) / R[i][j]
            e = e / CNT
            e = pow(e, 1/2)
            if e < 0.001:
                break
            training_error = e
            print ("completed " + str(step) + " out of " + str(steps) + " error = " + str(e))
        #beta = beta + 0.008
        print ('Error on training data = '+ str(training_error))
    print("Returning...")
    return P, Q

#matrix factorization with the semantic+tagtime data
def matrix_factorization_implicit_semantic_tag_time(G, R, S, T, P, Q, C, K, Cat, Tag, c, userMatrix, CNT, alpha = 0.003, steps=300, beta=0.001, gamma = 0.003):
    Q1 = Q
    Q = Q.T
    B = numpy.zeros(shape=(4))
    R = R / 4

    Q1G = numpy.zeros(shape = (artist_count, tag_count))
    Q1C = numpy.zeros(shape = (artist_count, tag_count))
    catVal = numpy.zeros(shape = (artist_count))
    tagVal = numpy.zeros(shape = (artist_count))

    for j in range(len(R[0])):
        for s in range(Tag):
            Q1G[j][s] = numpy.dot(Q1[j,:],G[:,s])
            tagVal[j] = tagVal[j] + c[j][s] * (T[j][s] - Q1G[j][s])
            
    for j in range(len(R[0])):
        for s in range(Cat):
            Q1C[j][s] = numpy.dot(Q1[j,:],C[:,s])
            catVal[j] = catVal[j] + S[j][s] - Q1C[j][s]
    
    iterations = 1
    print("Pre calculations completed...")
    for it in range(iterations):
        for step in range(steps):
            if (step % 50 == 0 and step != 0):
                for j in range(len(R[0])):
                    for s in range(Tag):
                        Q1G[j][s] = numpy.dot(Q1[j,:],G[:,s])
                        tagVal[j] = tagVal[j] + c[j][s] * (T[j][s] - Q1G[j][s])
                        
                for j in range(len(R[0])):
                    for s in range(Cat):
                        Q1C[j][s] = numpy.dot(Q1[j,:],C[:,s])
                        catVal[j] = catVal[j] + S[j][s] - Q1C[j][s]
            check_cnt = 0
            for i in range(len(userMatrix)):
                B[0] = B[1] = B[2] = B[3] = 0
                for j in userMatrix[i]:
                    if (R[i][j] > 0):
                        if R[i][j] > 0.75 and R[i][j] <= 1:
                            B[3] = B[3] + 1
                        elif R[i][j] > 0.5 and R[i][j] <= 0.75:
                            B[2] = B[2] + 1
                        elif R[i][j] > 0.25 and R[i][j] <= 0.5:
                            B[1] = B[1] + 1
                        else:
                            B[0] = B[0] + 1
                for j in userMatrix[i]:
                    if R[i][j] > 0:
                        val = catVal[j]
                        val1 = tagVal[j]
                        if R[i][j] > 0.75 and R[i][j] <= 1:
                           div = B[3]
                        elif R[i][j] > 0.5 and R[i][j] <= 0.75:
                            div = B[2]
                        elif R[i][j] > 0.25 and R[i][j] <= 0.5:
                            div = B[1]
                        else:
                            div = B[0]
                        if div != 0:
                            eij = (R[i][j] - numpy.dot(P[i,:],Q[:,j]))/pow(div, 0.5) + (2 * gamma * val) + (2 * gamma * val1)#added the semantic+tagtime error
                        else:
                            eij = (R[i][j] - numpy.dot(P[i,:],Q[:,j])) + (2 * gamma * val) + (2 * gamma * val1)
                        for k in range(K):
                            P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])  
                            Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
            eR = numpy.dot(P,Q)
            e = 0
            for i in range(len(userMatrix)):
                for j in userMatrix[i]:
                    if R[i][j] > 0:
                        e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                        for k in range(K):
                            if (R[i][j]  != 0):
                                e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2)) / R[i][j]
            e = e / CNT
            e = pow(e, 1/2)
            if e < 0.001:
                break
            training_error = e
            print ("completed " + str(step) + " out of " + str(steps) + " error = " + str(e))
        print ('Error on training data = '+ str(training_error))
    print("Returning...")
    return P, Q


R = []

count = numpy.zeros(shape=(user_count,artist_count))

S = numpy.zeros(shape=(artist_count, 5))
j = 0
i = 0
#the categories data entered
with open('mappedPCA5.csv') as csvfile:
    readCSV = csv.reader(csvfile,delimiter = ',')
    j = 0
    for row in readCSV:
        for i in range(5):
            S[j][i] = row[i]
        if (j < artist_count - 1):
                j = j + 1    
print("Artist X Categories has been succesfully calculated...")         
            
#the tag data entered
tags = {}
with open("tag1.dat") as f:
    flag = 0;
    for line in f:
        if (flag == 0):
            flag = 1
            continue
        words = line.split()
        tags[int(words[0])] = 1
    
T = numpy.zeros(shape = (artist_count, tag_count))
#T matrix as used in eq 12 intialized then tag frequency calculated
with open("user_tag_data.dat") as f:
    flag = 0
    for line in f:
        words = line.split()
        if (flag == 0):
            flag = 1
            continue
        artist_id = int(words[1])
        tag_id = int(words[2])
        if (tag_id in tags and tag_id < tag_count and artist_id < artist_count):
            tag_id = tag_id
            T[artist_id][tag_id] = T[artist_id][tag_id] + 1


print("Artists X Tags frequency has been succesfully calculated...")

K = 5
Cat = 5
Tag = tag_count
Ta = numpy.zeros(shape=(Tag))
for i in range(len(T[0])):
    for j in range(len(T)):
        Ta[i] = Ta[i] + T[j][i]

for i in range(len(T)):
    for j in range(len(T[0])):
        b = T[i][j] - 1
        a = Ta[j]
        if (a > 0 and b > 0):
            T[i][j] = round(math.log10((Tag/b))/a, 2) #eq8 calculated

print("Artists X Tags (T) has been succesfully calculated...")

st = 1
che = 0
for ri in range(4):
    #implicit data being taken as input both for test and training
    CNT = 0
    CNT_TEST = 0
    userMatrix = [()] * user_count
    userMatrixTest = [()] * user_count
    with open("user_artists.dat", "r") as f:
        for line in f:
            r = random.randint(0, 4);
            if che == 1:
                final = line.rstrip('\n')
                ff = re.split(r'\t+', final.rstrip('\t'))
                a = int(ff[0])
                b = int(ff[1])
                c = int(ff[2])
                if (a < user_count and b < artist_count and r <= ri):
                    count[a][b] = c
                    userMatrix[a] = userMatrix[a] + (b,)
                    CNT = CNT + 1
                if (a < user_count and b < artist_count and r >= ri + 1):
                    count[a][b] = c
                    userMatrixTest[a] = userMatrixTest[a] + (b,)
                    CNT_TEST = CNT_TEST + 1
            che = 1
    f.close()
    che = 0
    #print("Ratings has been succesfully inputted...")
    count = numpy.array(count)
    freq = numpy.zeros(shape=(user_count, artist_count))
    tr = numpy.zeros(shape=(user_count, artist_count))

    #implicit data being processed for training
    for i in range(len(userMatrix)):
        sum = 0
        cc = 0
        for j in userMatrix[i]:
            sum = sum + count[i][j]
            cc = cc + 1
        #print(i)
        #print(cc)
        if (sum > 0):
            for j in userMatrix[i]:
                if count[i][j] != 0.0:
                    freq[i][j] = round(count[i][j]/sum, 2)
                    tr[i][j] = freq[i][j]
        

        arr = [None] * len(userMatrix[i])
        arr_idx = 0
        for j in userMatrix[i]:
            arr[arr_idx] = freq[i][j]
            arr_idx = arr_idx + 1
        arr.sort(reverse=True)
        sum = 0
        d = dict()
        if (len(userMatrix[i]) == 0):
            continue
        #rating being given to implicit data matrix between 0 and 4, freq holds these rating
        if arr[0] != 0:
            for j in range(len(arr)):
                if (arr[j] == 0):
                    break
                if j != 0:
                    sum = sum + arr[j-1]
                if j != 0 and arr[j] != arr[j-1]:
                    d[arr[j]] = 4 * (1 - sum)
                if j == 0:
                    d[arr[j]] = 4
            for j in userMatrix[i]:
                if (freq[i][j] <= 0):
                    freq[i][j] = 0
                else:
                    freq[i][j] = d[freq[i][j]]

    #same process as above happening again for test data
    for i in range(len(userMatrixTest)):
        sum = 0
        for j in userMatrixTest[i]:
            sum = sum + count[i][j]
        if (sum > 0):
            for j in userMatrixTest[i]:
                if count[i][j] != 0.0:
                    freq[i][j] = round(count[i][j]/sum, 2)
                    tr[i][j] = count[i][j]/sum

        arr = [None] * len(userMatrixTest[i])
        arr_idx = 0
        for j in userMatrixTest[i]:
            arr[arr_idx] = freq[i][j]
            arr_idx = arr_idx + 1
        arr.sort(reverse=True)
        sum = 0
        d = dict()
        if (len(userMatrixTest[i]) == 0):
            continue
        if arr[0] != 0:
            for j in range(len(arr)):
                if (arr[j] == 0):
                    break
                if j != 0 and arr[j] != arr[j-1]:
                    d[arr[j]] = 4 * (1 - sum)
                if j != 0:
                    sum = sum + arr[j-1]
                if j == 0:
                    d[arr[j]] = 3
            #print (sum)
            for j in userMatrixTest[i]:
                if (freq[i][j] <= 0):
                    freq[i][j] = 0
                else:
                    freq[i][j] = d[freq[i][j]]
        
    freq = numpy.array(freq)
    
    #print("Frequency has been succesfully calculated...")
    R = numpy.array(R)

    tags = {}

    with open("tag1.dat") as f:
        flag = 0;
        for line in f:
            if (flag == 0):
                flag = 1
                continue
            words = line.split()
            tags[int(words[0])] = 1

    N = len(freq)
    M = len(freq[0])
    C = numpy.random.rand(K, Cat)
    G = numpy.random.rand(K, Tag)
    CXT = numpy.dot(S.T, T)
    #matrix factorization being applied on c, g so they are not just random values
    C, G = matrix_factorization(CXT, C.T, G.T, K)
    C = C.T
    G = G.T
    P = numpy.random.rand(N,K)
    P = P * 0.5
    Q = numpy.random.rand(M,K)
    Q = Q * 0.4

    #tagspecivity and c[i][j[ being calculated as in eq 10 and 11
    c = numpy.zeros(shape = (artist_count, tag_count))
    postScore = numpy.zeros(shape = (artist_count, tag_count))
    tagSpecificity = numpy.zeros(shape = (tag_count))
    time_ratings = freq
    with open("user_tag_data.dat") as f:
        flag = 0
        for line in f:
            words = line.split()
            if (flag == 0):
                flag = 1
                continue
            user_id = int(words[0])
            artist_id = int(words[1])
            tag_id = int(words[2])
            time = curr_year - int(words[5])
            if (tag_id in tags and tag_id < tag_count and artist_id < artist_count):
                tag_id = tag_id
                time_ratings[i][j] = time_ratings[i][j] * (0.9) ** time
                postScore[artist_id][tag_id] = postScore[artist_id][tag_id] + lbd**time #eq 9
                tagSpecificity[tag_id] = tagSpecificity[tag_id] + 1

    for i in range(tag_count):
        tagSpecificity[i] = math.log10(tagSpecificity[i] + 50) #eq 10
            
    for i in range(artist_count):
        for j in range(tag_count):
            c[i][j] = postScore[i][j] / tagSpecificity[j] #eq 11
    '''
    if st > 0:
        rr0 = rr1 = rr2 = rr3 = 0
        st = 0
    rat0 = rat1 = rat2 = rat3 = ratt = 0
    ddd = ddd1 = 0
    for i in range(user_count):
        for j in range(artist_count):
            if freq[i][j] > 0:
                ddd = ddd + 1
            ddd1 = ddd1 + 1
    sparsity = 1 - (ddd/ddd1)
    print(sparsity)
    print(len(userMatrix))
    for i in range(len(userMatrix)):
        B0 = B1 = B2 = B3 = BB = 0
        for j in userMatrix[i]:
            if freq[i][j] > 0:
                if freq[i][j] > 0.75 and freq[i][j] <= 1:
                    B3 = B3 + 1
                elif freq[i][j] > 0.5 and freq[i][j] <= 0.75:
                    B2 = B2 + 1
                elif freq[i][j] > 0.25 and freq[i][j] <= 0.5:
                    B1 = B1 + 1
                else:
                    B0 = B0 + 1
            else:
                BB = BB + 1
        #print(B0+B1+B2+B3)
        rat3 = rat3 + B3
        rat2 = rat2 + B2
        rat1 = rat1 + B1
        rat0 = rat0 + B0
        #ratt = ratt + BB
    print(rat3/len(userMatrix))
    print(rat2/len(userMatrix))
    print(rat1/len(userMatrix))
    print(rat0/len(userMatrix))
    rr3 = rr3 + rat3/len(userMatrix)
    rr2 = rr2 + rat1/len(userMatrix)
    rr1 = rr1 + rat2/len(userMatrix)
    rr0 = rr0 + rat0/len(userMatrix)
    '''
    for i in range(artist_count):
        r = random.randint(0, 1000)
        if r < 100:
            for j in range(user_count):
                freq[j][i] = 0
    alpha = 0.001
    #method being called to get P and Q
    nP, nQ = matrix_factorization_implicit_semantic_tag_time(G, freq, S, T, P, Q, C, K, Cat, Tag, c, userMatrix, CNT, alpha)
    nQ = nQ.T
    nR = numpy.dot(nP, nQ.T)
    find_max = 0
    for i in range(len(nR)):
        for j in range(len(nR[i])):
            if(nR[i][j] < 0):
                nR[i][j] = 0
            if(nR[i][j] > 1):
                nR[i][j] = 1
            
    error = 0
    #testing error getting calculated
    for i in range(len(userMatrixTest)):
        for j in userMatrixTest[i]:
            cal_rating = nR[i][j]
            if (cal_rating < 0):
                cal_rating = cal_rating * (-1)

            error = error + pow((tr[i][j]) - cal_rating, 2)
    error = error / CNT_TEST
    error = pow(error, 1/2)
    print ("Train size = " + str(CNT))
    print ("Test size = " + str(CNT_TEST))
    print ('Error on testing data = '+ str(error))
print("Done...")
'''print(rr3/4)
print(rr2/4)
print(rr1/4)
print(rr0/4)
'''
