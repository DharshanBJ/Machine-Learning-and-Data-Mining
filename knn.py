
# coding: utf-8

# Problem 1: Python & Data Exploration

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
iris = np.genfromtxt("/Users/dharshanbj/Desktop/HW1-code/data/iris.txt",delimiter=None) # load the text file 
Y = iris[:,-1] # target value is the last column
X = iris[:,0:-1] # features are the other columns


# (a)To get the number of features, and to get the number of data points.

# In[3]:


print(X.shape[1]) #number of features
print(X.shape[0]) #number of data points


# (b)For each feature, plot a histogram of the data values

# In[62]:


plt.hist(iris[:,0],facecolor='green',label='Feature 1') #Feature 1
plt.hist(iris[:,1],facecolor='yellow',label='Feature 2') #Feature 2
plt.hist(iris[:,2],facecolor='red',label='Feature 3') #Feature 3
plt.hist(iris[:,3],facecolor='blue',label='Feature 4') #Feature 4

plt.xlabel('Feature value')
plt.ylabel('Count')
plt.title('Histogram of different features')
plt.legend()


# (c) Compute the mean & standard deviation of the data points for each feature

# In[18]:


print('mean of feature 1 data points ',np.mean(iris[:,0])) #mean of feature 1 data points
print('standard deviation of feature 1 data points',np.std(iris[:,0])) #standard deviation of feature 1 data points
print('mean of feature 2 data points',np.mean(iris[:,1])) #mean of feature 2 data points
print('standard deviation of feature 2 data points',np.std(iris[:,1])) #standard deviation of feature 2 data points
print('mean of feature 3 data points',np.mean(iris[:,2])) #mean of feature 3 data points
print('standard deviation of feature 3 data points',np.std(iris[:,2])) #standard deviation of feature 3 data points
print('mean of feature 4 data points',np.mean(iris[:,3])) #mean of feature 4 data points
print('standard deviation of feature 4 data points',np.std(iris[:,3])) #standard deviation of feature 4 data points


# d) For each pair of features (1,2), (1,3), and (1,4), plot a scatterplot of the feature values, colored according to their target value (class).

# In[44]:


col=[]
#print(Y)
for i in range(0,Y.size):
    if Y[i]==0:
        col.append('r')  #red marks Y=0
    elif Y[i]==1:
        col.append('b') #blue marks Y=1
    else:
        col.append('g') #green marks Y=2

        x1=iris[:,1]
        x2=iris[:,2]
                
#print(col)

#scatter plot
plt.title('Scatter plot - Feature 1 & Feature 2')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

colors = {'r':"Y=0", 'b':"Y=1",'g':"Y=2"}

for i in range(Y.size):
    plt.scatter(x1[i], x2[i],c=col[i])
  #label=colors[col[i]]


   
                
#print(col)



# In[45]:


x1=iris[:,1]
x3=iris[:,3]
#scatter plot
plt.title('Scatter plot - Feature 1 & Feature 3 ')
plt.xlabel('Feature 1')
plt.ylabel('Feature 3')

colors = {'r':"Y=0", 'b':"Y=1",'g':"Y=2"}

for i in range(Y.size):
    plt.scatter(x1[i], x3[i],c=col[i])
  #label=colors[col[i]]


# In[46]:


x1=iris[:,1]
x4=iris[:,4]
#scatter plot
plt.title('Scatter plot - Feature 1 & Feature 4 ')
plt.xlabel('Feature 1')
plt.ylabel('Feature 4')

colors = {'r':"Y=0", 'b':"Y=1",'g':"Y=2"}

for i in range(Y.size):
    plt.scatter(x1[i], x4[i],c=col[i])
  #label=colors[col[i]]


# In[47]:


x2=iris[:,2]
x3=iris[:,3]
#scatter plot
plt.title('Scatter plot - Feature 2 & Feature 3 ')
plt.xlabel('Feature 2')
plt.ylabel('Feature 3')

colors = {'r':"Y=0", 'b':"Y=1",'g':"Y=2"}

for i in range(Y.size):
    plt.scatter(x2[i], x3[i],c=col[i])
  #label=colors[col[i]]


# In[48]:


x2=iris[:,2]
x4=iris[:,4]
#scatter plot
plt.title('Scatter plot - Feature 2 & Feature 4 ')
plt.xlabel('Feature 2')
plt.ylabel('Feature 4')

colors = {'r':"Y=0", 'b':"Y=1",'g':"Y=2"}

for i in range(Y.size):
    plt.scatter(x2[i], x4[i],c=col[i])
  #label=colors[col[i]]


# In[49]:


x3=iris[:,3]
x4=iris[:,4]
#scatter plot
plt.title('Scatter plot - Feature 3 & Feature 4 ')
plt.xlabel('Feature 3')
plt.ylabel('Feature 4')

colors = {'r':"Y=0", 'b':"Y=1",'g':"Y=2"}

for i in range(Y.size):
    plt.scatter(x3[i], x4[i],c=col[i])
  #label=colors[col[i]]


# Problem 2: kNN predictions (30 pts)

# (a) Modify the code listed above to use only the first two features of X (e.g., let X be only the first two columns of iris, instead of the first four), and visualize (plot) the classification boundary for varying values of K =[1,5,10,50] using plotClassify2D.

# In[53]:


Y = iris[:,-1]
X = iris[:,0:2]

import mltools as ml

X,Y = ml.shuffleData(X,Y);
Xtr,Xva,Ytr,Yva = ml.splitData(X,Y, 0.75);

knn = ml.knn.knnClassify()
knn.train(Xtr, Ytr, 1) # where K is an integer, e.g. 1 for nearest neighbor predict
YvaHat = knn.predict(Xva) # get estimates of y for each data point in Xva

ml.plotClassify2D(knn, Xtr, Ytr);


# In[54]:


knn = ml.knn.knnClassify()
knn.train(Xtr, Ytr, 5) # where K is an integer, e.g. 1 for nearest neighbor predict
YvaHat = knn.predict(Xva) # get estimates of y for each data point in Xva

ml.plotClassify2D(knn, Xtr, Ytr);


# In[58]:


knn = ml.knn.knnClassify()
knn.train(Xtr, Ytr, 10) # where K is an integer, e.g. 1 for nearest neighbor predict
YvaHat = knn.predict(Xva) # get estimates of y for each data point in Xva

ml.plotClassify2D(knn, Xtr, Ytr);


# In[59]:


knn = ml.knn.knnClassify()
knn.train(Xtr, Ytr, 50) # where K is an integer, e.g. 1 for nearest neighbor predict
YvaHat = knn.predict(Xva) # get estimates of y for each data point in Xva

ml.plotClassify2D(knn, Xtr, Ytr);


# (b) Again using only the first two features, compute the error rate (number of misclassifications) on both the training and validation data as a function of K = [1,2,5,10,50,100,200]. You can do this most easily with a for-loop:

# In[72]:


Y = iris[:,-1]
X = iris[:,0:2]

X,Y = ml.shuffleData(X,Y);

Xtr,Xva,Ytr,Yva = ml.splitData(X,Y, 0.75);

errTrain = []
errValidation=[]
countTrain=0
countVal=0

K=[1,2,5,10,50,100,200];
for i,k in enumerate(K):
    learner = ml.knn.knnClassify() # TODO: complete code to train model
    learner.train(Xtr, Ytr, k)
    Yhat = learner.predict(Xtr) # TODO: complete code to predict results on training data 
   
    for j in range(len(Yhat)-1):
        if Yhat[j] !=Ytr[j]:
            countTrain = countTrain+1 # TODO: " " to count what fraction of predictions are wrong  
    
    errTrain.append(countTrain/75)
    countTrain=0
    
    Yhatvl = learner.predict(Xva) # TODO: complete code to predict results on training data 
    
    for j in range(len(Yva)-1):
        if Yhatvl[j] !=Yva[j]:
            countVal = countVal+1 # TODO: " " to count what fraction of predictions are wrong  
        
    errValidation.append(countVal/25)
    countVal=0
     #k=5 optimal value   

print(errTrain)
print(errValidation)
#TODO: repeat prediction / error evaluation for validation data 
    
plt.semilogx(K,errTrain,c='red',label='Training error')
plt.semilogx(K,errValidation,c='green',label='Validation error')
plt.title("K vs Error rate")
plt.xlabel("K")
plt.ylabel("Error")

plt.legend()


# For the K value 10,both the training error and the testing error is minimum.Hence, K=10 is the recommended value.
