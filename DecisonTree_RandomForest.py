
# coding: utf-8

# DHARSHAN BIRUR JAYAPRABHU

# 54773179

# 2(A)

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
from sklearn.metrics import mean_squared_error

dataXtr = np.genfromtxt("/Users/dharshanbj/Desktop/X_train.txt",delimiter=None)
dataYtr = np.genfromtxt("/Users/dharshanbj/Desktop/Y_train.txt",delimiter=None)
dataXte = np.genfromtxt("/Users/dharshanbj/Desktop/X_test.txt",delimiter=None)
X=dataXtr[:20000]
Y=dataYtr[:20000]
Xtest=dataXte[:,:]


Xtr,Xva,Ytr,Yva = ml.splitData(X,Y,0.5) #first 10000 training ,next 10000 is testing data set

Xtr=np.array(Xtr) # learner takes array as an input
Ytr=np.array(Ytr)
Xtest=np.array(Xtest)


# 2(B)

# In[5]:


#train the learner
learner = ml.dtree.treeClassify(Xtr,Ytr, maxDepth=50)

#predict values of y for training data
YhatTr=learner.predict(Xtr)

#error rate for training data
rate=0;
for i,j in zip(YhatTr,Ytr):
    if(i!=j):
        rate=rate+1;
        
print(mean_squared_error(Ytr,YhatTr))
print(rate/len(Ytr))
#predict values of y for training data
#YhatVa=learner.predict(Xva)

#error rate for validation data
#print(mean_squared_error(Yva,YhatVa))


# 2 (C)

# In[10]:


MSE_tr=[]
MSE_va=[]
for i in range(0,16):
    #for training data
    learner = ml.dtree.treeClassify(Xtr,Ytr, maxDepth=i) 
    YhatTr=learner.predict(Xtr)
    MSE_tr.append(mean_squared_error(Ytr,YhatTr))
    #for testing data 
    YhatVa=learner.predict(Xva)
    MSE_va.append(mean_squared_error(Yva,YhatVa))   
    


# In[11]:


depth=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
plt.plot(depth,MSE_tr,c='r',label='Training error')
plt.plot(depth,MSE_va,c='g',label='Validation error')
plt.xlabel('Depth')
plt.ylabel('MSE')
plt.legend()


# The complexity is increasing as the depth cutoff increases.
# Yes,the model starts overfitting after a depth of 4.
# Depth 4 has to selected for best complexity control.

# 2 (D)

# In[ ]:


#def train(self, X, Y, minParent=2, maxDepth=np.inf, minLeaf=1, nFeatures=None):


# In[12]:


MSE_tr=[]
MSE_va=[]
powers=[4,8,16,32,64,128,256,512,1024,2048,4096]
for i in powers:
    #for training data
    learner = ml.dtree.treeClassify(Xtr,Ytr, maxDepth=50,minLeaf=i) 
    YhatTr=learner.predict(Xtr)
    MSE_tr.append(mean_squared_error(Ytr,YhatTr))
    #for testing data 
    YhatVa=learner.predict(Xva)
    MSE_va.append(mean_squared_error(Yva,YhatVa))   


# In[13]:


plt.plot(powers,MSE_tr,c='r',label='Training error')
plt.plot(powers,MSE_va,c='g',label='Validation error')
plt.xlabel('Min Leaf')
plt.ylabel('MSE')
plt.legend()


# Over-fitting occurs when the min Leaf is less.
# The model complexity reduces as the min-Leaf increases.
# 

# 2 (E)

# In[14]:


MSE_tr=[]
MSE_va=[]
depth=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
for i in depth:
    #for training data
    learner = ml.dtree.treeClassify(Xtr,Ytr, maxDepth=50,minParent=i) 
    YhatTr=learner.predict(Xtr)
    MSE_tr.append(mean_squared_error(Ytr,YhatTr))
    #for testing data 
    YhatVa=learner.predict(Xva)
    MSE_va.append(mean_squared_error(Yva,YhatVa))   


# In[8]:


plt.plot(depth,MSE_tr,c='r',label='Training error')
plt.plot(depth,MSE_va,c='g',label='Validation error')
plt.xlabel('Min Parent')
plt.ylabel('MSE')
plt.legend()


# 2 (F)

# In[10]:


from sklearn.metrics import roc_curve, auc

XtrP = ml.transforms.fpoly(Xtr, 1, bias=False);
XtrP,params = ml.transforms.rescale(XtrP);
Phi = lambda X: ml.transforms.rescale( ml.transforms.fpoly(X,1,False), params)[0]

# Compute ROC curve and ROC area for the best parameterized model
learner = ml.dtree.treeClassify(XtrP,Ytr, maxDepth=3,minLeaf=512,minParent=2) 
#computing and ploting for validation data
Yhatval=learner.predict(Phi(Xva))
    
fpr,tpr,_ = roc_curve(Yva, Yhatval)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange',label='ROC curve (area = %f)'%roc_auc)
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')


# 2 (G)

# In[13]:


#train the learner
#X=dataXtr[:,:]
#Y=dataYtr[:,:]

learner = ml.dtree.treeClassify(dataXtr,dataYtr, maxDepth=5,minLeaf=512,minParent=3)

Ypred = learner.predictSoft(Xtest)
# Now output a file with two columns, a row ID and a confidence in class 1:
np.savetxt('/Users/dharshanbj/Desktop/Yhat_dtreenew2.txt',
np.vstack( (np.arange(len(Ypred)) , Ypred[:,1]) ).T,'%d, %.2f',header='ID,Prob1',comments='',delimiter=',');


# RANDOM FORESTS

# 3 (A)

# In[58]:


#Random Forest of size 25
# Load data set X, Y for training the ensemble…
m,n = Xtr.shape
ensemble = [ None ] * 25 # Allocate space for learners
n=7
for i in range(25):
    #ind = np.floor( m * np.random.rand(n) ).astype(int)
    #Xb, Yb = Xtr[ind,:],Ytr[ind]
    Xb,Yb=ml.bootstrapData(Xtr,Ytr)
    ensemble[i]=ml.dtree.treeClassify(Xb,Yb, maxDepth=5,minLeaf=256,nFeatures=n) 


# In[15]:


# test on data Xva
mTest = Xva.shape[0]
predictTe = np.zeros( (mTest, 25) ) # Allocate space for predictions from each model
predictTr = np.zeros( (mTest, 25) )
for i in range(25):
    predictTe[:,i] = ensemble[i].predict(Xva) # Apply each classifier
    predictTr[:,i] = ensemble[i].predict(Xtr)
    
predictTest = np.mean(predictTe, axis=1)
predictTrain = np.mean(predictTr, axis=1)
print(mean_squared_error(Ytr,predictTrain)) 
print(mean_squared_error(Yva,predictTest)) 
#print(predictTrain) 


# In[16]:


#Random Forest of size 10

# Load data set X, Y for training the ensemble…
m,n = Xtr.shape
ensemble = [ None ] * 10 # Allocate space for learners
n=7
for i in range(10):
    #ind = np.floor( m * np.random.rand(n) ).astype(int)
    #Xb, Yb = Xtr[ind,:],Ytr[ind]
    Xb,Yb=ml.bootstrapData(Xtr,Ytr)
    ensemble[i]=ml.dtree.treeClassify(Xb,Yb, maxDepth=5,minLeaf=256,nFeatures=n) 

# test on data Xva
mTest = Xva.shape[0]
predictTe = np.zeros( (mTest, 10) ) # Allocate space for predictions from each model
predictTr = np.zeros( (mTest, 10) )

for i in range(10):
    predictTe[:,i] = ensemble[i].predict(Xva) # Apply each classifier
    predictTr[:,i] = ensemble[i].predict(Xtr)
    
predictTest = np.mean(predictTe, axis=1)
predictTrain = np.mean(predictTr, axis=1)
#print(predictTest)   
#print(predictTrain) 
print(mean_squared_error(Ytr,predictTrain)) 
print(mean_squared_error(Yva,predictTest)) 


# In[17]:


#Random Forest of size 5

# Load data set X, Y for training the ensemble…
m,n = Xtr.shape
ensemble = [ None ] * 5 # Allocate space for learners
n=7
for i in range(5):
    #ind = np.floor( m * np.random.rand(n) ).astype(int)
    #Xb, Yb = Xtr[ind,:],Ytr[ind]
    Xb,Yb=ml.bootstrapData(Xtr,Ytr)
    ensemble[i]=ml.dtree.treeClassify(Xb,Yb, maxDepth=5,minLeaf=256,nFeatures=n) 

# test on data Xva
mTest = Xva.shape[0]
predictTe = np.zeros( (mTest, 5) ) # Allocate space for predictions from each model
predictTr = np.zeros( (mTest, 5) )

for i in range(5):
    predictTe[:,i] = ensemble[i].predict(Xva) # Apply each classifier
    predictTr[:,i] = ensemble[i].predict(Xtr)
    
predictTest = np.mean(predictTe, axis=1)
predictTrain = np.mean(predictTr, axis=1)
#print(predictTest)   
#print(predictTrain) 
print(mean_squared_error(Ytr,predictTrain)) 
print(mean_squared_error(Yva,predictTest)) 


# In[18]:


#Random Forest of size 1

# Load data set X, Y for training the ensemble…
m,n = Xtr.shape
ensemble = [ None ] * 1 # Allocate space for learners
n=7
for i in range(1):
    #ind = np.floor( m * np.random.rand(n) ).astype(int)
    #Xb, Yb = Xtr[ind,:],Ytr[ind]
    Xb,Yb=ml.bootstrapData(Xtr,Ytr)
    ensemble[i]=ml.dtree.treeClassify(Xb,Yb, maxDepth=5,minLeaf=256,nFeatures=n) 

# test on data Xva
mTest = Xva.shape[0]
predictTe = np.zeros( (mTest, 1) ) # Allocate space for predictions from each model
predictTr = np.zeros( (mTest, 1) )

for i in range(1):
    predictTe[:,i] = ensemble[i].predict(Xva) # Apply each classifier
    predictTr[:,i] = ensemble[i].predict(Xtr)
    
predictTest = np.mean(predictTe, axis=1)
predictTrain = np.mean(predictTr, axis=1)
#print(predictTest)   
#print(predictTrain) 
print(mean_squared_error(Ytr,predictTrain)) 
print(mean_squared_error(Yva,predictTest)) 


# In[19]:


#plot

learners=[25,10,5,1]
MSE_Training=[0.26140608000000004,0.26353499999999996,0.26652000000000003,0.3134]
MSE_Validation=[0.26434336,0.26852200000000004,0.270612,0.3192]
plt.plot(learners,MSE_Training,c='r',label='Training error')
plt.plot(learners,MSE_Validation,c='g',label='Validation error')
plt.xlabel('learners')
plt.ylabel('MSE')
plt.legend()


# 3 (B)

# In[20]:


#Random Forest of size 25
# Load data set X, Y for training the ensemble…
learners=[50]
for k in learners:
    m,n = Xtr.shape
    ensemble = [ None ] * k # Allocate space for learners
    n=12
    for i in range(k):
        Xb,Yb=ml.bootstrapData(dataXtr,dataYtr)
        ensemble[i]=ml.dtree.treeClassify(Xb,Yb, maxDepth=50,minLeaf=4,nFeatures=n) 
        
        # test on data Xtest
    mTest = Xtest.shape[0]
    mVal = Xva.shape[0]
    predictTest = np.zeros( (mTest, k) ) # Allocate space for predictions from each model
    predictVal=np.zeros((mVal,k))
    
    for i in range(k):
        predictTest[:,i] = ensemble[i].predict(Xtest) # Apply each classifier
        predictVal[:,i]=ensemble[i].predict(Xva)
    
    predictTest = np.mean(predictTest, axis=1)
    predictVal=np.mean(predictVal,axis=1)
    
    print(mean_squared_error(Yva,predictVal))
    fpr,tpr,_ = roc_curve(Yva, predictVal)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, color='darkorange',label='ROC curve (area = %f)'%roc_auc)
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.show()

# Now output a file with two columns, a row ID and a confidence in class 1:
np.savetxt('/Users/dharshanbj/Desktop/Yhat_dtreeEnsembleFinal_70.txt',
np.vstack( (np.arange(len(predictTest)) , predictTest) ).T,'%d, %.2f',header='ID,Prob1',comments='',delimiter=',');


# After uploading the predicted Y values for test data for ensemble size 50 and max depth 50,i got an AUC of 0.75391.
# My kaggel username is Dharshan.
