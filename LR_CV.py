
# coding: utf-8

# Dharshan Birur Jayaprabhu

# Student ID - 54773179 

# Problem 1: Linear Regression

# (a)

# In[15]:


import numpy as np
import matplotlib.pyplot as plt
import mltools as ml

data = np.genfromtxt("/Users/dharshanbj/Desktop/HW2-code/data/curve80.txt",delimiter=None) # load the text file

X = data[:,0]
X = X[:,np.newaxis] # code expects shape (M,N) so make sure it's 2-dimensiona 
Y = data[:,1] # doesn't matter for Y
Xtr,Xte,Ytr,Yte = ml.splitData(X,Y,0.75)


# (b)

# In[76]:


from sklearn.metrics import mean_squared_error
lr = ml.linear.linearRegress( Xtr, Ytr ); # create and train model
xs = np.linspace(0,10,200);
xs = xs[:,np.newaxis]
ys = lr.predict( xs );

#plot the resulting function by simply evaluating the model at a large number of x values, xs
plt.plot(xs,ys,c='green',label='xs')

#Plot the training data along with your prediction function in a single plot
plt.scatter(Xtr,Ytr,c='red',label='Xtr')

plt.legend()
#the linear regression coefficients
print(lr.theta)

#mean square error for training
print(mean_squared_error(Ytr,yhattr))

#mean square error for testing
yhatte=lr.predict(Xte)
print(mean_squared_error(Yte,yhatte))


# (c)

# In[89]:


#degree 1
XtrP = ml.transforms.fpoly(Xtr, 1, bias=False);
XtrP,params = ml.transforms.rescale(XtrP);
lr = ml.linear.linearRegress( XtrP, Ytr ); # create and train model

# Now, apply the same polynomial expansion & scaling transformation to Xtest:
#XteP = ml.transforms.fpoly(Xte, 3, bias=False);
#XteP,params = ml.transforms.rescale(XteP);
#lr = ml.linear.linearRegress( XteP, Yte );

Phi = lambda X: ml.transforms.rescale( ml.transforms.fpoly(X,1,False), params)[0]
YhatTrain = lr.predict(Phi(Xtr));  
YhatTest = lr.predict(Phi(Xte));
#(1) plot their learned prediction function f (x)
xs = np.linspace(0,10,200);
xs = xs[:,np.newaxis]
ys=lr.predict(Phi(xs));
plt.plot(Phi(xs),ys)

print(mean_squared_error(Ytr,YhatTrain))
print(mean_squared_error(Yte,YhatTest))

#(2) their training and test errors (plot the error values on a log scale, e.g., semilogy)
#YhatTrain = lr.predict(Phi(Xtr));  
#YhatTest = lr.predict(Phi(Xte));
#plt.semilogy(Xtr,Ytr-YhatTrain,c='yellow')

#plt.semilogy(Xte,Yte-YhatTest,c='blue')


# In[91]:


#degree 3
XtrP = ml.transforms.fpoly(Xtr, 3, bias=False);
XtrP,params = ml.transforms.rescale(XtrP);
lr = ml.linear.linearRegress( XtrP, Ytr ); # create and train model

# Now, apply the same polynomial expansion & scaling transformation to Xtest:
#XteP = ml.transforms.fpoly(Xte, 3, bias=False);
#XteP,params = ml.transforms.rescale(XteP);
#lr = ml.linear.linearRegress( XteP, Yte );

Phi = lambda X: ml.transforms.rescale( ml.transforms.fpoly(X,3,False), params)[0]
YhatTrain = lr.predict(Phi(Xtr));  
YhatTest = lr.predict(Phi(Xte));
#(1) plot their learned prediction function f (x)
xs = np.linspace(0,10,200);
xs = xs[:,np.newaxis]
ys=lr.predict(Phi(xs));
plt.plot((xs),ys)

print(mean_squared_error(Ytr,YhatTrain))
print(mean_squared_error(Yte,YhatTest))

#(2) their training and test errors (plot the error values on a log scale, e.g., semilogy)
#YhatTrain = lr.predict(Phi(Xtr));  
#YhatTest = lr.predict(Phi(Xte));
#plt.semilogy(Xtr,Ytr-YhatTrain,c='yellow')

#plt.semilogy(Xte,Yte-YhatTest,c='blue')


# In[96]:


#degree 5
XtrP = ml.transforms.fpoly(Xtr, 5, bias=False);
XtrP,params = ml.transforms.rescale(XtrP);
lr = ml.linear.linearRegress( XtrP, Ytr ); # create and train model

# Now, apply the same polynomial expansion & scaling transformation to Xtest:
#XteP = ml.transforms.fpoly(Xte, 3, bias=False);
#XteP,params = ml.transforms.rescale(XteP);
#lr = ml.linear.linearRegress( XteP, Yte );

Phi = lambda X: ml.transforms.rescale( ml.transforms.fpoly(X,5,False), params)[0]
YhatTrain = lr.predict(Phi(Xtr));  
YhatTest = lr.predict(Phi(Xte));
#(1) plot their learned prediction function f (x)
xs = np.linspace(0,10,200);
xs = xs[:,np.newaxis]
ys=lr.predict(Phi(xs));
plt.plot((xs),ys)

print(mean_squared_error(Ytr,YhatTrain))
print(mean_squared_error(Yte,YhatTest))
#(2) their training and test errors (plot the error values on a log scale, e.g., semilogy)
#YhatTrain = lr.predict(Phi(Xtr));  
#YhatTest = lr.predict(Phi(Xte));
#plt.semilogy(Xtr,Ytr-YhatTrain,c='yellow')

#plt.semilogy(Xte,Yte-YhatTest,c='blue')


# In[95]:


#degree 7
XtrP = ml.transforms.fpoly(Xtr, 7, bias=False);
XtrP,params = ml.transforms.rescale(XtrP);
lr = ml.linear.linearRegress( XtrP, Ytr ); # create and train model

# Now, apply the same polynomial expansion & scaling transformation to Xtest:
#XteP = ml.transforms.fpoly(Xte, 3, bias=False);
#XteP,params = ml.transforms.rescale(XteP);
#lr = ml.linear.linearRegress( XteP, Yte );

Phi = lambda X: ml.transforms.rescale( ml.transforms.fpoly(X,7,False), params)[0]
YhatTrain = lr.predict(Phi(Xtr));  
YhatTest = lr.predict(Phi(Xte));
#(1) plot their learned prediction function f (x)
xs = np.linspace(0,10,200);
xs = xs[:,np.newaxis]
ys=lr.predict(Phi(xs));
plt.plot((xs),ys)

print(mean_squared_error(Ytr,YhatTrain))
print(mean_squared_error(Yte,YhatTest))

#(2) their training and test errors (plot the error values on a log scale, e.g., semilogy)
#YhatTrain = lr.predict(Phi(Xtr));  
#YhatTest = lr.predict(Phi(Xte));
#plt.semilogy(Xtr,Ytr-YhatTrain,c='yellow')

#plt.semilogy(Xte,Yte-YhatTest,c='blue')


# In[94]:


#degree 10
XtrP = ml.transforms.fpoly(Xtr, 10, bias=False);
XtrP,params = ml.transforms.rescale(XtrP);
lr = ml.linear.linearRegress( XtrP, Ytr ); # create and train model

# Now, apply the same polynomial expansion & scaling transformation to Xtest:
#XteP = ml.transforms.fpoly(Xte, 3, bias=False);
#XteP,params = ml.transforms.rescale(XteP);
#lr = ml.linear.linearRegress( XteP, Yte );

Phi = lambda X: ml.transforms.rescale( ml.transforms.fpoly(X,10,False), params)[0]
YhatTrain = lr.predict(Phi(Xtr));  
YhatTest = lr.predict(Phi(Xte));
#(1) plot their learned prediction function f (x)
xs = np.linspace(0,10,200);
xs = xs[:,np.newaxis]
ys=lr.predict(Phi(xs));
plt.plot((xs),ys)

print(mean_squared_error(Ytr,YhatTrain))
print(mean_squared_error(Yte,YhatTest))
#(2) their training and test errors (plot the error values on a log scale, e.g., semilogy)
#YhatTrain = lr.predict(Phi(Xtr));  
#YhatTest = lr.predict(Phi(Xte));
#plt.semilogy(Xtr,Ytr-YhatTrain,c='yellow')

#plt.semilogy(Xte,Yte-YhatTest,c='blue')


# In[92]:


#degree 18
XtrP = ml.transforms.fpoly(Xtr, 18, bias=False);
XtrP,params = ml.transforms.rescale(XtrP);
lr = ml.linear.linearRegress( XtrP, Ytr ); # create and train model

# Now, apply the same polynomial expansion & scaling transformation to Xtest:
#XteP = ml.transforms.fpoly(Xte, 3, bias=False);
#XteP,params = ml.transforms.rescale(XteP);
#lr = ml.linear.linearRegress( XteP, Yte );

Phi = lambda X: ml.transforms.rescale( ml.transforms.fpoly(X,18,False), params)[0]
YhatTrain = lr.predict(Phi(Xtr));  
YhatTest = lr.predict(Phi(Xte));
#(1) plot their learned prediction function f (x)
xs = np.linspace(0,10,200);
xs = xs[:,np.newaxis]
ys=lr.predict(Phi(xs));
plt.plot((xs),ys)

print(mean_squared_error(Ytr,YhatTrain))
print(mean_squared_error(Yte,YhatTest))
#(2) their training and test errors (plot the error values on a log scale, e.g., semilogy)
#YhatTrain = lr.predict(Phi(Xtr));  
#YhatTest = lr.predict(Phi(Xte));
#plt.semilogy(Xtr,Ytr-YhatTrain,c='yellow')

#plt.semilogy(Xte,Yte-YhatTest,c='blue')


# In[101]:


MSE_Train=[1.1277119556093909,0.6339652063119635,0.4042489464459089,0.31563467398924466,0.29894797966941294,0.28051682005715184]
MSE_Test=[2.242349203010125,0.8616114815449996,1.034419020563194,0.6502246079666845,0.6090600748624672,482.28125812120953]

degree=[1,3,5,7,10,18]
plt.semilogy(degree,MSE_Train,c='blue',label='Training error')
plt.semilogy(degree,MSE_Test,c='orange',label='Testing error')
plt.legend();


# Problem 2: Cross-validation

# In[72]:


#degree 1
J=[]
nFolds = 5;
Phi = lambda X: ml.transforms.rescale( ml.transforms.fpoly(X,1,False), params)[0]
for iFold in range(nFolds):
    Xti,Xvi,Yti,Yvi = ml.crossValidate(Xtr,Ytr,nFolds,iFold);
    XtrP = ml.transforms.fpoly(Xti, 1, bias=False);
    XtrP,params = ml.transforms.rescale(XtrP);
    learner = ml.linear.linearRegress(XtrP, Yti); # TODO: train on Xti, Yti
    Yhatvi=learner.predict(Phi(Xvi))
    J.append(mean_squared_error(Yvi,Yhatvi))

print(np.mean(J))


# In[67]:


#degree 3
J=[]
nFolds = 5;
Phi = lambda X: ml.transforms.rescale( ml.transforms.fpoly(X,3,False), params)[0]
for iFold in range(nFolds):
    Xti,Xvi,Yti,Yvi = ml.crossValidate(Xtr,Ytr,nFolds,iFold);
    XtrP = ml.transforms.fpoly(Xti, 3, bias=False);
    XtrP,params = ml.transforms.rescale(XtrP);
    learner = ml.linear.linearRegress(XtrP, Yti); # TODO: train on Xti, Yti
    Yhatvi=learner.predict(Phi(Xvi))
    J.append(mean_squared_error(Yvi,Yhatvi))

print(np.mean(J))


# In[68]:


#degree 5
J=[]
nFolds = 5;
Phi = lambda X: ml.transforms.rescale( ml.transforms.fpoly(X,5,False), params)[0]
for iFold in range(nFolds):
    Xti,Xvi,Yti,Yvi = ml.crossValidate(Xtr,Ytr,nFolds,iFold);
    XtrP = ml.transforms.fpoly(Xti, 5, bias=False);
    XtrP,params = ml.transforms.rescale(XtrP);
    learner = ml.linear.linearRegress(XtrP, Yti); # TODO: train on Xti, Yti
    Yhatvi=learner.predict(Phi(Xvi))
    J.append(mean_squared_error(Yvi,Yhatvi))

print(np.mean(J))


# In[69]:


#degree 7
J=[]
nFolds = 5;
Phi = lambda X: ml.transforms.rescale( ml.transforms.fpoly(X,7,False), params)[0]
for iFold in range(nFolds):
    Xti,Xvi,Yti,Yvi = ml.crossValidate(Xtr,Ytr,nFolds,iFold);
    XtrP = ml.transforms.fpoly(Xti, 7, bias=False);
    XtrP,params = ml.transforms.rescale(XtrP);
    learner = ml.linear.linearRegress(XtrP, Yti); # TODO: train on Xti, Yti
    Yhatvi=learner.predict(Phi(Xvi))
    J.append(mean_squared_error(Yvi,Yhatvi))

print(np.mean(J))


# In[70]:


#degree 10
J=[]
nFolds = 5;
Phi = lambda X: ml.transforms.rescale( ml.transforms.fpoly(X,10,False), params)[0]
for iFold in range(nFolds):
    Xti,Xvi,Yti,Yvi = ml.crossValidate(Xtr,Ytr,nFolds,iFold);
    XtrP = ml.transforms.fpoly(Xti, 10, bias=False);
    XtrP,params = ml.transforms.rescale(XtrP);
    learner = ml.linear.linearRegress(XtrP, Yti); # TODO: train on Xti, Yti
    Yhatvi=learner.predict(Phi(Xvi))
    J.append(mean_squared_error(Yvi,Yhatvi))

print(np.mean(J))


# In[71]:


#degree 18
J=[]
nFolds = 5;
Phi = lambda X: ml.transforms.rescale( ml.transforms.fpoly(X,18,False), params)[0]
for iFold in range(nFolds):
    Xti,Xvi,Yti,Yvi = ml.crossValidate(Xtr,Ytr,nFolds,iFold);
    XtrP = ml.transforms.fpoly(Xti, 18, bias=False);
    XtrP,params = ml.transforms.rescale(XtrP);
    learner = ml.linear.linearRegress(XtrP, Yti); # TODO: train on Xti, Yti
    Yhatvi=learner.predict(Phi(Xvi))
    J.append(mean_squared_error(Yvi,Yhatvi))

print(np.mean(J))


# (b)

# In[75]:


degree=[1,3,5,7,10,18]
cvError=[1.2118626629641984,0.742900575205166,0.5910703726406653,0.7335637831346695,0.767705685939558,216818.184600881]

plt.semilogy(degree,cvError)
plt.xlabel('Degree')
plt.ylabel('Cross-validation error')


# (c) Degree 5 has the minimum cross validation error

# (d)

# In[74]:


#Degree 5
XtrP = ml.transforms.fpoly(Xtr, 5, bias=False);
XtrP,params = ml.transforms.rescale(XtrP);
lr = ml.linear.linearRegress( XtrP, Ytr ); # create and train model

Phi = lambda X: ml.transforms.rescale( ml.transforms.fpoly(X,5,False), params)[0]
YhatTest = lr.predict(Phi(Xte));

print(mean_squared_error(Yte,YhatTest))


# MSE evaluated on actual test data is 1.034419020563194
# MSE evaluated on cross-validation is 0.5910703726406653
# Hence MSE for cross-validation is less than MSE for actual test data.
