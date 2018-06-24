
# coding: utf-8

# DHARSHAN BIRUR JAYAPRABHU

# 54773179

# 1 (a)

# In[14]:


import numpy as np
import matplotlib.pyplot as plt
import mltools as ml

iris = np.genfromtxt("/Users/dharshanbj/Desktop/HW3-code/data/iris.txt",delimiter=None)
X, Y = iris[:,0:2], iris[:,-1] #X-first 2 columns Y-last column
#X,Y = ml.shuffleData(X,Y) 

# Rescale the data matrix so that the features have similar ranges / variance
Xrs,params = ml.transforms.rescale(X);
#print(Xrs)

Yrs,params = ml.transforms.rescale(Y);
#print(Yrs)
        
XA, YA = Xrs[Y<2,:], Y[Y<2]#YA-values with y<2(0,1)
XB, YB = Xrs[Y>0,:], Y[Y>0]#YB-values with y>0(1,2)

col1=[]
#print(Y)
for i in range(0,YA.size):
    if YA[i]==0:
        col1.append('r')  #red marks Y=0
    elif YA[i]==1:
        col1.append('b') #blue marks Y=1
    else:
        col1.append('g') #green marks Y=2

col2=[]
for i in range(0,YB.size):
    if YB[i]==0:
        col2.append('r')  #red marks Y=0
    elif YB[i]==1:
        col2.append('b') #blue marks Y=1
    else:
        col2.append('g') #green marks Y=2
        
        

plt.scatter(XA[:,0],XA[:,1],c=col1)


# In[15]:


plt.scatter(XB[:,0],XB[:,1],c=col2)


# From visual inspection, it can be seen that scatter plot of dataset 1 is linearly seperable,while the scatter plot of dataset 2 is not linearly seperable.

# (B)

# In[16]:


import mltools as ml
from logisticClassify2 import *

learner = logisticClassify2(); # create "blank" learner
learner.classes = np.unique(YA) # define class labels using YA or YB
wts = np.array([0.5,1,-0.25]); # TODO: fill in values
learner.theta = wts; # set the learner's parameters
learner.plotBoundary(XA,YA);


# In[17]:


# For dataset B
learnerB = logisticClassify2();
learnerB.classes = np.unique(YB)
wts = np.array([0.5,1,-0.25]);
learnerB.theta = wts; # set the learner's parameters
learnerB.plotBoundary(XB,YB);
plt.show()


# (C)

# In[18]:


# For dataset A
learnerA = logisticClassify2();
learnerA.classes = np.unique(YA)
wts = np.array([0.5,1,-0.25]);
learnerA.theta = wts
learnerA.err(XA, YA)


# In[19]:


# For dataset B
learnerB = logisticClassify2();
learnerB.classes = np.unique(YB)
wts = np.array([0.5,1,-0.25]);
learnerB.theta = wts
learnerB.err(XB, YB)


# (D)

# In[20]:


ml.plotClassify2D(learnerA,XA,YA)
plt.show()


# In[21]:


ml.plotClassify2D(learnerB,XB,YB)
plt.show()


# (E)

# The derivation of the gradient is done my computing the ∂Jj(θ) over the surrogate loss for ∂θi
# point j given by x(j), y(j). The derivative is given Equation 1, where σ(z) = (1 + exp(z))−1 and z = θ.x(j)T
# 
# ∂Jj(θ)/∂θi = x(j)(σ(z)-y(j))+2.α.θi

# In[ ]:


(F)
'''
def train(self, X, Y, initStep=1., stopTol=1e-4, stopEpochs=5000, plot=None):
        """ Train the logistic regression using stochastic gradient descent """
        M,N = X.shape;                     # initialize the model if necessary:
        self.classes = np.unique(Y);       # Y may have two classes, any values
        XX = np.hstack((np.ones((M,1)),X))   # XX is X, but with an extra column of ones
        YY = ml.toIndex(Y,self.classes);   # YY is Y, but with canonical values 0 or 1
        if len(self.theta)!=N+1: self.theta=np.random.rand(N+1);
        # init loop variables:
        epoch=0; done=False; Jnll=[]; J01=[]; 
        while not done:
            stepsize, epoch = initStep*2.0/(2.0+epoch), epoch+1; # update stepsize
            # Do an SGD pass through the entire data set:
            for i in np.random.permutation(M):
                ri    = XX[i].dot(self.theta)     # TODO: compute linear response r(x)
                sigmoid    = 1./(1.+np.exp(-ri))
                gradient = -(1-sigmoid)*XX[i,:] if YY[i] else sigmoid*XX[i,:];     # TODO: compute gradient of NLL loss
                self.theta -= stepsize * gradient;  # take a gradient step

            J01.append( self.err(X,Y) )  # evaluate the current error rate 
            ## TODO: compute surrogate loss (logistic negative log-likelihood)
            ##  Jsur = sum_i [ (log sigmoid) if yi==1 else (log(1-sigmoid)) ]
            S = 1./(1.+np.exp(-(XX.dot(self.theta))))
            Jsur = -np.mean(YY*np.log(S)+(1-YY)*np.log(1-S))
            Jnll.append(Jsur) # TODO evaluate the current NLL loss
            ## For debugging: you may want to print current parameters & losses
            # print self.theta, ' => ', Jsur[-1], ' / ', J01[-1]  
            # raw_input()   # pause for keystroke
            # TODO check stopping criteria: exit if exceeded # of epochs ( > stopEpochs)
            # or if Jnll not changing between epochs ( < stopTol )
            done = epoch>=stopEpochs or (epoch>1 and abs(Jnll[-1]-Jnll[-2])< stopTol);
        plt.figure(1);plt.clf(); plt.plot(Jnll,'b-',J01,'r-'); plt.draw();    # plot losses
        if N==2: plt.figure(2);plt.clf(); self.plotBoundary(X,Y); plt.draw(); # & predictor if 2D
        plt.pause(.01);                    # let OS draw the plot
'''


# (G)

# In[22]:


learnerA = logisticClassify2();
wts=np.array([0.,0.,0.]);
learnerA.theta = wts
learnerA.train(XA, YA, initStep=1e-1,stopEpochs=1000,stopTol=1e-5);
ml.plotClassify2D(learnerA,XA,YA)
print("Training error rate: ",learnerA.err(XA,YA))
plt.show()


# In[23]:


learnerB = logisticClassify2()
wts=np.array([0.,0.,0.]);
learnerB.theta = wts
learnerB.train(XB,YB,initStep=1e-1,stopEpochs=1000,stopTol=1e-5);
ml.plotClassify2D(learnerB,XB,YB)
print("Training error rate: ",learnerB.err(XB,YB))
plt.show()


# In[ ]:


'''
def regularisationtrain(self,X,Y, initStep=1.,stopTol=1e-4,stopEpochs=5000,alpha=0.,plot=None):
        """ Train the logistic regression using stochastic gradient descent """
        M,N = X.shape;                     # initialize the model if necessary:
        self.classes = np.unique(Y);       # Y may have two classes, any values
        XX = np.hstack((np.ones((M,1)),X)) # XX is X, but with an extra column of ones
        YY = ml.toIndex(Y,self.classes);   # YY is Y, but with canonical values 0 or 1
        if len(self.theta)!=N+1: self.theta=np.random.rand(N+1);
        # init loop variables:
        epoch=0; done=False; Jnll=[]; J01=[];
        while not done:
            stepsize, epoch = initStep*2.0/(2.0+epoch), epoch+1; # update stepsize
            # Do an SGD pass through the entire data set:
            for i in np.random.permutation(M):
                ri = XX[i].dot(self.theta)     # compute linear response r(x)
                si    = 1./(1.+np.exp(-ri))
                gradi = -(1-si)*XX[i,:] if YY[i] else si*XX[i,:]  # compute gradient of NLL loss
                gradi += 2.*alpha*self.theta # gradient of the additional L2 regularization term
                self.theta -= stepsize * gradi;  # take a gradient step
            J01.append( self.err(X,Y) )  # evaluate the current error rate
            ## compute surrogate loss (logistic negative log-likelihood)
            ##  Jnll = sum_i [ (log si) if yi==1 else (log(1-si)) ]
            S = 1./(1.+np.exp(-(XX.dot(self.theta))))
            Jsur = -np.mean(YY*np.log(S)+(1-YY)*np.log(1-S))
            Jnll.append( Jsur ) # evaluate the current NLL loss
            done = epoch>=stopEpochs or (epoch>1 and abs(Jnll[-1]-Jnll[-2])<stopTol);
                             # or if Jnll not changing between epochs ( < stopTol )
        plt.figure(1); plt.clf(); plt.plot(Jnll,'b-',J01,'r-'); plt.draw();
        # plot losses
        if N==2: plt.figure(2); plt.clf(); self.plotBoundary(X,Y); plt.draw();
        # & predictor if 2D
        plt.pause(.01);
        '''


# (H)

# In[24]:


learnerA = logisticClassify2();
wts=np.array([0.,0.,0.]);
learnerA.theta = wts
learnerA.regularisationtrain(XA, YA, initStep=1e-1,stopEpochs=1000,stopTol=1e-5,alpha=1.);
ml.plotClassify2D(learnerA,XA,YA)
print("Training error rate: ",learnerA.err(XA,YA))
plt.show()


# In[27]:


learnerB = logisticClassify2();
wts=np.array([0.,0.,0.]);
learnerB.theta = wts
learnerB.regularisationtrain(XB, YB, initStep=1e-1,stopEpochs=1000,stopTol=1e-5,alpha=1.);
ml.plotClassify2D(learnerB,XB,YB)
print("Training error rate: ",learnerB.err(XB,YB))
plt.show()


# Problem 2: Shattering and VC dimensions

# a) The learner T(a+bx1) represents a vertical line .This learner can shatter (a) and (b) as the points can be on either side of the vertical line or across the same side of the vertical line.
# It cannot shatter case(c) and case(d) as for a particular combination of points we cannot classify all the points correctly.

# (b) The learner T((x1 − a)^2 + (x2 − b)^2 + c) represents a circle with center(a,b).
# This learner would be able to shatter case(a) and case(b),as the points can either stay inside the circle,outside the circle or one point inside and one point ouside the circle.
# It cannot shatter case(c) and case(d),as for a particular combination of classes we cannot draw a circle such that only 2 points lie inside the circle and the 3rd point lies outside the circle,case(d) has similar problems as well.

# (c) The learner T((a∗b)x1 + (c/a)x2) represents a straight line going throught the origin.
# This learner can shatter case(a) and case(b),as the line can be on either side of the points and in the middle to classify the points as +1 or -1 or into two seperate classes.

# (d)The learner  T(a + b ∗ x1 + c ∗ x2 ) · T( d + b ∗ x1 + c ∗ x2 )represents two parallel lines, with the gap and the slope changeable. The region outside the parallel lines will be +1 and the region inside will be -1.This learner can shatter (a),(b) and (c) as the line can be on either side of the point and in the middle to classify any set of points into seperate classes. 
# For case(d),it can shatter them as well, since for XOR case,the points with +1 can be inside the parallel lines and the points outside the parallel lines will be -1.
