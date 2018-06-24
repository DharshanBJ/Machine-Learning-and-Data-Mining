
# coding: utf-8

# DHARSHAN BIRUR JAYAPRABHU
# 54773179

# 1(a)

# In[40]:


import numpy as np
import mltools as ml
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

iris = np.genfromtxt("/Users/dharshanbj/Desktop/HW5-code/data/iris.txt",delimiter=None) # load the text file 
X=iris[:,0:2]
plt.scatter(X[:,0],X[:,1])
plt.title("Scatter plot of the first two features")


# 1(B)

# In[41]:


#try for different initializations and select the model values with the least cost
mincost=np.inf
for i in range(10):
    z,Y,l=ml.cluster.kmeans(X,5)
    if l<mincost:
        mincost=l
        z_leastcost=z
        Y_leastcost=Y
         


# In[42]:


#ml.plotClassify2D(None,X,Y[0])
plt.scatter(Y_leastcost[:,0],Y_leastcost[:,1],c='r',marker='x')#mark centers
ml.plotClassify2D(None,X,z_leastcost)#color points based on clustering
plt.title("k-means clustering for k=5")


# In[43]:


#try for different initializations and select the model values with the least cost
mincost=np.inf
for i in range(10):
    z,Y,l=ml.cluster.kmeans(X,20)
    if l<mincost:
        mincost=l
        z_leastcost=z
        Y_leastcost=Y
        


# In[44]:


plt.scatter(Y_leastcost[:,0],Y_leastcost[:,1],c='r',marker='x')#mark centers
ml.plotClassify2D(None,X,z_leastcost)#color points based on clustering
plt.title('k-means clustering for k=10')


# 1(C)

# In[35]:


#Agglomerative clustering


# In[36]:


#single linkage
z,Y=ml.cluster.agglomerative(X,5,method='min')
#plt.scatter(Y[:,0],Y[:,1],c='r',marker='x')#mark centers
ml.plotClassify2D(None,X,z)#color points based on clustering
plt.title("Aggglomerative clustering - single linkage for k=5")


# In[37]:


#single linkage
z,Y=ml.cluster.agglomerative(X,20,method='min')
#plt.scatter(Y[:,0],Y[:,1],c='r',marker='x')#mark centers
ml.plotClassify2D(None,X,z)#color points based on clustering
plt.title("Aggglomerative clustering - single linkage for k=20")


# In[38]:


#Complete linkage
z,Y=ml.cluster.agglomerative(X,5,method='max')
#plt.scatter(Y[:,0],Y[:,1],c='r',marker='x')#mark centers
ml.plotClassify2D(None,X,z)#color points based on clustering
plt.title("Aggglomerative clustering - Complete linkage for k=5")


# In[39]:


#Complete linkage
z,Y=ml.cluster.agglomerative(X,20,method='max')
#plt.scatter(Y[:,0],Y[:,1],c='r',marker='x')#mark centers
ml.plotClassify2D(None,X,z)#color points based on clustering
plt.title("Aggglomerative clustering - Complete linkage for k=20")


# When the clusters are hyper-spherical in shape k-means clustering works better.
# In the case of agglomerative clustering,minimum spanning tree will be formed if we consider the minimum
# distances.If we use max distance, it will be avoiding elongated clusters,also clusters are built incrementaly.

# Problem 2:

# 2(a)

# In[45]:


X = np.genfromtxt("/Users/dharshanbj/Desktop/HW5-code/data/faces.txt", delimiter=None) # load face dataset
plt.figure()
# pick a data point i for display
img = np.reshape(X[i,:],(24,24)) # convert vectorized data point to 24x24 image patch
plt.imshow( img.T , cmap="gray") # display image patch; you may have to squint


# In[48]:


#remove the mean
mn=np.mean(X)
X0=X-mn
plt.figure()
img = np.reshape(X0[i,:],(24,24)) # convert vectorized data point to 24x24 image patch
plt.imshow( img.T , cmap="gray") # display image patch; you may have to squint


# 2(b)

# In[50]:


import scipy.linalg
U,S,V = scipy.linalg.svd(X0, False)
W = U.dot( np.diag(S) ); 
print (W.shape, V.shape)


# 2(c)

# In[61]:


err = [None]*10
for k in range(10):
    Xhat0 = W[:,:k].dot( V[:k,:] ) 
    err[k] = ((X0-Xhat0)**2).mean()
plt.plot(range(10),err,'r');
plt.xlabel('Number of Eigen directions')
plt.ylabel('Mean Squared error')


# 2(d)

# In[55]:


for k in range(3):
    alp = 2*np.median( np.abs( W[:,k] ));
    image1 = np.reshape(mn + alp*V[k,:], (24,24)); 
    image2 = np.reshape(mn - alp*V[k,:], (24,24)); # TODO: subplots
    plt.figure();
    f,(ax1,ax2) = plt.subplots(1,2); 
    ax1.imshow(image1.T, cmap="gray"); 
    ax2.imshow(image2.T, cmap="gray");


# 2(e)

# In[59]:


for i in [24,35]:
    image = X[i,:];
    image = np.reshape(image, (24,24)); 
    plt.figure()
    f,ax = plt.subplots(1,5); 
    ax[0].imshow(image.T, cmap="gray");
    for j,k in enumerate([5,10,50,100]):
        image = mn + W[i,0:k].dot( V[0:k,:] );
        image = np.reshape(image, (24,24));
        ax[j+1].imshow(image.T, cmap="gray");


# 2 (f)

# In[64]:


ids = np.floor( 4800*np.random.rand(40) ); # pick some data 
ids = ids.astype('int')
plt.rcParams['figure.figsize'] = (8.0, 8.0)
coord,params = ml.transforms.rescale(W[:,0:2]) 
for i in ids:
    loc = (coord[i,0],coord[i,0]+.5, coord[i,1],coord[i,1]+.5)
    plt.imshow(np.reshape(X[i,:],(24,24)).T , cmap="gray", extent=loc );
    plt.axis((-3,3,-3,3))

