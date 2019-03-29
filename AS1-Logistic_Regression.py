#!/usr/bin/env python
# coding: utf-8

# # M2608.001300 기계학습 기초 및 전기정보 응용<br> Assignment 1: Logistic Regression

# In[62]:


def learn_and_return_weights(X, y):
    from sklearn.linear_model import LogisticRegression
    
    # YOUR CODE COMES HERE
    train=LogisticRegression(solver='lbfgs',max_iter=10000)
    train.fit(X,y)
    w=train.coef_
    w=w.flatten()
    b=train.intercept_    
    # w: coefficient of the model to input features,
    # b: bias of the model
    return w, b


# ## Dataset load & Plot

# In[63]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


# In[64]:


data = np.loadtxt('data.csv', delimiter=',')
X = data[:, :2]
y = data[:, 2]
label_mask = np.equal(y, 1)

plt.scatter(X[:, 0][label_mask], X[:, 1][label_mask], color='red')
plt.scatter(X[:, 0][~label_mask], X[:, 1][~label_mask], color='blue')
plt.show()


# ## Problem 1-1. sklearn model로 Logistic Regression 모델 train 시켜보기
# scikit-learn library의 LogisticRegression 클래스를 이용해 train 시켜 보세요.

# In[65]:


def plot_data_and_weights(X, y, w, b):
    plt.scatter(X[:, 0][label_mask], X[:, 1][label_mask], color='red')
    plt.scatter(X[:, 0][~label_mask], X[:, 1][~label_mask], color='blue')

    x_lin = np.arange(20, 70)
    y_lin = -(0.5 + b + w[0] * x_lin) / w[1]

    plt.plot(x_lin, y_lin, color='black');
    plt.show()

w, b = learn_and_return_weights(X, y)
plot_data_and_weights(X, y, w, b)


# ## Problem 1-2. numpy로 Logistic Regression 짜보기
# scikit-learn library를 사용하지 않고 Logistic Regression을 구현해보세요.

# In[66]:


def learn_and_return_weights_numpy(X, y):
    # YOUR CODE COMES HERE
    dim=len(X[0]) #dimension of data
    n=len(X) #number of data
    
    #normalize the data
    normalize=np.max(X,0)
    Xn=X/normalize
    ext_w=np.random.random((dim+1,1))-0.5 #put bias as w0
    ext_X=np.append(np.ones((n,1)),Xn,axis=1) #add ones to X for the bias
    lr=0.01 # learning_rate
    for epoch in range(1000):
        hyp=np.matmul(ext_X,ext_w)
        sig=1/(1+np.exp(-hyp)) # sigmoid
        err=sig - y.reshape(n,1) # h(x)-y
        cost=np.mean(-y*np.log(sig)-(1-y)*np.log(1-sig)) #just to check
        t=np.transpose(ext_X)
        dw=1/n * np.matmul(t,err)
        ext_w=ext_w-4*dw
    ext_w.flatten()
    b=ext_w[0]
    wn=ext_w[1:]
    w=wn.reshape(1,2)/np.max(X,0)
    w=w.flatten()
    b=b/np.max(w)
    w=w/np.max(w)
    # w: coefficient of the model to input features,
    # b: bias of the model
    return w, b


# In[67]:


w, b = learn_and_return_weights_numpy(X, y)
plot_data_and_weights(X, y, w, b)


# ## Problem 2. sklearn model로 Logistic Regression 모델 train 시켜보기 + regularizer 사용하기
# scikit-learn library의 Logistic Regression 에 대한 API문서 (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)를 읽어보고, L1-regularization을 사용할 때와 L2-regularization을 사용할 때의 weight의 변화를 살펴보세요.

# In[68]:


def learn_and_return_weights_l1_regularized(X, y):    
    # YOUR CODE COMES HERE
    train=LogisticRegression(penalty='l1',max_iter=1000)
    train.fit(X,y)
    w=train.coef_
    w=w.flatten()
    b=train.intercept_
    return w, b

def learn_and_return_weights_l2_regularized(X, y):    
    # YOUR CODE COMES HERE
    train=LogisticRegression(penalty='l2',max_iter=1000)
    train.fit(X,y)
    w=train.coef_
    w=w.flatten()
    b=train.intercept_
    return w, b


# In[69]:


def get_dataset():
    D = 1000
    N = 80

    X = np.random.random((N, D))
    w = np.zeros(D)
    w[0] = 1
    w[1] = 1
    
    e = np.random.random(N) - 0.5
    
    y_score = np.dot(X, w)
    y_score_median = np.median(y_score)
    print(y_score.max(), y_score.min(), y_score_median)
    
    # y_score += 0.01 * e
    y = y_score >= y_score_median
    y = y.astype(np.int32)
    
    return (X[:N // 2], y[:N // 2]), (X[N // 2:], y[N // 2:])


# In[70]:


(x_train, y_train), (x_test, y_test) = get_dataset()

w_l1, b_l1 = learn_and_return_weights_l1_regularized(x_train, y_train)
w_l2, b_l2 = learn_and_return_weights_l2_regularized(x_train, y_train)

print(w_l1[:5])
print(w_l2[:5])


# ## Problem 3. Logistic Regression으로 multi-class classification 하기
# Logistic Regression은 기본적으로 binary classifier 입니다. 즉, input *X*를 2개의 class로 밖에 분류하지 못합니다. 하지만, 이같은 Logistic Regression 모델을 연달아 사용한다면 data를 여러 class로 분류할 수도 있겠죠?
# 
# 참고: https://en.wikipedia.org/wiki/Multiclass_classification#Transformation_to_binary

# In[71]:


# YOUR CODE COMES IN THIS CELL
already_trained=False #boolean to check if data is already learned
result=[0 for i in range(10)]
def learn(x,y):
    prob_of=[0 for i in range(10)] #prob of each class 0 to 9
    for i in range(10):
        chk=np.zeros(len(y))
        for j in range(len(y)):
            if y[j] == i:
                chk[j]=1
            else:
                chk[j]=0
        prob_of[i]=LogisticRegression(penalty='l2').fit(x,chk)
    return prob_of

def classifier(x):
    global already_trained
    global result
    if already_trained ==False:
        result=learn(x_train[:1000], y_train[:1000])
        already_trained=True
    classified=[0 for i in range(10)]
    for i in range(10):
        classified[i]=(result[i].predict_proba(x.reshape(1,-1)))[0][1]
    
    y=np.argmax(classified)
    return y # return label from x.


# In[72]:


def get_dataset():
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((-1, 28 * 28)).astype(np.float32)
    x_test = x_test.reshape((-1, 28 * 28)).astype(np.float32)
    return (x_train, y_train), (x_test, y_test)
(x_train, y_train), (x_test, y_test) = get_dataset()


# In[73]:


preds = np.array([classifier(x) for x in x_test])
accuracy = np.sum(preds == y_test) / y_test.shape[0]
print('Accuracy:', accuracy)


# In[ ]:




