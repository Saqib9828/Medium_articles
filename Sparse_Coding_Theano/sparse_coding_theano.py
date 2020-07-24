import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
import pickle
import time
theano.config.exception_verbosity = 'high'
train = pd.read_csv("/content/train.csv")
#test = pd.read_csv("/content/train.csv")
#just reading the 4200 images out of 42000 images
trX = np.array(train.values[0:4200][:, 1:], dtype=np.float32)/256
#just reading the labels of 4200 images out of 42000 images
trY = np.array(train.values[0:4200][:, 0], dtype=int)
#teX = np.array(train.values[0:4200][:, 1:], dtype=np.float32)/256
trY_onehot = np.zeros((trY.shape[0], 10), dtype=np.float32)
#here we encode the Y as onehot encoder
trY_onehot[np.arange(trY.shape[0]), trY] = 1
#Relu function
def rectify(Z):
 return T.maximum(Z, 0.)
#initialisation the weights
def init_weights(shape):
 return theano.shared(np.random.randn(*shape)*0.01)
#function for updating the parameters
def get_updates(cost, params, lr=np.float32(0.05)):
 updates = []
 grads = T.grad(cost, params)
 for p, g in zip(params, grads):
 updates.append([p, p - (g * lr)])
 return updates
#defining the model
def model(X, w_h, w_o):
 h = rectify(T.dot(X, w_h))
 return T.nnet.softmax(T.dot(h, w_o))
w_h = theano.shared(np.random.randn(784, 60).astype(np.float32)*0.01,
name='w_h')
w_o = theano.shared(np.random.randn(60, 10).astype(np.float32)*0.01,
name='w_o')
X = T.fmatrix(name='X')
labels = T.fmatrix(name='labels')
prediction = model(X, w_h, w_o)
cost = T.mean(T.nnet.categorical_crossentropy(prediction, labels))
updates = get_updates(cost, [w_h, w_o])
train_func = theano.function(
 inputs=[X, labels], outputs=cost, updates=updates,
 allow_input_downcast=True)
predict_func = theano.function(
 inputs=[X], outputs=prediction, allow_input_downcast=True)
### first we encode the input images into a sparse vector through a sparse encoder
from sklearn.decomposition import DictionaryLearning
sparse_coder=DictionaryLearning(784,max_iter=25,transform_max_iter=5)
x_new=sparse_coder.fit_transform(trX)
### here we are using the encoded images for training a neural network for
classification
costs = []
niters = 200
t = time.clock()
for i in range(niters):
 print("Iter: "+str(i))
 costt = train_func(x_new, trY_onehot)
 print("Cost: "+str(costt))
 costs.append(float(costt))
 print("time ", (time.clock()-t))
 t = time.clock()
pickle.dump(costs, open("costs.p", 'wb'))
plt.scatter(np.arange(len(costs)), costs)
plt.savefig("cost.png")
plt.show()