import numpy as np
import tensorflow as tf
import pickle
import word_vec
import regression_based
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo,encoding='bytes')
    fo.close()
    return dict

embeddings = word_vec.get_vectors()

# load cifar-10 data
train_d = unpickle('cifar-10-batches-py/data_batch_1')
X = train_d[b'data']
Y = train_d[b'labels']
for i in range(2,6):    
    train_d = unpickle('cifar-10-batches-py/data_batch_'+str(i))
    X = np.vstack((X,train_d[b'data']))
    Y = Y + train_d[b'labels']

X = np.reshape(X, (50000,3,32,32)).transpose(0,2,3,1)

# first 4w datas for training, 1w for validation 
X_train = X[:40000]
Y_train = Y[:40000]
X_validation  = X[40000:]
Y_validation = Y[40000:]

# all labels of cifar-10 dataset
class_labels = unpickle('cifar-10-batches-py/batches.meta')[b'label_names']

Y_8_train = np.array(Y_train)
X_8_train = np.array(X_train)

# let label 1, 4 be testing data ('automobile' and 'deer')
removed_indices = np.where(Y_8_train!=1)
Y_8_train = Y_8_train[removed_indices]
X_8_train = X_8_train[removed_indices]
removed_indices = np.where(Y_8_train!=4)
Y_8_train = Y_8_train[removed_indices]
X_8_train = X_8_train[removed_indices]

Y_8_train = [embeddings[str(class_labels[i], encoding='ascii')] for i in Y_8_train]

Y_8_validation = np.array(Y_validation)
X_8_validation = np.array(X_validation)

# train with L2 loss regression
regression_based.train(X_8_train, Y_8_train)

# validation using 1, 4 data
Y_2_validation = np.array(Y_validation)
X_2_validation = np.array(X_validation)
indices = np.where(np.logical_or(Y_2_validation==1 ,Y_2_validation==4))
Y_2_validation = Y_2_validation[indices]
X_2_validation = X_2_validation[indices]

# predict image data of label 1, 4
validaiton_embeddings = regression_based.predict(X_2_validation)

# embedding of label 1, 4
targets_embeddings = []
for i in ['automobile', 'deer']:
    targets_embeddings.append(embeddings[i])
targets_embeddings = np.array(targets_embeddings, dtype=np.float32)


# find neighbors of label 1, 4 image data 
neigh = KNeighborsClassifier(n_neighbors = 1)
neigh.fit(targets_embeddings, [1,4])
Y_pred_validation = neigh.predict(validaiton_embeddings)


print(accuracy_score(Y_2_validation, Y_pred_validation))


