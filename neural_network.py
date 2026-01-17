import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


data = pd.read_csv('train.csv')


data = np.array(data)
m,n = data.shape

split_index = int(0.8*m)
np.random.shuffle(data)

data_dev = data[split_index:].T

X_dev = data_dev[1:n]/255



Y_dev = data_dev[0]

data_train = data[:split_index].T

X_train = data_train[1:n]/255


Y_train = data_train[0]

def sigmoid(z):
    return 1/(1+np.exp(-z))


def initialize_parameters():
    W1 = np.random.rand(10, 784)*0.1
    b1 = np.zeros((10, 1)) 
    W2 = np.random.rand(10, 10)*0.1
    b2 = np.zeros((10, 1) )

    parameters = {
        'W1' : W1,
        'W2' : W2,
        'b1' : b1,
        'b2' : b2
    }
    return parameters


def one_hot_code(Y):
    hot_code_Y = np.zeros((Y.size,Y.max()+1))
    hot_code_Y[np.arange(Y.size),Y] = 1
    hot_code_Y = hot_code_Y.T

    return hot_code_Y




def Relu_optimal(z):
    return np.maximum(0,z)
    
def ReLU_deriv(Z):
    return (Z > 0).astype(float)


def SoftMax(Z):
    Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    

def forward_propagation(parameters,X):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    
    Z1 = np.dot(W1,X) + b1
    A1 = Relu_optimal(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = SoftMax(Z2)

    cache = {
        'Z1' : Z1,
        'A1' : A1,
        'Z2' : Z2,
        'A2' : A2
    }

    return A2,cache



def backward_propagation(parameters,A2,cache,X,Y,lambda_):
    W1 = parameters['W1']
    W2 = parameters['W2']
    b1,b2 = parameters['b1'],parameters['b2']
    m,n = X.shape
    
    A1 = cache['A1']
    Z2 = cache['Z2']
    Z1 = cache['Z1']
    

    
    
    dz2 = A2 - one_hot_code(Y)

    dw2 = 1/m*(np.dot(dz2,A1.T))
    
    dz1 = np.dot(W2,dz2) * ReLU_deriv(Z1)
    db2 = 1 / m * np.sum(dz2, axis=1, keepdims=True)
    db1 = 1 / m * np.sum(dz1, axis=1, keepdims=True)
    
    dw1 = 1/m*np.dot(dz1,X.T)
   

    
#   regularization    //////

    dw1 = dw1 + (lambda_/m)*W1
    dw2 = dw2 + (lambda_/m)*W2
    
    grads = {
        'dw1' : dw1,
        'dw2' : dw2,
        'db1' : db1,
        'db2' : db2
    }
    return grads



def update_parameters(parameters,grads,learning_rate):
    W1 = parameters['W1']
    W2 = parameters['W2']
    b1 = parameters['b1']
    b2 = parameters['b2']

    dw1  = grads['dw1']
    dw2 = grads['dw2']
    db1 = grads['db1']
    db2 = grads['db2']

    W1 = W1 - learning_rate*dw1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dw2
    b2 = b2 - learning_rate*db2
    
    parameters = {
        'W1' : W1,
        'W2' : W2,
        'b1' : b1,
        'b2' : b2
    }
    return parameters



def get_predictions(A2):
    return np.argmax(A2,0)

def check_accuracy(predictions,Y):
    print(predictions,Y)
    return np.sum(predictions==Y)/Y.size




def compute_loss(A2, Y, parameters,lambda_):
    m = Y.size
    W1 = parameters['W1']
    W2 = parameters['W2'] 

    # loss_regularization = 0
#  regularization /////
    
    loss_regularization = (lambda_/(2*m))*(np.sum(W1*W1) +np.sum(W2*W2))
    eps = 1e-15
    log_probs = -np.log(np.clip(A2[Y, np.arange(m)], eps, 1 - eps))
    # log_probs = -np.log(A2[Y, np.arange(m)])
    loss = np.sum(log_probs) / m
    
    return loss + loss_regularization



def make_predictions(parameters,X):
    A2,cache = forward_propagation(parameters,X)
    predictions = get_predictions(A2)
    return predictions


def gradient_descent(X,Y,X_dev,Y_dev,iterations,learning_rate,lambda_):
    parameters = initialize_parameters()
    W1 = parameters['W1']
    W2 = parameters['W2']
    b1 = parameters['b1']
    b2 = parameters['b2']
    train_accuracy = []
    dev_accuracy = []
    losses = []
    lrs = []


    
    w1_norms = []
    w2_norms = []
    b1_norms = []
    b2_norms = []


    
    for i  in range(iterations):
        A2,cache = forward_propagation(parameters,X)
        grads = backward_propagation(parameters,A2,cache,X,Y,lambda_)

        if (i % 250 ==0):
            learning_rate = learning_rate/3
        loss_i = compute_loss(A2,Y,parameters,lambda_)
        losses.append(loss_i)
        lrs.append(learning_rate)
        
        parameters = update_parameters(parameters,grads,learning_rate)



        
        w1_norms.append(np.linalg.norm(parameters['W1']))
        w2_norms.append(np.linalg.norm(parameters['W2']))
        b1_norms.append(np.linalg.norm(parameters['b1']))
        b2_norms.append(np.linalg.norm(parameters['b2']))




        
       
        if i %10 ==0:
            print(f" number of iterations : {i}")
            predictions_2 = get_predictions(A2)
            acu_train  = check_accuracy(predictions_2,Y)
            loss = compute_loss(A2,Y,parameters,lambda_)
            print(f"check accuracy : {acu_train}")
            print(f"loss : {loss}")

    #  Accuracy for the Dev set
            A2_dev,_ = forward_propagation(parameters,X_dev)
            dev_pre = get_predictions(A2_dev)
            dev_acu = check_accuracy(dev_pre,Y_dev)
            dev_accuracy.append(dev_acu)
            
    #  Accuracy for Train set
            train_accuracy.append(acu_train)
            
            
            
        
    return parameters,losses, lrs,train_accuracy,dev_accuracy,w1_norms,w2_norms,b1_norms,b2_norms
    

def test_predictions(index,parameters,X,Y):
    current_image = X[:,index]
    prediction = make_predictions(parameters,X[:,index])
    label = Y[index]
    print(prediction)
    print(label)

    current_image = current_image.reshape(28,28) *255
    plt.gray()
    plt.imshow(current_image,interpolation='nearest')
    plt.show()