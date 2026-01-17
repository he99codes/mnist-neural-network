import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from neural_network import make_predictions,forward_propagation,get_predictions,gradient_descent,X_dev,X_train,Y_dev,Y_train,test_predictions,data_train,check_accuracy

iterations = 1010
learning_rate = 0.01
lambda_ = 0

parameters,losses, lrs ,train_accuracy ,dev_accuracy, w1_norms,w2_norms,b1_norms,b2_norms = gradient_descent(X_train,Y_train,X_dev,Y_dev,iterations,learning_rate,lambda_)



print(f"b1_norms last value: {b1_norms[-1]}")
print(f"b2_norms last value: {b2_norms[-1]}")
print(f"Are they the same? {np.allclose(b1_norms, b2_norms)}")



test_predictions(99,parameters,X_train,Y_train)
test_predictions(51,parameters,X_dev,Y_dev)
test_predictions(40,parameters,X_train,Y_train)
test_predictions(104,parameters,X_dev,Y_dev)



dev_predictions = make_predictions(parameters,X_dev) 
x = check_accuracy(dev_predictions,Y_dev)
print(x)


import matplotlib.pyplot as plt

plt.figure()
plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Training Loss vs Iterations")
plt.show()
