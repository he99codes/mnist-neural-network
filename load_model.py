import joblib
from neural_network import make_predictions,forward_propagation,get_predictions,gradient_descent,X_dev,X_train,Y_dev,Y_train,test_predictions,data_train,check_accuracy
from matplotlib import pyplot as plt
package = joblib.load("nn_model.pkg")

parameters = package["parameters"]

index = 40
current_image = X_dev[:,index]
predictions = make_predictions(parameters, X_dev[:,index])

current_image = current_image.reshape(28,28) *255
plt.gray()
plt.imshow(current_image,interpolation='nearest')

label = Y_dev[index]
print(predictions,label)

plt.show()

# print()