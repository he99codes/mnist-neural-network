# MNIST Neural Network from Scratch

A multi-layer neural network built from scratch using only NumPy to classify MNIST handwritten digits.

## Results
- **Training Accuracy:** ~90%
- **Dev Accuracy:** ~90%
- **No overfitting!**

##  Architecture
- **Input Layer:** 784 neurons (28×28 pixels)
- **Hidden Layers:** 10 neurons with ReLU activation
- **Output Layer:** 10 neurons with Softmax activation



##  Usage

1. Download MNIST dataset from Kaggle
2. Place `train.csv` in the project directory
3. Run training:
```bash
python train.py
```

##  Features
-  Built entirely with NumPy (no frameworks!)
-  L2 Regularization
-  Learning rate decay
-  Training visualizations
-  Parameter norm tracking

##  Project Structure
```
mnist-neural-network/
└── model/
      └── nn_model.pkl
├── neural_network.py      # Core NN implementation
├── train.py              # Training script
├── load_model.py
|── nn_model.pkl
├── README.md             # This file    
└── .gitignore           # Git ignore rules
```

##  What I Learned
- Forward and backward propagation from scratch
- Gradient descent optimization
- Matrix calculus and chain rule
- Weight initialization strategies
- Debugging neural networks
- The importance of proper variable management in Python!

##  Future Improvements
- [ ] Implement mini-batch training
- [ ] Add Adam/momentum optimizer
- [ ] Try deeper architectures
- [ ] Add dropout regularization
- [ ] Implement batch normalization



## 🔗 Links
- [Kaggle Notebook](https://www.kaggle.com/code/priyanshusharmamf/hand-written-digits-recognition-mnist)
- [MNIST Dataset](https://www.kaggle.com/code/priyanshusharmamf/hand-written-digits-recognition-mnist)
```

*


