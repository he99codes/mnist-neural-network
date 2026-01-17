from model import iterations,losses, lrs ,train_accuracy ,dev_accuracy, w1_norms,w2_norms,b1_norms,b2_norms

import matplotlib.pyplot as plt

plt.figure()
plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Training Loss vs Iterations")
plt.show()

# Plot accuracies
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(0, iterations, 10), train_accuracy, label='Training Accuracy', marker='o', markersize=3)
plt.plot(range(0, iterations, 10), dev_accuracy, label='Dev Accuracy', marker='s', markersize=3)
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Training vs Dev Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(losses, label='Training Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.grid(True)


plt.tight_layout()
plt.show()

# Plot
# import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Loss
axes[0, 0].plot(losses, color='red', linewidth=2)
axes[0, 0].set_xlabel('Iterations')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training Loss')
axes[0, 0].grid(True)

# Plot 2: Weight Norms
axes[0, 1].plot(w1_norms, label='W1 norm', linewidth=2)
axes[0, 1].plot(w2_norms, label='W2 norm', linewidth=2)
axes[0, 1].set_xlabel('Iterations')
axes[0, 1].set_ylabel('Weight Norm')
axes[0, 1].set_title('Weight Norms Over Time')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Plot 3: Bias Norms
# Plot 3: Bias Norms - LINES ONLY
axes[1, 0].plot(b1_norms, label='b1 norm', linewidth=2, color='blue')
axes[1, 0].plot(b2_norms, label='b2 norm', linewidth=2, color='orange')
axes[1, 0].set_xlabel('Iterations')
axes[1, 0].set_ylabel('Bias Norm')
axes[1, 0].set_title('Bias Norms Over Time')
axes[1, 0].legend()
axes[1, 0].grid(True)
axes[1, 0].set_facecolor('white')  # Ensure white background

# Plot 4: Accuracy
axes[1, 1].plot(range(0, iterations, 10), train_accuracy, label='Train Accuracy', marker='o', markersize=3)
axes[1, 1].plot(range(0, iterations, 10), dev_accuracy, label='Dev Accuracy', marker='s', markersize=3)
axes[1, 1].set_xlabel('Iterations')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].set_title('Train vs Dev Accuracy')
axes[1, 1].legend()
axes[1, 1].grid(True)


plt.tight_layout()
plt.show()