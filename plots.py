import json
import matplotlib.pyplot as plt

with open("performance.json", "r") as f:
    performance = json.load(f)

# Extract values from the performance list
epochs = list(range(1, len(performance) + 1))
train_losses = [epoch["avg_train_loss"] for epoch in performance]
val_losses = [epoch["avg_val_loss"] for epoch in performance]
train_accuracies = [epoch["train_accuracy"] for epoch in performance]
val_accuracies = [epoch["val_accuracy"] for epoch in performance]

# Plot Training and Validation Loss
plt.figure(figsize=(14, 6))

# Subplot for Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label="Training Loss")
plt.plot(epochs, val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.xticks([1] + epochs[9::10] + [epochs[-1]])

# Subplot for Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label="Training Accuracy")
plt.plot(epochs, val_accuracies, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.xticks([1] + epochs[9::10] + [epochs[-1]])

plt.tight_layout()

# Save the plot as an image file
plt.savefig("performance_plot.png", dpi=300)

plt.show()
