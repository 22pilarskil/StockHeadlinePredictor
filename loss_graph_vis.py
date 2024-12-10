import matplotlib.pyplot as plt

file_name = "report"
# File path
file_path = f'/Users/liampilarski/Downloads/{file_name}.txt'
file_path_baseline = f'/Users/liampilarski/Downloads/{file_name}_baseline.txt'

# Read data
epochs = []
loss = []
accuracy = []
f1_score = []

epochs_baseline = []
loss_baseline = []
accuracy_baseline = []
f1_score_baseline = []

with open(file_path, 'r') as file:
    lines = file.readlines()
    for line in lines[2:]:  # Skip the first two lines (headers)
        epoch, l, acc, f1 = map(float, line.strip().split(', '))
        epochs.append(int(epoch))
        loss.append(l)
        accuracy.append(acc)
        f1_score.append(f1)

with open(file_path_baseline, 'r') as file:
    lines = file.readlines()
    for line in lines[2:]:  # Skip the first two lines (headers)
        epoch, l, acc, f1 = map(float, line.strip().split(', '))
        epochs_baseline.append(int(epoch))
        loss_baseline.append(l)
        accuracy_baseline.append(acc)
        f1_score_baseline.append(f1)

# Plot data (one row of three graphs)
fig, axs = plt.subplots(1, 3, figsize=(16, 4), sharey=False)

# Plot Loss
axs[0].plot(epochs_baseline, loss_baseline, label='Baseline Model Loss', marker='o', color='blue')
axs[0].plot(epochs, loss, label='Proposed Model Loss', marker='o', color='red')
axs[0].set_title('Loss Over Epochs')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].grid(True)
axs[0].legend()

# Plot Accuracy
axs[1].plot(epochs_baseline, accuracy_baseline, label='Baseline Model Accuracy', marker='o', color='blue')
axs[1].plot(epochs, accuracy, label='Proposed Model Accuracy', marker='o', color='red')
axs[1].set_title('Accuracy Over Epochs')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')
axs[1].grid(True)
axs[1].legend()

# Plot F1 Score
axs[2].plot(epochs_baseline, f1_score_baseline, label='Baseline Model F1 Score', marker='o', color='blue')
axs[2].plot(epochs, f1_score, label='Proposed Model F1 Score', marker='o', color='red')
axs[2].set_title('F1 Score Over Epochs')
axs[2].set_xlabel('Epoch')
axs[2].set_ylabel('F1 Score')
axs[2].grid(True)
axs[2].legend()

# Adjust layout
plt.tight_layout()
plt.savefig(file_name + "_overlay.png")
plt.show()
