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

# Plot data
fig, axs = plt.subplots(2, 3, figsize=(16, 8), sharey=False)

# Plot Loss
axs[0][0].plot(epochs_baseline, loss_baseline, label='Loss', marker='o', color='blue')
axs[0][0].set_title('Loss Over Epochs (baseline)')
axs[0][0].set_xlabel('Epoch')
axs[0][0].set_ylabel('Loss')
axs[0][0].grid(True)

# Plot Accuracy
axs[0][1].plot(epochs_baseline, accuracy_baseline, label='Accuracy', marker='o', color='green')
axs[0][1].set_title('Accuracy Over Epochs (baseline)')
axs[0][1].set_xlabel('Epoch')
axs[0][1].set_ylabel('Accuracy')
axs[0][1].grid(True)

# Plot F1 Score
axs[0][2].plot(epochs_baseline, f1_score_baseline, label='F1 Score', marker='o', color='red')
axs[0][2].set_title('F1 Score Over Epochs (baseline)')
axs[0][2].set_xlabel('Epoch')
axs[0][2].set_ylabel('F1 Score')
axs[0][2].grid(True)

axs[1][0].plot(epochs, loss, label='Loss', marker='o', color='blue')
axs[1][0].set_title('Loss Over Epochs (proposed)')
axs[1][0].set_xlabel('Epoch')
axs[1][0].set_ylabel('Loss')
axs[1][0].grid(True)

# Plot Accuracy
axs[1][1].plot(epochs, accuracy, label='Accuracy', marker='o', color='green')
axs[1][1].set_title('Accuracy Over Epochs (proposed)')
axs[1][1].set_xlabel('Epoch')
axs[1][1].set_ylabel('Accuracy')
axs[1][1].grid(True)

# Plot F1 Score
axs[1][2].plot(epochs, f1_score, label='F1 Score (proposed)', marker='o', color='red')
axs[1][2].set_title('F1 Score Over Epochs')
axs[1][2].set_xlabel('Epoch')
axs[1][2].set_ylabel('F1 Score')
axs[1][2].grid(True)

# Adjust layout
plt.tight_layout()
plt.savefig(file_name + ".png")
plt.show()
