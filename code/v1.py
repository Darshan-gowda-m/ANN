import matplotlib.pyplot as plt

# Function to parse logs
def parse_log(filename):
    epochs = []
    train_errors = []
    test1_acc = []
    test2_acc = []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip() and line[0].isdigit():
                parts = line.strip().split()
                if len(parts) >= 8:
                    epochs.append(int(parts[0]))
                    train_errors.append(float(parts[1]))
                    test1_acc.append(float(parts[4]))
                    test2_acc.append(float(parts[7]))
    return epochs, train_errors, test1_acc, test2_acc

# Parse each log
e1, tr1, a1_1, a2_1 = parse_log('train_log_shades1.txt')
e2, tr2, a1_2, a2_2 = parse_log('train_log_shades2.txt')
e3, tr3, a1_3, a2_3 = parse_log('train_log_pose.txt')

# Plot all training errors
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(e1, tr1, label='shades1.net')
plt.plot(e2, tr2, label='shades2.net')
plt.plot(e3, tr3, label='pose.net')
plt.xlabel("Epochs")
plt.ylabel("Training Error")
plt.title("Training Error vs Epochs")
plt.grid(True)
plt.legend()

# Plot all accuracies
plt.subplot(1, 2, 2)
plt.plot(e1, a1_1, label='shades1 - Test1')
plt.plot(e1, a2_1, label='shades1 - Test2')
plt.plot(e2, a1_2, label='shades2 - Test1')
plt.plot(e2, a2_2, label='shades2 - Test2')
plt.plot(e3, a1_3, label='pose - Test1')
plt.plot(e3, a2_3, label='pose - Test2')
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Test Accuracies vs Epochs")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("training_comparison.png")  # optional save
plt.show()


