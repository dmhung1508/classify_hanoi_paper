import matplotlib.pyplot as plt
import json,os
file_path = os.path.join('weight', 'weight.json')
with open(file_path, 'r') as f:
    data = json.load(f)
loss = []
a = data['Realtime']
for fold in a:
    for epoch in a[fold]:
        loss.append(a[fold][epoch]['Training']['Average training loss'])

print(loss)
# Số fold và số epoch trong mỗi fold
num_folds = 5
epochs_per_fold = 5

# Tính tổng số lượng epoch
total_epochs = num_folds * epochs_per_fold

# Chia list loss thành các nhóm tương ứng với mỗi fold
loss_per_fold = [loss[i:i+epochs_per_fold] for i in range(0, total_epochs, epochs_per_fold)]

# Vẽ đồ thị
for fold, loss_fold in enumerate(loss_per_fold):
    plt.plot(range(1, epochs_per_fold + 1), loss_fold, label=f'Fold {fold + 1}')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch per Fold')
plt.legend()
plt.savefig('weight/loss_plot.png')