import matplotlib.pyplot as plt
import re
import numpy as np
def extract(time_string):
    return float(time_string.replace("s", ""))

log_file = 'data/model/dcrnn_DR_2_h_3_64-64_lr_0.01_bs_64_0612223001/info.log'

val_mae_values = []
stop_epoch = None
patience = 20
times = []
with open(log_file, 'r') as f:
    for line in f:
        if 'val_mae' in line:
            val_mae = float(line.split('val_mae: ')[1].split(', ')[0])
            val_mae_values.append(val_mae)
            times.append(extract(line.split('val_mae: ')[1].split(', ')[2]))
        if 'Early stopping at epoch' in line:
            stop_epoch = int(line.split('epoch: ')[1]) - patience

print(np.mean(times))
log_file = 'data/model/lstm/info6.log'

lstm_val_mae_values = []
lstm_stop_epoch = None
patience = 10
times = []
with open(log_file, 'r') as f:
    for line in f:
        if 'val:' in line:
            lstm_val_mae = float(line.split('val: ')[1].split(',')[0])
            lstm_val_mae_values.append(lstm_val_mae)
            times.append(extract(line.split('duration: ')[1]))
        if 'Early stopping! epoch:' in line:
            lstm_stop_epoch = int(line.split('epoch: ')[1]) - patience

print(np.mean(times))
log_file = 'data/model/lstm/info7.log'

lstm2_val_mae_values = []
lstm2_stop_epoch = None
patience = 10
times = []
with open(log_file, 'r') as f:
    for line in f:
        if 'val:' in line:
            lstm2_val_mae = float(line.split('val: ')[1].split(',')[0])
            lstm2_val_mae_values.append(lstm2_val_mae)
            times.append(extract(line.split('duration: ')[1]))
        if 'Early stopping! epoch:' in line:
            lstm2_stop_epoch = int(line.split('epoch: ')[1]) - patience

print(np.mean(times))


plt.plot(val_mae_values,color='r',label='DCRNN')
plt.plot(lstm_val_mae_values,color='b',label='LSTM (256 LSTM units)')
plt.plot(lstm2_val_mae_values,color='y',label='LSTM (50 LSTM units)')
if lstm_stop_epoch:
    plt.axvline(x=lstm_stop_epoch, color='b', linestyle='--')

if lstm2_stop_epoch:
    plt.axvline(x=lstm2_stop_epoch, color='y', linestyle='--')

if stop_epoch:
    plt.axvline(x=stop_epoch, color='r', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Validation MAE')
plt.title('DCRNN and LSTMs Learning Curve - 35 sensors')
plt.legend()
plt.show()