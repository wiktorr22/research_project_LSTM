import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import os
import sys
sys.path.append(os.getcwd())
import torch.optim as optim
import datetime
from lib import metrics, utils
from model.pytorch_LSTM.loss import masked_mae_loss
import time
data = utils.load_dataset("data/METR-LA", 64, 64)
scaler = data['scaler']
train_loader = data['train_loader']
test_loader = data['test_loader']
val_loader = data['val_loader']
time.strftime('%m%d%H%M%S')
SENSORS = 35

# logging.
_log_dir = "data/model/lstm"
_writer = SummaryWriter('runs/' + _log_dir)

_logger = utils.get_logger(_log_dir, __name__, 'info_.log')

def _prepare_data(x, y):
    x, y = _get_x_y(x, y)
    x, y = _get_x_y_in_correct_dims(x, y)
    return x.to(device), y.to(device)


def _get_x_y(x, y):
    """
    :param x: shape (batch_size, seq_len, num_sensor, input_dim)
    :param y: shape (batch_size, horizon, num_sensor, input_dim)
    :returns x shape (seq_len, batch_size, num_sensor, input_dim)
             y shape (horizon, batch_size, num_sensor, input_dim)
    """
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    x = x.permute(1, 0, 2, 3)
    y = y.permute(1, 0, 2, 3)
    return x, y


def _get_x_y_in_correct_dims( x, y):
    """
    :param x: shape (seq_len, batch_size, num_sensor, input_dim)
    :param y: shape (horizon, batch_size, num_sensor, input_dim)
    :return: x: shape (seq_len, batch_size, num_sensor * input_dim)
             y: shape (horizon, batch_size, num_sensor * output_dim)
    """
    batch_size = x.size(1)
    x = x[..., :1].view(3, batch_size, SENSORS * 1)
    y = y[..., :1].view(3, batch_size, SENSORS * 1)
    return x, y


def _compute_loss(y_true, y_predicted):
    y_true = scaler.inverse_transform(y_true)
    y_predicted = scaler.inverse_transform(y_predicted)
    return masked_mae_loss(y_predicted, y_true)


def evaluate(dataset='val', batches_seen=0):
    """
    Computes mean L1Loss
    :return: mean L1Loss
    """
    with torch.no_grad():
        model.eval()

        val_iterator = data['{}_loader'.format(dataset)].get_iterator()
        losses = []

        y_truths = []
        y_preds = []
        for _, (x, y) in enumerate(val_iterator):
            x, y = _prepare_data(x, y)


            output = model(x)
            loss =+ _compute_loss(y, output)
            losses.append(loss.item())

            y_truths.append(y.cpu())
            y_preds.append(output.cpu())

        mean_loss = np.mean(losses)

        y_preds = np.concatenate(y_preds, axis=1)
        y_truths = np.concatenate(y_truths, axis=1)  # concatenate on batch dimension

        for t in range(y_preds.shape[0]):
            y_truth = scaler.inverse_transform(y_truths[t])
            y_pred = scaler.inverse_transform(y_preds[t])

            mae = metrics.masked_mae_np(y_pred, y_truth, null_val=0)
            mape = metrics.masked_mape_np(y_pred, y_truth, null_val=0)
            rmse = metrics.masked_rmse_np(y_pred, y_truth, null_val=0)
            if dataset != 'val':

                print(
                    "Horizon {:02d}, MAE: {:.4f}, MAPE: {:.4f}, RMSE: {:.2f}".format(
                        t + 1, mae, mape, rmse
                    )
                )

        return mean_loss
# LSTM model
class CustomTrafficLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, num_horizons):
        super(CustomTrafficLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_horizons = num_horizons
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, output_size)
        self.count = 0

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        outputs = []
        for t in range(self.num_horizons):  # number of steps to forecast
            out, (h0, c0) = self.lstm(x, (h0, c0))
            out = out[-1, :, :]  # Get the last time step output (seq_len, batch, hidden) -> (batch, hidden)
            out = self.fc(out)  # (batch, output_size)
            outputs.append(out.unsqueeze(0))  # (1, batch, output_size)

            # Replace the oldest value in x with the new prediction
            out = out.unsqueeze(0)  # (1, batch, output_size)
            x = torch.cat((x[1:], out), dim=0)  # (new_seq_len, batch, input_size)

        outputs = torch.cat(outputs, dim=0)  # (forecast_steps, batch, output_size)
        return outputs

# Initialize the model
lstmUnits = 256 #50 #256

model = CustomTrafficLSTM(SENSORS, lstmUnits, 2, SENSORS, 3)

criterion = nn.L1Loss()
mae_criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
best_val_loss = float('inf')
patience = 10
trigger_times = 0

train_mae = []
val_mae = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_logger.info('Start training ...')
for epoch in range(500):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
    print(f'{current_time} - Epoch {epoch + 1} started')
    model.train()
    train_loss = 0
    train_mae_accum = 0
    start_time = time.time()
    for _, (inputs, labels) in enumerate(train_loader.get_iterator()):
        x, y = _prepare_data(inputs, labels)
        optimizer.zero_grad()
        outputs = model(x)
        loss = _compute_loss(y, outputs)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_mae_accum += mae_criterion(outputs, y).item()

    _logger.info("epoch complete")
    _logger.info("evaluating now!")
    val_loss = evaluate("val")
    end_time = time.time()
    _logger.info("val: %s, epoch: %s, duration: %.1fs" % (val_loss, epoch, (end_time - start_time)))
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping!")
            _logger.info("Early stopping! epoch: %s" % epoch)
            break


# Test evaluation
test_loss = 0
test_mae_accum = 0
result = evaluate('test')
print(f'MAE {result}')