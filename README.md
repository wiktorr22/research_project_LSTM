# Effectiveness of Graph Neural Networks and Simpler Network Models in Various Traffic Scenarios

This is a PyTorch implementation of Long Short-Term Memory model used in comparison with [Diffusion Convolutional Recurrent Neural Network]((https://arxiv.org/abs/1707.01926))
The DCRNN implementation is available [here](https://github.com/mehulbhuradia/DCRNN_PyTorch)

## Requirements 
Dependency can be installed using the following command:
```bash
pip install -r requirements.txt
```
## Data Preparation
The traffic data files for Los Angeles (METR-LA) and the Bay Area (PEMS-BAY), i.e., `metr-la.h5` and `pems-bay.h5`, are available at [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g), and should be
put into the `data/` folder.

Run the following commands to generate train/test/val dataset at  `data/{METR-LA,PEMS-BAY}/{train,val,test}.npz`.
```bash
# Create data directories
mkdir -p data/{METR-LA,PEMS-BAY}

# METR-LA
python -m scripts.generate_training_data --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python -m scripts.generate_training_data --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5
```

## Model training and evaluation
All variables and hyperparameters should be set directly in the model class lstm_model.py  
```bash
python model/pytorch_LSTM/lstm_model.py    
```