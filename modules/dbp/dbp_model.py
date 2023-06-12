import warnings 
warnings.filterwarnings('ignore')
import json 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import torch.nn as nn
from torchmetrics import MeanSquaredError

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import random
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
seed_everything(2023)

def preprocess_data(df, transformer):
  categorical_cols = ['sex','BE5_1','pa_aerobic','HE_HPfh1','HE_HPfh2','HE_HPfh3']
  continuous_cols = list(set(df.columns.tolist()) - set(categorical_cols + ['HE_dbp']))

  processed_data = transformer.transform(df)
  categorical_encoder = transformer.named_transformers_['categorical']
  new_categorical_cols = categorical_encoder.get_feature_names_out(categorical_cols)
  feature_names = continuous_cols + list(new_categorical_cols) + ['HE_dbp']
  processed_df = pd.DataFrame(processed_data, columns=feature_names)

  return processed_df
  
def data_setup():  
  DF = pd.read_csv('./data/HN16_ALL.csv', na_values=[' '])
  col = ['sex','age','HE_ht','HE_wt','HE_BMI','HE_glu','BE5_1','HE_HPfh1','HE_HPfh2','HE_HPfh3','pa_aerobic', 'HE_dbp']
  df = DF[col].dropna()
  df = df.loc[df['age'] >= 19]

  obj_col = ['sex','BE5_1','pa_aerobic','HE_HPfh1','HE_HPfh2','HE_HPfh3']
  df = df.astype({i : ('int' if i in obj_col else 'float') for i in col}).dropna()
#   df['HE_dbp'] = np.log1p(df['HE_dbp'])


  train , test = train_test_split(df,  test_size =0.1)
  train, valid = train_test_split(train, test_size = 0.1)

  categorical_cols = ['sex','BE5_1','pa_aerobic','HE_HPfh1','HE_HPfh2','HE_HPfh3']
  continuous_cols = list(set(df.columns.tolist()) - set(categorical_cols + ['HE_dbp']))

  preprocess = ColumnTransformer([
      ('continuous', RobustScaler(), continuous_cols),
      ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
      ], remainder='passthrough')
  preprocess.fit(train)
  
  train = preprocess_data(train, preprocess)
  valid = preprocess_data(valid, preprocess)
  test = preprocess_data(test, preprocess)
  return train, valid, test

class DBPDataset(Dataset):
  def __init__(self, df):
    self.df = df
    self.data = df
    self.x = self.data.iloc[:, :-1].values
    self.y = self.data.iloc[:, -1].values
  
  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    x = torch.FloatTensor(self.x[index]).squeeze(0)
    y = torch.FloatTensor([self.y[index]]).squeeze(0)
    return x, y
  
class DBPDataModule(pl.LightningDataModule):
  def __init__(self, batch_size):
    self.batch_size = batch_size
    self.prepare_data()
    self.prepare_data_per_node = True
    self._log_hyperparams = None
    self.allow_zero_length_dataloader_with_multiple_devices = False

  def prepare_data(self):
    train, valid, test = data_setup()
    self.train = DBPDataset(train)
    self.valid = DBPDataset(train)
    self.test = DBPDataset(test)

  def setup(self, stage):
    if stage in (None, 'fit'):
      self.train
      self.valid
    
    if stage == 'test':
      self.test
  
  def train_dataloader(self):
    return DataLoader(self.train, batch_size = self.batch_size, shuffle=True)

  def val_dataloader(self):
    return DataLoader(self.valid, batch_size = self.batch_size, shuffle=False)

  def test_dataloader(self):
    return DataLoader(self.test, batch_size = self.batch_size,  shuffle=False)
  
class DBPRegression(pl.LightningModule):
  def __init__(self, config : dict):
    super(DBPRegression, self).__init__()
    self.config = config

    self.fc1 = nn.Linear(self.config['num_features'], 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 32)
    self.fc4 = nn.Linear(32, 1)

    self.loss_func = nn.MSELoss(reduction='mean')

  def forward(self, x, y = None):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = torch.relu(self.fc3(x))
    x = torch.relu(self.fc4(x))
    return x
  
  def training_step(self, batch, batch_idx):
    x, y = batch
    y_pred = self.forward(x)
    loss = self.loss_func(y_pred, y.squeeze(0))

    self.log(
        "train_loss", 
        loss, 
        prog_bar=True, 
        logger=True, 
        batch_size=len(batch))

    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    y_pred = self.forward(x)
    loss = self.loss_func(y_pred, y.squeeze(0))

    self.log(
        "val_loss", 
        loss, 
        prog_bar=True, 
        logger=True, 
        batch_size=len(batch))

    return loss

  def test_step(self, batch, batch_idx):
    x, y = batch
    y_pred = self.forward(x)
    loss = self.loss_func(y_pred, y.squeeze(0))

    self.log(
        "test_loss", 
        loss, 
        prog_bar=True, 
        logger=True, 
        batch_size=len(batch))

    return loss

  def configure_optimizers(self):
      return torch.optim.AdamW(self.parameters(), lr= self.config['lr'])
