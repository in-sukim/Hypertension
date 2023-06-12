import warnings 
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTENC

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import torch.nn as nn
from torchmetrics import Accuracy, F1Score

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import json
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
# DI1_pr
def preprocess_data(df, transformer):
  categorical_cols = ['sex','BE5_1','pa_aerobic','HE_HPfh1','HE_HPfh2','HE_HPfh3']
  # continuous_cols = list(set(df.columns.tolist()) - set(categorical_cols + ['DI1_dg']))
  continuous_cols = list(set(df.columns.tolist()) - set(categorical_cols + ['DI1_pr']))
  # ignored_cols = ['DI1_dg'] 
  ignored_cols = ['DI1_pr'] 


  processed_data = transformer.transform(df)
  categorical_encoder = transformer.named_transformers_['categorical']
  new_categorical_cols = categorical_encoder.get_feature_names_out(categorical_cols)
  # feature_names = continuous_cols + list(new_categorical_cols) +['DI1_dg'] 
  feature_names = continuous_cols + list(new_categorical_cols) +['DI1_pr'] 
  processed_df = pd.DataFrame(processed_data, columns=feature_names)

  return processed_df

def data_setup():  

  DF = pd.read_csv('./data/HN16_ALL.csv', na_values=[' '])
  # col = ['sex','age','HE_ht','HE_wt','HE_BMI', 'HE_sbp','HE_dbp','HE_glu','BE5_1','HE_HPfh1','HE_HPfh2','HE_HPfh3','pa_aerobic','DI1_dg']
  col = ['sex','age','HE_ht','HE_wt','HE_BMI', 'HE_sbp','HE_dbp','HE_glu','BE5_1','HE_HPfh1','HE_HPfh2','HE_HPfh3','pa_aerobic','DI1_pr']
  df = DF[col].dropna()
  df = df.loc[df['age'] >= 19]

  # obj_col = ['sex','BE5_1','pa_aerobic','HE_HPfh1','HE_HPfh2','HE_HPfh3','DI1_dg']
  obj_col = ['sex','BE5_1','pa_aerobic','HE_HPfh1','HE_HPfh2','HE_HPfh3','DI1_pr']
  df = df.astype({i : ('int' if i in obj_col else 'float') for i in col}).dropna()
  # df['DI1_dg'] = df['DI1_dg'].astype('int')


  # train , test = train_test_split(df, stratify = df['DI1_dg'], test_size =0.1)
  # train, valid = train_test_split(train, stratify = train['DI1_dg'], test_size = 0.1)
  df['DI1_pr'] = df['DI1_pr'].astype('int')


  train , test = train_test_split(df, stratify = df['DI1_pr'], test_size =0.1)
  train, valid = train_test_split(train, stratify = train['DI1_pr'], test_size = 0.1)

  X_train, y_train = train.iloc[:, :-1], train.iloc[:,-1]
  sm = SMOTENC(random_state=2023, categorical_features=[8,9, 10, 11,12],sampling_strategy=.5)
  X_train_over, y_train_over = sm.fit_resample(X_train, y_train)

  over_train = pd.concat([X_train_over, y_train_over], axis = 1)

  categorical_cols = ['sex','BE5_1','pa_aerobic','HE_HPfh1','HE_HPfh2','HE_HPfh3']
  # continuous_cols = list(set(df.columns.tolist()) - set(categorical_cols + ['DI1_dg']))
  continuous_cols = list(set(df.columns.tolist()) - set(categorical_cols + ['DI1_pr']))
  preprocess = ColumnTransformer([
      ('continuous', StandardScaler(), continuous_cols),
      ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
      ], remainder='passthrough')

  preprocess.fit(over_train)
  
  train = preprocess_data(over_train, preprocess)
  valid = preprocess_data(valid, preprocess)
  test = preprocess_data(test, preprocess)
  return train, valid, test

class hypertensionDataset(Dataset):
  def __init__(self, df):
    self.df = df
    self.data = self.df
    self.x = self.data.iloc[:, :-1].values
    self.y = self.data.iloc[:, -1].values

  
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, index):
    x = torch.FloatTensor(self.x[index]).squeeze(0)
    y = torch.LongTensor([self.y[index]]).squeeze(0)
    return x, y

class hypertensionDataModule(pl.LightningDataModule):
  def __init__(self, batch_size):
    self.batch_size = batch_size
    self.prepare_data()
    self.prepare_data_per_node = True
    self._log_hyperparams = None
    self.allow_zero_length_dataloader_with_multiple_devices = False
    
  def prepare_data(self):
    train, valid, test = data_setup()
    self.train = hypertensionDataset(train)
    self.valid = hypertensionDataset(train)
    self.test = hypertensionDataset(test)
  
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

class hypertensionClassifier(pl.LightningModule):
  def __init__(self, config: dict):
    super(hypertensionClassifier, self).__init__()
    self.config = config


    self.fc1 = nn.Linear(self.config['num_features'], self.config['hidden_size'])
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(self.config['hidden_size'], config['n_classes'])
    self.loss_func = nn.CrossEntropyLoss()

    self.f1 = F1Score(task="binary", num_classes= config['n_classes'])
  def forward(self, x, y = None):
    out = self.fc1(x)
    out = self.relu(out)
    # out = self.dropout(out)
    out = self.fc2(out)

    return out
  
  def training_step(self, batch, batch_idx):
    x, y = batch
    logits = self.forward(x)
    loss = self.loss_func(logits, y.squeeze(0))
    preds = torch.argmax(logits, dim=1)

    self.log(
        "train_loss", 
        loss, 
        prog_bar=True, 
        logger=True, 
        batch_size=len(batch))
    

    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    logits = self.forward(x)
    loss = self.loss_func(logits, y.squeeze(0))
    preds = torch.argmax(logits, dim=1)
    f1 = self.f1(preds, y.squeeze(0))

    self.log(
        "val_loss", 
        loss, 
        prog_bar=True, 
        logger=True, 
        batch_size=len(batch))
    self.log(
        "val_f1", 
        f1, 
        prog_bar=True, 
        logger=True, 
        batch_size=len(batch))

    return {"val_loss": loss, "val_f1": f1}

  def test_step(self, batch, batch_idx):
    x, y = batch
    logits = self.forward(x)
    loss = self.loss_func(logits, y.squeeze(0))
    preds = torch.argmax(logits, dim=1)

    class2_preds = (preds == 2)
    class2_labels = (y == 2)

    f1 = self.f1(preds, y.squeeze(0))
    self.log(
        "test_loss", 
        loss, 
        prog_bar=True, 
        logger=True, 
        batch_size=len(batch))

    self.log(
        "test_f1", 
        f1, 
        prog_bar=True, 
        logger=True, 
        batch_size=len(batch))
    

    return {"test_loss": loss, "test_f1": f1}

  def configure_optimizers(self):
      return torch.optim.AdamW(self.parameters(), lr= self.config['lr'])