import warnings 
warnings.filterwarnings('ignore')

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

import json
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from scipy import stats
from imblearn.over_sampling import SMOTENC

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
  continuous_cols = list(set(df.columns.tolist()) - set(categorical_cols + ['HE_sbp']))

  processed_data = transformer.transform(df)
  categorical_encoder = transformer.named_transformers_['categorical']
  new_categorical_cols = categorical_encoder.get_feature_names_out(categorical_cols)
  feature_names = continuous_cols + list(new_categorical_cols) + ['HE_sbp']
  processed_df = pd.DataFrame(processed_data, columns=feature_names)

  return processed_df

def remove_outliers(data, threshold=3):
    z_scores = stats.zscore(data)
    outliers = np.abs(z_scores) > threshold
    cleaned_data = data[~outliers]
    return cleaned_data

def data_setup():
  DF = pd.read_csv('./data/HN16_ALL.csv', na_values=[' '])
  col = ['sex','age','HE_ht','HE_wt','HE_BMI','HE_glu','BE5_1','HE_HPfh1','HE_HPfh2','HE_HPfh3','pa_aerobic', 'HE_sbp']
  df = DF[col].dropna()
  df = df.loc[df['age'] >= 19]

  obj_col = ['sex','BE5_1','pa_aerobic','HE_HPfh1','HE_HPfh2','HE_HPfh3']
  df = df.astype({i : ('int' if i in obj_col else 'float') for i in col}).dropna()


  train , test = train_test_split(df, test_size =0.1)
  train, valid = train_test_split(train, test_size = 0.1)

  train['HE_sbp_class'] = [0 if x < 130 else 1 for x in train['HE_sbp']]

  sm = SMOTENC(random_state=2023, categorical_features=[0,6,7,8,9,10])
  x_smote = train.iloc[:,:-1].values
  y_smote = train.iloc[:, -1].values

  x_over, y_over = sm.fit_resample(x_smote, y_smote)

  over_train = pd.DataFrame(x_over)
  over_train.columns = train.columns.tolist()[:-1]

  categorical_cols = ['sex','BE5_1','pa_aerobic','HE_HPfh1','HE_HPfh2','HE_HPfh3']
  continuous_cols = list(set(df.columns.tolist()) - set(categorical_cols + ['HE_sbp']))

  preprocess = ColumnTransformer([
      ('continuous', StandardScaler(), continuous_cols),
      ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
      ], remainder='passthrough')
  preprocess.fit(over_train)


  train = preprocess_data(over_train, preprocess)
  train['HE_sbp'] = remove_outliers(train['HE_sbp'])
  train['HE_glu'] = remove_outliers(train['HE_glu'])



  valid = preprocess_data(valid, preprocess)
  valid['HE_sbp'] = remove_outliers(valid['HE_sbp'])
  valid['HE_glu'] = remove_outliers(valid['HE_glu'])

  test = preprocess_data(test, preprocess)
  test['HE_sbp'] = remove_outliers(test['HE_sbp'])
  test['HE_glu'] = remove_outliers(test['HE_glu'])

  train = train.dropna()
  valid = valid.dropna()
  test = test.dropna()

  train['HE_sbp'] = np.log1p(train['HE_sbp'])
  valid['HE_sbp'] = np.log1p(valid['HE_sbp'] )
  test['HE_sbp'] = np.log1p(test['HE_sbp'] )

  return train, valid, test

class SBPDataset(Dataset):
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

class SBPDataModule(pl.LightningDataModule):
  def __init__(self, batch_size):
    self.batch_size = batch_size
    self.prepare_data()
    self.prepare_data_per_node = True
    self._log_hyperparams = None
    self.allow_zero_length_dataloader_with_multiple_devices = False

  def prepare_data(self):
    train, valid, test = data_setup()
    self.train = SBPDataset(train)
    self.valid = SBPDataset(train)
    self.test = SBPDataset(test)

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

class SBPRegression(pl.LightningModule):
  def __init__(self, config : dict):
    super(SBPRegression, self).__init__()
    self.config = config

    self.relu = nn.ReLU()
    self.fc1 = nn.Linear(self.config['num_features'], 1024)
    self.fc2 = nn.Linear(1024, 512)
    self.fc3 = nn.Linear(512, 128)
    self.fc4 = nn.Linear(128, 1)

    # 평균 제곱 오차(Mean Squared Error, MSE)와 평균 절대 오차(Mean Absolute Error, MAE)의 장점을 결합한 손실함수
    # 입력값의 차이에 따라 다르게 반응. 작은 오차에는 제곱 오차를 사용하여 회귀 모델을 학습, 큰 오차에는 절대 오차를 사용하여 로버스트한 학습을 수행합니다.
    # 이를 통해 이상치(outlier)에 민감하지 않고, 일반적인 데이터 패턴에 잘 적응하는 모델을 구축
    # self.loss_func = nn.functional.huber_loss
    # self.loss_func = nn.MSELoss(reduction = 'mean')
    self.loss_func = nn.L1Loss()

  def forward(self, x, y = None):
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    x = self.relu(x)
    x = self.fc3(x)
    x = self.relu(x)
    x = self.fc4(x)
    return x

  def inverse_transform(self, y_log):
    return torch.exp(y_log)

  def training_step(self, batch, batch_idx):
    x, y_log = batch
    y_pred_log = self.forward(x)
    y_pred = self.inverse_transform(y_pred_log)

    # delta의 의미
    # loss = self.loss_func(y_pred, self.inverse_transform(y_log).squeeze(0), delta = 20)
    loss = self.loss_func(y_pred, self.inverse_transform(y_log).squeeze(0))

    self.log("train_loss", loss, prog_bar=True, logger=True)

    return loss
    self.log(
        "train_loss",
        loss,
        prog_bar=True,
        logger=True,
        batch_size=len(batch))

    return loss

  def validation_step(self, batch, batch_idx):
    x, y_log = batch
    y_pred_log = self.forward(x)
    y_pred = self.inverse_transform(y_pred_log)
    # loss = self.loss_func(y_pred, self.inverse_transform(y_log).squeeze(0), delta = 20)
    loss = self.loss_func(y_pred, self.inverse_transform(y_log).squeeze(0))

    self.log(
        "val_loss",
        loss,
        prog_bar=True,
        logger=True,
        batch_size=len(batch))

    return loss

  def test_step(self, batch, batch_idx):
    x, y_log = batch
    y_pred_log = self.forward(x)
    y_pred = self.inverse_transform(y_pred_log)
    # loss = self.loss_func(y_pred, self.inverse_transform(y_log).squeeze(0), delta = 20)
    loss = self.loss_func(y_pred, self.inverse_transform(y_log).squeeze(0))

    self.log(
        "test_loss",
        loss,
        prog_bar=True,
        logger=True,
        batch_size=len(batch))

    return loss

  def configure_optimizers(self):
      return torch.optim.AdamW(self.parameters(), lr= self.config['lr'])

