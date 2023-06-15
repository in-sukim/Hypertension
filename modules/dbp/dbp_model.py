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

# 변수의 형태별로 전처리 과정을 저장해둔 ColumnTransformer를 통해 변수 전처리를 하고 return 값을 DataFrame형태로 전달
# OneHoeEncoding을 적용하면 sex_1, sex_2와 같은 형태로 변수가 생기는데 이때 컬럼명을 추출하여 DataFrame형태로 변환

def preprocess_data(df, transformer):
  # 범주형 Feature
  categorical_cols = ['sex','BE5_1','pa_aerobic','HE_HPfh1','HE_HPfh2','HE_HPfh3']
  # 연속형 Feature
  continuous_cols = list(set(df.columns.tolist()) - set(categorical_cols + ['HE_dbp']))

  # 하나의 데이터에서 Train,Valid,Test 데이터셋으로 분할하여 진행하기 때문에 Train 데이터셋에 fit한 transformer를 사용
  processed_data = transformer.transform(df)

  # 범주형변수 OneHotEncoding 적용 시 생기는 변수명 추출
  categorical_encoder = transformer.named_transformers_['categorical']
  new_categorical_cols = categorical_encoder.get_feature_names_out(categorical_cols)

  # 연속형변수를 포함한 모든 변수명
  feature_names = continuous_cols + list(new_categorical_cols) + ['HE_dbp']

  # return 데이터 DataFrame으로 변환
  processed_df = pd.DataFrame(processed_data, columns=feature_names)

  return processed_df

# Z_score를 이용한 이상치 제거
# EDA 결과 HE_glu(공복혈당)에 이상치가 존재하고 왜도가 큰 것을 관찰
# 이상치 존재 할 경우 모델의 일반화 성능을 저해할 수 있다고 판단. 공복혈당은 이상치 제거 전처리 단계를 수행

def remove_outliers(data, threshold=3):
    z_scores = stats.zscore(data)
    outliers = np.abs(z_scores) > threshold
    cleaned_data = data[~outliers]
    return cleaned_data

# HN16_ALL data를 통해 train, valid, test 데이터 분할 및 전처리 과정
def data_setup():
  # 전체 데이터 load
  DF = pd.read_csv('/content/drive/MyDrive/딥러닝모델과추론/과제//final/data/HN16_ALL.csv', na_values=[' '])

  # 사용할 전체 변수명
  col = ['sex','age','HE_ht','HE_wt','HE_BMI','HE_glu','BE5_1','HE_HPfh1','HE_HPfh2','HE_HPfh3','pa_aerobic', 'HE_dbp']
  df = DF[col].dropna()

  # 모델링에 필요한 변수 사용을 위해 건강설문조사 항목 기분 19세 이상인 대상자를 추출
  df = df.loc[df['age'] >= 19]

  # 데이터를 load 했을 때 float 형태의 변수가 object로 load 되는 등의 문제 발생: 올바른 변수형태로 변경
  obj_col = ['sex','BE5_1','pa_aerobic','HE_HPfh1','HE_HPfh2','HE_HPfh3']
  df = df.astype({i : ('int' if i in obj_col else 'float') for i in col}).dropna()


  # train, valid, test 데이터셋 분할
  train , test = train_test_split(df, test_size =0.1)
  train, valid = train_test_split(train, test_size = 0.1)

  # 변수의 형태별로 다른 전처리 방법을 저장한 ColumnTransformer 정의 및 훈련데이터에 적용
  categorical_cols = ['sex','BE5_1','pa_aerobic','HE_HPfh1','HE_HPfh2','HE_HPfh3']
  continuous_cols = list(set(df.columns.tolist()) - set(categorical_cols + ['HE_dbp']))

  preprocess = ColumnTransformer([
      ('continuous', RobustScaler(), continuous_cols),
      ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
      ], remainder='passthrough')
  preprocess.fit(train)

  # preprocess_data 함수를 통해 훈련데이터에 fit된 transformer를 통해 train, valid, test 데이터셋 전처리 실행
  # HE_glu(공복혈당)에 앞서 정의된 remove_outliers함수를 통해 동일한 기준으로 각각의 데이터에서 이상치 제거 수행
  train = preprocess_data(train, preprocess)
  train['HE_glu'] = remove_outliers(train['HE_glu'])

  valid = preprocess_data(valid, preprocess)
  valid['HE_glu'] = remove_outliers(valid['HE_glu'])
  
  test = preprocess_data(test, preprocess)
  test['HE_glu'] = remove_outliers(test['HE_glu'])

  train = train.dropna()
  valid = valid.dropna()
  test = test.dropna()

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
  
# 모델 구조 설계 및 train, valid, test 과정 진행 시 log 설정하고 손실,최적화함수 설정
class DBPRegression(pl.LightningModule):
  def __init__(self, config : dict):
    super(DBPRegression, self).__init__()


    # 입력변수 갯수(num_features), epoch, learning_rate 등 파라미터를 저장한 dictionary
    self.config = config

    # 모델 구조 설계
    # 첫번째 층(입력변수 갯수(26), 128)
    # 두번째 층(128, 64)
    # 세번째 층(64, 32)
    # 네번째 층(32, 1)

    # 활성화 함수 ReLU
    self.fc1 = nn.Linear(self.config['num_features'], 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 32)
    self.fc4 = nn.Linear(32, 1)

    # 손실함수 설정
    self.loss_func = nn.L1Loss()

  # model의 입력에 대한 output을 내는 과정
  def forward(self, x, y = None):
    # 모델 구조에 따라 입력값을 각 층에 입력
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = torch.relu(self.fc3(x))
    x = torch.relu(self.fc4(x))
    return x

  # 훈련과정에서 batch 데이터를 입력값으로 받아 model에 입력하고 손실함수를 통해 loss 계산
  def training_step(self, batch, batch_idx):
    x, y = batch
    y_pred = self.forward(x)
    loss = self.loss_func(y_pred, y.squeeze(0))

    # 로그 설정
    self.log(
        "train_loss",
        loss,
        prog_bar=True,
        logger=True,
        batch_size=len(batch))

    return loss

  # 검증과정에서 batch 데이터를 입력값으로 받아 model에 입력하고 손실함수를 통해 loss 계산
  def validation_step(self, batch, batch_idx):
    x, y = batch
    y_pred = self.forward(x)
    loss = self.loss_func(y_pred, y.squeeze(0))

    # 로그 설정
    self.log(
        "val_loss",
        loss,
        prog_bar=True,
        logger=True,
        batch_size=len(batch))

    return loss

  # 테스트과정에서 batch 데이터를 입력값으로 받아 model에 입력하고 손실함수를 통해 loss 계산
  def test_step(self, batch, batch_idx):
    x, y = batch
    y_pred = self.forward(x)
    loss = self.loss_func(y_pred, y.squeeze(0))

    # 로그 설정
    self.log(
        "test_loss",
        loss,
        prog_bar=True,
        logger=True,
        batch_size=len(batch))

    return loss

  # 최적화 함수 설정: AdamW
  def configure_optimizers(self):
      return torch.optim.AdamW(self.parameters(), lr= self.config['lr'])
