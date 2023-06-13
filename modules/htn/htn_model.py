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

# 변수의 형태별로 전처리 과정을 저장해둔 ColumnTransformer를 통해 변수 전처리를 하고 return 값을 DataFrame형태로 전달
# OneHoeEncoding을 적용하면 sex_1, sex_2와 같은 형태로 변수가 생기는데 이때 컬럼명을 추출하여 DataFrame형태로 변환

# 변수의 형태별로 전처리 과정을 저장해둔 ColumnTransformer를 통해 변수 전처리를 하고 return 값을 DataFrame형태로 전달
# OneHoeEncoding을 적용하면 sex_1, sex_2와 같은 형태로 변수가 생기는데 이때 컬럼명을 추출하여 DataFrame형태로 변환

def preprocess_data(df, transformer):
  # 범주형 Feature
  categorical_cols = ['sex','BE5_1','pa_aerobic','HE_HPfh1','HE_HPfh2','HE_HPfh3']
  # 연속형 Feature
  continuous_cols = list(set(df.columns.tolist()) - set(categorical_cols + ['DI1_pr']))
  # Target
  ignored_cols = ['DI1_pr']

  # 하나의 데이터에서 Train,Valid,Test 데이터셋으로 분할하여 진행하기 때문에 Train 데이터셋에 fit한 transformer를 사용
  processed_data = transformer.transform(df)

  # 범주형변수 OneHotEncoding 적용 시 생기는 변수명 추출
  categorical_encoder = transformer.named_transformers_['categorical']
  new_categorical_cols = categorical_encoder.get_feature_names_out(categorical_cols)

  # 연속형변수를 포함한 모든 변수명
  feature_names = continuous_cols + list(new_categorical_cols) +['DI1_pr']

  # return 데이터 DataFrame으로 변환
  processed_df = pd.DataFrame(processed_data, columns=feature_names)

  return processed_df

# HN16_ALL data를 통해 train, valid, test 데이터 분할 및 전처리 과정
def data_setup():
  # 전체 데이터 load
  DF = pd.read_csv('./data/HN16_ALL.csv', na_values=[' '])
  # 사용할 전체 변수명
  col = ['sex','age','HE_ht','HE_wt','HE_BMI', 'HE_sbp','HE_dbp','HE_glu','BE5_1','HE_HPfh1','HE_HPfh2','HE_HPfh3','pa_aerobic','DI1_pr']

  df = DF[col].dropna()
  # 모델링에 필요한 변수 사용을 위해 건강설문조사 항목 기분 19세 이상인 대상자를 추출
  df = df.loc[df['age'] >= 19]

  # 데이터를 load 했을 때 float 형태의 변수가 object로 load 되는 등의 문제 발생: 올바른 변수형태로 변경
  obj_col = ['sex','BE5_1','pa_aerobic','HE_HPfh1','HE_HPfh2','HE_HPfh3','DI1_pr']
  df = df.astype({i : ('int' if i in obj_col else 'float') for i in col}).dropna()
  df['DI1_pr'] = df['DI1_pr'].astype('int')

  # 종속변수인 고혈압 현재 유병 여부에서 비해당(청소년, 소아, 의사진단 받지 않음) 경우 제외: 없음, 있음의 경우만 포함하여 이진분류 계획
  df = df.loc[df['DI1_pr'] != 8]

  # train, valid, test 데이터셋 분할 종속변수를 기준으로 층화추출
  train , test = train_test_split(df, stratify = df['DI1_pr'], test_size =0.1)
  train, valid = train_test_split(train, stratify = train['DI1_pr'], test_size = 0.1)

  # 변수의 형태별로 다른 전처리 방법을 저장한 ColumnTransformer 정의 및 오버샘플링을 실시한 훈련데이터에 적용
  categorical_cols = ['sex','BE5_1','pa_aerobic','HE_HPfh1','HE_HPfh2','HE_HPfh3']
  continuous_cols = list(set(df.columns.tolist()) - set(categorical_cols + ['DI1_pr']))
  preprocess = ColumnTransformer([
      ('continuous', RobustScaler(), continuous_cols),
      ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
      ], remainder='passthrough')

  preprocess.fit(train)

  # preprocess_data 함수를 통해 훈련데이터에 fit된 transformer를 통해 train, valid, test 데이터셋 전처리 실행
  train = preprocess_data(train, preprocess)
  valid = preprocess_data(valid, preprocess)
  test = preprocess_data(test, preprocess)
  return train, valid, test

# 전달된 데이터를 모델에 input형태에 맞게 변환하는 과정
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

# train, valid, test 데이터셋을 분할하고 모델에 input 형태에 맞게 변환, 배치 단위로 데이터 전달하는 과정
class hypertensionDataModule(pl.LightningDataModule):
  def __init__(self, batch_size):
    self.batch_size = batch_size

    # hypertensionDataModule 클래스를 정의하면 prepare_data 함수가 실행되어 train, valid, test 데이터셋 생성
    self.prepare_data()

    self.prepare_data_per_node = True
    self._log_hyperparams = None
    self.allow_zero_length_dataloader_with_multiple_devices = False

  # data_setup 함수를 통해 train, valid, test 데이터셋을 분할하고 hypertensionDataset을 이용하여 tensor 형태로 데이터 변환
  def prepare_data(self):
    train, valid, test = data_setup()
    self.train = hypertensionDataset(train)
    self.valid = hypertensionDataset(train)
    self.test = hypertensionDataset(test)

  # 훈련과정 테스트 과정을 구별하여 필요한 DataModule을 전달할 수 있도록 하는 함수
  def setup(self, stage):
    if stage in (None, 'fit'):
      self.train
      self.valid

    if stage == 'test':
      self.test
  # 훈련데이터 배치단위 전달
  def train_dataloader(self):
    return DataLoader(self.train, batch_size = self.batch_size, shuffle=True)
  # 검증데이터 배치단위 전달
  def val_dataloader(self):
    return DataLoader(self.valid, batch_size = self.batch_size, shuffle=False)
  # 테스트데이터 배치단위 전달
  def test_dataloader(self):
    return DataLoader(self.test, batch_size = self.batch_size,  shuffle=False)

# 모델 구조 설계 및 train, valid, test 과정 진행 시 log 설정하고 손실,최적화함수 설정
class hypertensionClassifier(pl.LightningModule):
  def __init__(self, config: dict):
    super(hypertensionClassifier, self).__init__()

    # 입력변수 갯수(num_features), epoch, learning_rate 등 파라미터를 저장한 dictionary
    self.config = config

    # 모델 구조 설계
    # 첫번째 층(입력변수 갯수(28), hidden_size(100))
    # 활성화 함수 ReLU
    # 두번째 층(hidden_size(100), n_classes(1))
    self.fc1 = nn.Linear(self.config['input_dim'], self.config['hidden_dim'])
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(self.config['hidden_dim'], 1)

    # 성능평가를 위한 f1score
    self.f1 = F1Score(task="binary")

  # model의 입력에 대한 output을 내는 과정
  def forward(self, x, y = None):
    # 모델 구조에 따라 입력값을 각 층에 입력
    out = self.fc1(x)
    out = self.relu(out)
    out = self.fc2(out)
    out = torch.sigmoid(out)

    return out

  # 훈련과정에서 batch 데이터를 입력값으로 받아 model에 입력하고 손실함수를 통해 loss, f1_score 계산
  def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.forward(x)
    loss = nn.BCELoss()(y_hat, y.unsqueeze(1).float())

    self.log(
        "train_loss",
        loss,
        prog_bar=True,
        logger=True,
        batch_size=len(batch))
    return loss

  # 검증과정에서 batch 데이터를 입력값으로 받아 model에 입력하고 손실함수를 통해 loss, f1_score 계산
  def validation_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.forward(x)
    loss = nn.BCELoss()(y_hat, y.unsqueeze(1).float())
    f1 = self.f1(y_hat, y.unsqueeze(1).float())

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

  # 테스트과정에서 batch 데이터를 입력값으로 받아 model에 입력하고 손실함수를 통해 loss, f1_score 계산
  def test_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.forward(x)
    loss = nn.BCELoss()(y_hat, y.unsqueeze(1).float())
    f1 = self.f1(y_hat, y.unsqueeze(1).float())

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
  
  # 최적화 함수 설정: AdamW
  def configure_optimizers(self):
    return torch.optim.AdamW(self.parameters(), lr= self.config['lr'])