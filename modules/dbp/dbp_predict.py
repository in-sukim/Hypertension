import pandas as pd
from modules.dbp.dbp_model import *
from modules.seed_everything import *
seed_everything(2023)

class DBP():
    def __init__(self, df):
        self.config = json.load(open('./modules/dbp/dbp_model_config.json'))
        self.model = DBPRegression(self.config)
        self.model.load_state_dict(torch.load(f = './modules/dbp/dbp_model.pth'))
        self.model.eval()
        self.df = df
        self.data = df
        self.data_setup()

    def data_setup(self):
        col = ['sex','age','HE_ht','HE_wt','HE_BMI','HE_glu','BE5_1','HE_HPfh1','HE_HPfh2','HE_HPfh3','pa_aerobic']
        obj_col = ['sex','BE5_1','HE_HPfh1','HE_HPfh2','HE_HPfh3','pa_aerobic']

        # 입력값
        df = self.data.astype({i : ('int' if i in obj_col else 'float') for i in col}).dropna()

        # 훈련에서 사용한 전체 데이터
        DF = pd.read_csv('./data/HN16_ALL.csv', na_values=[' '])
        full_df = DF[col].dropna()
        full_df = full_df.loc[full_df['age'] >= 19]
        full_df = full_df.astype({i : ('int' if i in obj_col else 'float') for i in col}).dropna()

        train , test = train_test_split(full_df, test_size =0.1, random_state= 2023)
        train, valid = train_test_split(train, test_size = 0.1, random_state= 2023)

        categorical_cols = ['sex','BE5_1','HE_HPfh1','HE_HPfh2','HE_HPfh3','pa_aerobic']
        continuous_cols = list(set(df.columns.tolist()) - set(categorical_cols))

        preprocess = ColumnTransformer([
            ('continuous', RobustScaler(), continuous_cols),
            ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ], remainder='passthrough')
        
        preprocess.fit(train)

        processed_data = preprocess.transform(df)

        categorical_encoder = preprocess.named_transformers_['categorical']
        new_categorical_cols = categorical_encoder.get_feature_names_out(categorical_cols)
        feature_names = continuous_cols + list(new_categorical_cols)

        processed_df = pd.DataFrame(processed_data, columns=feature_names)

        self.data = processed_df
    
    
    def predict(self):
        x = torch.FloatTensor(self.data.iloc[:].values)
        pred = self.model(x).item()
        return pred


# test_df = pd.read_csv('/Users/insu/deep_learning/data/last_last_test.csv')
# print(DBP(test_df).predict())