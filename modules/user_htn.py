from modules.dbp.dbp_predict import *
from modules.sbp.sbp_predict import *
from modules.htn.htn_predict import *
from modules.seed_everything import *

seed_everything(2023)

class User_HTN():
    def __init__(self, df):
        self.df = df
        
    def user_type(self):
        if len(list(set(self.df.columns.tolist()).intersection(set(['HE_sbp', 'HE_dbp'])))) == 0:
            user_sbp = SBP(self.df).predict()
            user_dbp = DBP(self.df).predict()

            self.df['HE_sbp'] = user_sbp
            self.df['HE_dbp'] = user_dbp

            predict_HTN = HTN(self.df).predict()
            if predict_HTN == 0:
                HTN_label = '정상'

            elif predict_HTN == 1:
                HTN_label = '고혈압'

            string_list = ['수축기혈압','이완기혈압','고혈압분류']
            result_list = [user_sbp, user_dbp, HTN_label]

            return dict(zip(string_list, result_list))
        else:

            user_htn = HTN(self.df).predict()

            if user_htn == 0:
                HTN_label = '정상'

            elif user_htn == 1:
                HTN_label = '고혈압'

            return {'고혈압분류':HTN_label}


# test_df = pd.read_csv('./data/htn_test_df.csv')
# result = User_HTN(test_df).user_type()
# print(result)

# htn_test_df = pd.read_csv('./data/last_last_test.csv')
# result = User_HTN(htn_test_df).user_type()
# print(result)