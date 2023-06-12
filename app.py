from flask import Flask, request, render_template
from modules.user_htn import *
import openai
import re
import argparse

parser = argparse.ArgumentParser(description='Argparse Tutorial')
parser.add_argument('--apikey', type= str, help='OpenAPIKey')
args = parser.parse_args()

app = Flask(__name__)

@app.route('/', methods=['GET'])
def main():
    return render_template('form.html')


@app.route('/result', methods=['POST', 'GET'])
def result():
    # POST 요청에서 전달된 데이터 가져오기
    sex = request.form.get('sex')
    if sex == '남자':
        sex = '1'
    else:
        sex = '2'
    sex = int(sex)
    age = int(request.form.get('age'))
    height = float(request.form.get('height'))
    weight = float(request.form.get('weight'))
    # bmi = request.form.get('bmi')
    bmi = round(weight / (height ** 2), 1)
    sbp = request.form.get('sbp')
    dbp = request.form.get('dbp')
    if sbp:
        sbp = sbp
    else:
        sbp = '0'

    if dbp:
        dbp = dbp
    else:
        dbp = '0'
    glu = request.form.get('glu')
    BE5_1 = int(request.form.get('BE5_1'))
    HE_HPfh1 = int(request.form.get('HE_HPfh1'))
    HE_HPfh2 = int(request.form.get('HE_HPfh2'))
    HE_HPfh3 = int(request.form.get('HE_HPfh3'))
    pa_aerobic = int(request.form.get('pa_aerobic'))


    result_data = {
        'sex': sex,
        'age': age,
        'HE_ht': height,
        'HE_wt': weight,
        'HE_BMI': bmi,
        'HE_sbp': sbp,
        'HE_dbp': dbp,
        'HE_glu': glu,
        'BE5_1': BE5_1,
        'HE_HPfh1': HE_HPfh1,
        'HE_HPfh2': HE_HPfh2,
        'HE_HPfh3': HE_HPfh3,
        'pa_aerobic': pa_aerobic
    }

    df = pd.DataFrame(result_data, index = [0])

    if df['HE_sbp'].values[0] == '0':
        df = df.loc[:, ~df.columns.isin(['HE_sbp','HE_dbp'])]
         
    else:
        df = df
        
    user_htn = User_HTN(df).user_type()

    if user_htn['고혈압분류'] == '정상':
        prompt = '고혈압을 예방하기 위한 식습관과 운동습관 한가지씩 알려줘'

    else:
        prompt = '고혈압을 가지고 있는 경우 치료하기 위해 식습관과 운동습관 한가지씩 알려줘'
    openai.api_key = args.apikey
    
    def chating(text):
        completion = openai.ChatCompletion.create(
            model = 'gpt-3.5-turbo',
            messages = [
                {'role':'user','content': '고혈압을 치료하려면 어떻게 해야 하는지 운동습관과 식습관 각각 두가지씩 알려줘'},
                {'role':'assistant','content':'운동습관<br> 일주일에 적어도 3번 이상 유산소 및 근력운동을 해야합니다.<br>식습관<br>콜레스테롤이 낮은 음식을 드셔야 합니다.'},
                {'role':'user','content': '고혈압을 예방하려면 어떻게 해야 하는지 운동습관과 식습관 각각 두가지씩 알려줘'},
                {'role':'assistant','content':'운동습관<br> 심박수가 120이상이 되도록 유산소 운동을 해야합니다.<br>식습관<br>나트륨이 적게 함유된 음식을 드셔야 합니다.'},                
                {'role':'user', 'content':str(text)}
                ]) 

        chatbot_response = completion.choices[0].message.content
        text1 = re.sub(r'(식습관:?)', '<strong>식습관</strong><br>', chatbot_response)
        text2 = re.sub(r'(운동습관:?)', '<br><br><strong>운동습관</strong><br>', text1)
        return text2
    
    answer = chating(prompt)
    # 결과 테이블을 렌더링할 result.html 파일 호출
    return render_template('predict_result.html', result_data=user_htn, answer = answer)
    # return render_template('predict_result.html', result_data=user_htn)

if __name__ == '__main__':
    app.run()
