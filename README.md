# 고혈압 자가진단 API


## 프로젝트 개요
> 사용자의 건강정보를 토대로 고혈압 자가진단을 할 수 있는 API 개발<br>
> **개발기간 2023.05 ~ 2023.06.15**


## API 기능 소개
> - 사용자가 건강정보를 전부 입력할 경우 사용자의 고혈압 유형 분류 및 건강관리법 제공
> - 사용자가 수축기,이완기 혈압을 모를 경우 사용자의 예측 수축기,이완기 제공 및 사용자의 고혈압 유형 분류, 건강관리법 제공
> 
## Stacks 🐈


### 분류, 회귀 딥러닝 모델
<img src="https://img.shields.io/badge/pytorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">

### API
<img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=Flask&logoColor=white">


## 사용자 건강정보 입력 페이지
|||
| :-------------------------------------------: | :------------: |
|  <img width="696" alt="스크린샷 2023-06-15 오후 3 29 39" src="https://github.com/in-sukim/Hypertension/assets/43094223/0820da59-5f8b-4ee2-95c3-1c633ddae32b" width="200" height="300">|  <img width="687" alt="스크린샷 2023-06-15 오후 3 29 48" src="https://github.com/in-sukim/Hypertension/assets/43094223/8db837d8-1dcc-44f0-bfb7-9bd36fc9002f" width="200" height="300">|  
| <img width="612" alt="스크린샷 2023-06-15 오후 3 55 45" src="https://github.com/in-sukim/Hypertension/assets/43094223/7dc1a3f7-1aa9-4599-aa19-ef5dc571d983" width="200" height="300">|  <img width="686" alt="스크린샷 2023-06-15 오후 3 30 38" src="https://github.com/in-sukim/Hypertension/assets/43094223/4237af38-f944-4ba9-afce-f1a1fb8ba78c" width="200" height="300">|
|| |
---

## API 결과 화면
| 모든 건강정보 아는 경우|  수축기,이완기 혈압을 모를 경우 |
| :-------------------------------------------: | :------------: |
|  <img width="838" alt="스크린샷 2023-06-15 오후 4 05 19" src="https://github.com/in-sukim/Hypertension/assets/43094223/39ba39a6-c8bc-44f5-804c-80fca716c3a3" width="300" height="400">|  <img width="834" alt="스크린샷 2023-06-15 오후 4 08 58" src="https://github.com/in-sukim/Hypertension/assets/43094223/b04ce631-6814-4a53-8787-e12fcbb4a063" width="200" height="200">|
|  |  <img width="827" alt="스크린샷 2023-06-15 오후 4 09 07" src="https://github.com/in-sukim/Hypertension/assets/43094223/fe65e7b1-c617-46e3-9824-452dff01d295" width="200" height="400">|

---
## 디렉토리 구조

```bash
├── requirements.txt
├── README.md
├── app.py
├── modules : 
│   ├── htn : 고혈압 분류
│        ├── htn_model.pth: 저장된 고혈압 유형 분류 모델
│        ├── htn_model.py: 고혈압 유형 분류 모델
│        ├── htn_predict.py: 입력받은 사용자정보를 통해 고혈압 유형 분류
│        ├── hypertension_model_config.json: 고혈압 유형 분류모델 parameter json
│   ├── sbp : 수축기혈압
│        ├── sbp_model.pth: 저장된 수축기혈압 예측 모델
│        ├── sbp_model.py: 수축기혈압 예측 모델
│        ├── sbp_predict.py: 입력받은 사용자정보를 통해 수축기혈압 예측
│        ├── sbp_model_config.json: 수축기혈압 예측 모델 parameter json
│   ├── dbp: 이완기혈압
│        ├── dbp_model.pth: 저장된 이완기혈압 예측 모델
│        ├── dbp_model.py: 이완기혈압 예측 모델
│        ├── dbp_predict.py: 입력받은 사용자정보를 통해 이완기혈압 예측
│        ├── dbp_model_config.json:: 이완기혈압 예측 모델 parameter json
│   ├── seed_everything.py: For Reproduction
│   ├── user_htn.py: 입력된 사용자의 정보에 따라 고혈압 유형 분류 또는 예측 수축기,이완기혈압과 고혈압 유형 분류
│   ├── templates: Flask API HTML
│        ├── form.html: 사용자의 건강정보 입력 페이지
│        ├── predict_result.html: 입력된 건강정보를 통해 나온 결과 및 건강관리법 제공 페이지
---
```
## 유의사항
### OpenAI API Key 발급 필요

## 보완점
### 사용자가 영화를 선택할 때 고려하는 사항에 대한 정보를 제공할 수 있는 기능 부족
>   - 좋아하는 영화감독이나 배우가 출현하는 영화에 대한 정보 노출 기능<br>
>   - 다음영화 사이트에 스크래핑 과정 최적화를 통한 소요시간 단축
>   
### 긍/부정 모델 성능 개선
>   - 분류 정확도 향상
>   - 모델을 경량화하여 추론 시간 단축
>   - 모델 성능 향상 및 긴 길이의 리뷰 처리 가능한 모델로 교체(RoBERTa)
>   - 영화와 관련 없는 리뷰 필터링(정치, 광고 등에 관한 리뷰)
