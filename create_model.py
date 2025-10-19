import numpy as np
import pandas as pd
import re

## 학습용 리뷰 데이터셋
nsmc_train_df = pd.read_csv('https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt', sep='\t', encoding='utf8', engine='python')

## 테스트용 리뷰 데이터셋
nsmc_test_df = pd.read_csv('https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt', sep='\t', encoding='utf8', engine='python')

## 학습용 데이터 전처리
nsmc_train_df = nsmc_train_df[nsmc_train_df['document'].notnull()]
nsmc_train_df['document'] = nsmc_train_df['document'].apply(lambda x: re.sub(r'[^ ㄱ-ㅣ가-힣]+', ' ', x))

## 테스트용 데이터 전처리
nsmc_test_df = nsmc_test_df[nsmc_test_df['document'].notnull()]
nsmc_test_df['document'] = nsmc_test_df['document'].apply(lambda x: re.sub(r'[^ ㄱ-ㅣ가-힣]+', ' ', x))
nsmc_test_df = nsmc_test_df[nsmc_test_df['document'].str.strip() != ""]

# 모델 만들기
from konlpy.tag import Okt
#okt = Okt()
okt = Okt(jvmpath="C:\\Program Files\\Microsoft\\jdk-11.0.28.6-hotspot\\bin\\server\\jvm.dll")

# 토큰화 함수 정의
def okt_tokenizer(text):
    tokens = okt.morphs(text)
    return tokens

# TF-IDF 기반 피처 벡터 생성
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(tokenizer=okt_tokenizer, ngram_range=(1, 2), min_df=3, max_df=0.9)
tfidf.fit(nsmc_train_df['document'])
nsmc_train_tfidf = tfidf.transform(nsmc_train_df['document'])

# 로지스틱 회귀 기반 분석모델 생성
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=0)

# 로지스틱 회귀 모델 학습
model.fit(nsmc_train_tfidf, nsmc_train_df['label'])

# GridSearchCV를 이용한 best 파라미터 탐색
from sklearn.model_selection import GridSearchCV
params = {'C': [1, 3, 3.5, 4, 4.5, 5]}
SA_lr_grid_cv = GridSearchCV(model, param_grid=params, cv=3, scoring='accuracy', verbose=1)
SA_lr_grid_cv.fit(nsmc_train_tfidf, nsmc_train_df['label'])

# 최적 파라미터의 best 모델 저장
SA_lr_best = SA_lr_grid_cv.best_estimator_

import joblib
joblib.dump(tfidf, 'tfidf.pkl')
joblib.dump(SA_lr_best, 'SA_lr_best.pkl')

# 모델 평가
nsmc_test_tfidf = tfidf.transform(nsmc_test_df['document'])
# nsmc_test_tfidf 벡터를 사용하여 감성을 예측(predict())
test_predict = SA_lr_best.predict(nsmc_test_tfidf)
from sklearn.metrics import accuracy_score
print('감성 분석 정확도 : ', round(accuracy_score(nsmc_test_df['label'], test_predict), 3))

# 새로운 리뷰 데이터에 대한 감성 예측
st = input("문장입력 >> ")

# 입력 텍스트에 대한 전처리 수행
st2 = re.compile(r'[ㄱ-ㅣ가-힣]+').findall(st)
st3 = [" ".join(st2)]

# 입력 텍스트의 피처 벡터화
st_tfidf = tfidf.transform(st3)

# 최적 감성분석 모델에 적용하여 감성 분석 평가
st_predict = SA_lr_best.predict(st_tfidf)[0]

# 예측 값 출력하기
if(st_predict == 0):
    print(st, '==> 부정 감성')
else:
    print(st, '==> 긍정 감성')
