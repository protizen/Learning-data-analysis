# Learning-data-analysis
영화리뷰 데이터에 대한 감성분석을 위한 토크나이저 활용 및 모델생성  

## 실제 구현된 사이트
http://210.109.83.44:5000 (실시간 번역으로 초기 로드시 시간이 걸림)

## 목적
NSMC(ratings_train.txt, ratings_test.txt)를 이용해 한국어 텍스트 감성(긍정/부정) 분류 모델을 학습하고 저장하고 실제 영화리뷰 서비스 API르 활용하여 감성을 예측하는 서비스까지 구현

## 적용 시스템
- OS: Ubuntu 22.04 LTS (카카오클라우드)
- Python 패키지: pandas, scikit-learn, konlpy, jpype1, joblib, numpy
- NSMC 데이터 위치: `/workspaces/Learning-data-analysis/data/ratings_train.txt`, `/workspaces/Learning-data-analysis/data/ratings_test.txt`

## 단계 요약
1. 데이터 로드
   - pandas로 CSV(ratings_*.txt)를 읽음(헤더 없음, 구분자 탭).
2. 전처리
   - `document` 열의 null 제거.
   - 한글(가-힣)과 공백 외 문자를 공백으로 치환.
   - 테스트 데이터에서 빈 문자열 행 제거.
3. 토크나이저 설정
   - jpype로 JVM을 명시적으로 시작(예: `jpype.getDefaultJVMPath()` 사용, `-Dfile.encoding=UTF-8` 추가).
   - Konlpy의 `Okt()` 초기화.
   - `okt.morphs`를 이용한 `okt_tokenizer(text)` 정의.
4. 특징 벡터화
   - `TfidfVectorizer(tokenizer=okt_tokenizer, token_pattern=None)` 생성.
   - 학습 데이터로 `fit`, 학습/테스트 텍스트를 `transform`.
5. 분류기 학습
   - `LogisticRegression(max_iter=2000, random_state=42)` 생성 및 학습 준비.
6. 하이퍼파라미터 탐색
   - `GridSearchCV`로 `C` 후보 `[1, 3, 3.5, 4, 4.5, 5]`, `cv=3`로 최적 모델 탐색.
   - `best_estimator_`를 저장(타깃 정확도 예: 86.2%).
7. 모델 저장
   - `joblib.dump`로 `tfidf.pkl`, `SA_lr_best.pkl` 저장(스크립트 경로에 저장).
8. 평가 및 예측
   - 테스트 데이터에 대해 예측 후 `accuracy` 출력.
   - 사용자 입력 문장 전처리(한글 추출, 공백 정리) → TF-IDF 변환 → 최적 모델로 예측(결과: "긍정"/"부정").

## 실행 예시
1. 데이터 파일을 `.../data/`에 배치.
2. 스크립트 실행:
   ```
   python /workspaces/Learning-data-analysis/create_model.py
   ```
3. 실행 후 `tfidf.pkl`, `SA_lr_best.pkl` 파일이 생성됨.

## 웹에서 영화리뷰 API의 리뷰 내용을 위에서 생성한 모델로 추론 서비스 제작

- Flask 기반 웹 애플리케이션으로 영화 리뷰를 불러와 감성분석 결과를 함께 보여줍니다.
- TMDB API에서 영화 리뷰를 조회하며, 환경변수 TMDB_API_KEY를 사용하고 기본값을 제공합니다.
- 불러온 리뷰는 Google Translate(공개 엔드포인트)를 이용해 한국어로 번역하고, 번역 실패 시 원문을 사용합니다.
- 미리 학습된 TF-IDF 벡터화기(tfidf.pkl)와 로지스틱 회귀 모델(SA_lr_best.pkl)을 joblib로 로드합니다.
- 토크나이저는 Konlpy의 Okt를 동일한 jvm 경로로 초기화하여 학습 시 사용한 함수(okt_tokenizer)를 재사용합니다.
- 입력 텍스트는 한글/공백만 남기는 전처리 후 TF-IDF 변환을 거쳐 긍정/부정으로 분류합니다.
- 루트 엔드포인트('/')는 쿼리 파라미터 limit으로 반환할 리뷰 수를 제어하며 기본값은 5입니다.
- 실행: 모델 파일과 의존 패키지(Flask, requests, jdk, konlpy, joblib, scikit-learn 등)가 준비된 환경에서 python app.py로 실행합니다.

<img width="858" height="604" alt="image" src="https://github.com/user-attachments/assets/3aaf13ba-e18f-49b5-b874-dafa0ddd77ae" />

