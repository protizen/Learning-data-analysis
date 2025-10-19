# Learning-data-analysis
영화리뷰 데이터에 대한 감성분석을 위한 토크나이저 활용 및 모델생성

## 목적
NSMC(ratings_train.txt, ratings_test.txt)를 이용해 한국어 텍스트 감성(긍정/부정) 분류 모델을 학습하고 저장 및 예측하는 전체 파이프라인 정리.

## 전제조건
- OS: Ubuntu 24.04.2 LTS (dev container)
- Python 패키지: pandas, scikit-learn, konlpy, jpype1, joblib, numpy
- NSMC 데이터 위치: `/workspaces/Learning-data-analysis/data/ratings_train.txt`, `/workspaces/Learning-data-analysis/data/ratings_test.txt`

## 단계 요약
1. 데이터 로드
   - pandas로 TSV(ratings_*.txt)를 읽음(헤더 없음, 구분자 탭).
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
   python /workspaces/Learning-data-analysis/train_sentiment.py
   ```
3. 실행 후 `tfidf.pkl`, `SA_lr_best.pkl` 파일이 생성됨.

## 주의사항
- Konlpy 사용 시 JVM 초기화 문제가 발생할 수 있으므로 `jpype.startJVM()`를 명시적으로 호출하세요.
- `TfidfVectorizer`에 사용자 토크나이저를 쓸 때는 `token_pattern=None`을 지정해야 합니다.
- 데이터 파일 인코딩이나 파일 경로가 다르면 `FileNotFoundError`가 발생합니다.
