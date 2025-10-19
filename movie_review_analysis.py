from flask import Flask, render_template, request
import requests
from datetime import datetime
import os
import joblib
import re
from konlpy.tag import Okt

# Okt 초기화
okt = Okt()

# 토크나이저 함수 - 모델 생성 시 사용된 것과 동일한 함수
def okt_tokenizer(text):
    tokens = okt.morphs(text)
    return tokens

def preprocess_text(text):
    return re.sub(r'[^ ㄱ-ㅣ가-힣]+', ' ', text)

# 감성 분석 수행 함수
def analyze_sentiment(review, tfidf, model):
    processed_review = preprocess_text(review)
    review_tfidf = tfidf.transform([processed_review])
    prediction = model.predict(review_tfidf)[0]
    return "긍정" if prediction == 1 else "부정"

tfidf = joblib.load('tfidf.pkl')
model = joblib.load('SA_lr_best.pkl')

app = Flask(__name__)

def get_movie_reviews(movie_id="76600", limit=None):
    API_KEY = os.getenv("TMDB_API_KEY", "API_KEY")
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/reviews"
    params = {"api_key": API_KEY}
    headers = {"accept": "application/json"}
    
    try:
        res = requests.get(url, params=params, headers=headers, timeout=10)
        res.raise_for_status()
        results = res.json().get('results', [])
        if limit is not None:
            try:
                n = int(limit)
                if n >= 0:
                    results = results[:n]
            except Exception:
                pass
        return results
    except requests.exceptions.HTTPError:
        try:
            print(f"HTTP 에러 ({res.status_code}): {res.text}")
        except Exception:
            print("HTTP 에러 발생, 응답을 읽을 수 없습니다.")
        return []
    except Exception as e:
        print(f"에러 발생: {e}")
        return []

# 텍스트 목록을 받아 한국어로 번역해서 리스트로 반환
def translate_texts(texts, target_lang='ko'):
    translated = []
    for text in texts:
        if not text:
            translated.append(text)
            continue
        try:
            params = {
                "client": "gtx",
                "sl": "auto",
                "tl": target_lang,
                "dt": "t",
                "q": text
            }
            r = requests.get("https://translate.googleapis.com/translate_a/single", params=params, timeout=5)
            r.raise_for_status()
            data = r.json()
            translated_text = ''.join([seg[0] for seg in data[0]]) if data and data[0] else text
            translated.append(translated_text)
        except Exception:
            translated.append(text)
    return translated

@app.route('/')
def index():
    # 기본 5개 리뷰만 진행
    limit_param = request.args.get('limit', '5')
    reviews = get_movie_reviews(limit=limit_param)
    # 리뷰 내용 추출해서 번역
    contents = [r.get('content', '') for r in reviews]
    translated_contents = translate_texts(contents, 'ko')
    
    for r, t in zip(reviews, translated_contents):
        sentiment = analyze_sentiment(t, tfidf, model)
        r['content_ko'] = t + f" ({sentiment} 감성)"
    print(reviews)
    return render_template('reviews.html', reviews=reviews)

if __name__ == '__main__':
    app.run(debug=True)

