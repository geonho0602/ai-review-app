import streamlit as st
import joblib

# 모델 불러오기
@st.cache_resource
def load_model():
    return joblib.load('review_model.pkl')

model = load_model()

# UI
st.title("🤖 AI vs 사람 리뷰 판별기")
st.markdown("아래에 리뷰를 입력하면 AI가 쓴 건지 사람이 쓴 건지 알려드릴게요.")

text = st.text_area("✏️ 리뷰를 입력하세요", height=150)

if st.button("예측하기"):
    if text.strip() == "":
        st.warning("리뷰를 입력해주세요.")
    else:
        pred = model.predict([text])[0]
        proba = max(model.predict_proba([text])[0])
        st.success(f"예측 결과: **{pred}** (확신도: {proba:.4f})")
