!pip install streamlit
import streamlit as st
import numpy as np, joblib, json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sentence_transformers import SentenceTransformer

# ---------- Load models ----------
@st.cache_resource
def load_all():
    lgb_clf = joblib.load("hybrid_lightgbm_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("config.json") as f:
        cfg = json.load(f)
    sbert = SentenceTransformer(cfg["sbert_model_name"])
    return lgb_clf, tfidf, scaler, sbert, cfg

lgb_clf, tfidf, scaler, sbert, cfg = load_all()
vader = SentimentIntensityAnalyzer()

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Sentiment Analysis App", page_icon="💬")
st.title("💬 Sentiment Analysis with Hybrid ML Model")
st.write("Analyze text sentiment using TF-IDF + SBERT + Lexicon hybrid model.")

user_text = st.text_area("Enter text to analyze:", height=160)

if st.button("🔍 Analyze Sentiment"):
    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        # --- Lexicon features ---
        vader_score = vader.polarity_scores(user_text)["compound"]
        tb_score = TextBlob(user_text).sentiment.polarity
        lex_features = np.array([[vader_score, tb_score]])

        # --- TF-IDF ---
        tfidf_features = tfidf.transform([user_text]).toarray()

        # --- SBERT ---
        sbert_features = sbert.encode([user_text])

        # --- Combine and scale ---
        X_input = np.hstack([tfidf_features, sbert_features, lex_features])
        X_input = scaler.transform(X_input)

        # --- Predict ---
        pred = lgb_clf.predict(X_input)[0]
        sentiment = "😊 Positive" if pred == 1 else "☹️ Negative"

        st.markdown(f"### Predicted Sentiment: **{sentiment}**")
        st.write(f"**VADER compound:** {vader_score:.3f}")
        st.write(f"**TextBlob polarity:** {tb_score:.3f}")
