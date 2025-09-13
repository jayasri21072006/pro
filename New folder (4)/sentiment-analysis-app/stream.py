import streamlit as st
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import re

# -------------------- Helper Functions --------------------
def extract_comments_from_file(uploaded_file):
    """Extract comments from a .txt or .csv file."""
    comments = []
    if uploaded_file.name.endswith(".txt"):
        content = uploaded_file.read().decode("utf-8")
        comments = content.splitlines()
    elif uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        if "comment" in df.columns:
            comments = df["comment"].dropna().tolist()
    return comments

def clean_text(text):
    """Basic text cleaning."""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
    return text

# -------------------- Sentiment Analyzer --------------------
class SentimentAnalyzer:
    def __init__(self, model="distilbert-base-uncased-finetuned-sst-2-english"):
        # Force use of PyTorch on CPU (device=-1)
        self.analyzer = pipeline("sentiment-analysis", model=model, device=-1)

    def analyze_sentiment(self, text):
        res = self.analyzer(text)[0]
        return {"label": res["label"], "score": res["score"]}

    def get_overall_sentiment(self, texts):
        scores = []
        labels = []
        for text in texts:
            res = self.analyze_sentiment(text)
            scores.append(res["score"])
            labels.append(res["label"])
        overall_label = max(set(labels), key=labels.count)
        average_score = sum(scores) / len(scores)
        return {"label": overall_label, "average_score": average_score}

# -------------------- Summarizer --------------------
class Summarizer:
    def generate_summary(self, texts):
        # simple placeholder: first 2 sentences of combined text
        combined = " ".join(texts)
        sentences = combined.split(".")
        return ". ".join(sentences[:2]) + (". ..." if len(sentences) > 2 else "")

# -------------------- Streamlit Page --------------------
st.set_page_config(page_title="Sentiment Analysis App", layout="centered")
st.title("💬 Sentiment Analysis of Comments")
st.write(
    "Type a comment below OR upload a file to analyze its sentiment, generate a summary, and create a word cloud."
)

comment_input = st.text_area("📝 Enter your comment here:")
uploaded_file = st.file_uploader("📂 Or upload a file (.txt or .csv)", type=["txt", "csv"])

if st.button("🔍 Analyze"):
    comments = []

    # Extract comments from file or text area
    if uploaded_file:
        comments = extract_comments_from_file(uploaded_file)
    elif comment_input.strip():
        comments = [comment_input.strip()]
    else:
        st.warning("⚠️ Please enter a comment or upload a file.")

    if comments:
        cleaned_comments = [clean_text(c) for c in comments]

        # -------------------- Sentiment Analysis --------------------
        sentiment_analyzer = SentimentAnalyzer()
        individual_sentiments = [sentiment_analyzer.analyze_sentiment(c) for c in cleaned_comments]
        overall_sentiment = sentiment_analyzer.get_overall_sentiment(cleaned_comments)

        st.subheader("📊 Sentiment Analysis")
        st.write(f"**Overall Sentiment:** {overall_sentiment['label']}")
        st.write(f"**Average Score:** {overall_sentiment['average_score']:.2f}")

        st.write("**Individual Comment Sentiments:**")
        for idx, result in enumerate(individual_sentiments, start=1):
            st.write(f"{idx}. {result['label']} (Score: {result['score']:.2f})")

        # -------------------- Summary --------------------
        summarizer = Summarizer()
        summary = summarizer.generate_summary(cleaned_comments)
        st.subheader("📝 Summary of Comments")
        st.write(summary)

        # -------------------- Word Cloud --------------------
        text_for_wc = " ".join(cleaned_comments)
        if text_for_wc.strip():
            wc = WordCloud(width=800, height=400, background_color="white").generate(text_for_wc)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.subheader("☁️ Word Cloud")
            st.pyplot(fig)
        else:
            st.write("No text available to generate a word cloud.")
