import streamlit as st
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

# Page setup
st.set_page_config(page_title="Sentiment Analysis App", layout="centered")
st.title("💬 Sentiment Analysis of Comments")
st.write("Type a comment below OR upload a file to analyze its sentiment with a word cloud.")

# Text input
comment = st.text_area("📝 Enter your comment here:")

# File upload
uploaded_file = st.file_uploader("📂 Or upload a file (.txt or .csv)", type=["txt", "csv"])

# Function to extract text
def extract_text(file):
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    elif file.name.endswith(".csv"):
        df = pd.read_csv(file)
        # join all text from first column (you can change column index if needed)
        return " ".join(df.iloc[:,0].astype(str))
    return ""

if st.button("🔍 Analyze Sentiment"):
    text_to_analyze = ""

    if uploaded_file is not None:
        text_to_analyze = extract_text(uploaded_file)
    elif comment.strip() != "":
        text_to_analyze = comment
    else:
        st.warning("⚠️ Please enter a comment or upload a file to analyze.")

    if text_to_analyze.strip() != "":
        # Sentiment Analysis
        blob = TextBlob(text_to_analyze)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        # Sentiment Category
        if polarity > 0:
            sentiment = "😊 Positive"
        elif polarity < 0:
            sentiment = "😡 Negative"
        else:
            sentiment = "😐 Neutral"

        # Display Results
        st.subheader("📊 Sentiment Results")
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Polarity Score:** {polarity:.2f}")
        st.write(f"**Subjectivity Score:** {subjectivity:.2f}")

        # WordCloud
        st.subheader("☁️ Word Cloud")
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_to_analyze)

        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
