from flask import Flask, request, jsonify
import logging

# ----------------------------
# Dummy placeholder classes
# (Replace with your actual implementations)
# ----------------------------

class SentimentAnalyzer:
    def __init__(self, model=None):
        self.model = model or "default-model"

    def analyze_sentiment(self, text):
        # Very simple sentiment mock
        text = text.lower()
        if "good" in text or "love" in text or "happy" in text:
            return "positive"
        elif "bad" in text or "hate" in text or "angry" in text:
            return "negative"
        else:
            return "neutral"


class Summarizer:
    def generate_summary(self, comments):
        if not comments:
            return "No comments provided"
        return " ".join(comments[:2]) + " ..."   # mock summary


class WordCloudGenerator:
    def create_wordcloud(self, comments):
        if not comments:
            return {}
        # Return dummy word frequencies
        word_freq = {}
        for comment in comments:
            for word in comment.split():
                word_freq[word] = word_freq.get(word, 0) + 1
        return word_freq


# ----------------------------
# Flask App
# ----------------------------

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Initialize analyzers
sentiment_analyzer = SentimentAnalyzer(model='path/to/your/model')
summarizer = Summarizer()
wordcloud_generator = WordCloudGenerator()


@app.route('/')
def home():
    return jsonify({"message": "Flask Sentiment Analysis API is running"})


@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    try:
        logging.debug("Received request for /analyze_sentiment")
        data = request.json
        logging.debug(f"Request data: {data}")
        if not data or 'comment' not in data:
            return jsonify({'error': 'Missing \"comment\" in request data'}), 400

        comment = data['comment']
        sentiment = sentiment_analyzer.analyze_sentiment(comment)
        logging.debug(f"Sentiment: {sentiment}")
        return jsonify({'sentiment': sentiment})
    except Exception as e:
        logging.error(f"Error in /analyze_sentiment: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/generate_summary', methods=['POST'])
def generate_summary():
    try:
        logging.debug("Received request for /generate_summary")
        data = request.json
        logging.debug(f"Request data: {data}")
        if not data or 'comments' not in data:
            return jsonify({'error': 'Missing \"comments\" in request data'}), 400

        comments = data['comments']
        summary = summarizer.generate_summary(comments)
        logging.debug(f"Summary: {summary}")
        return jsonify({'summary': summary})
    except Exception as e:
        logging.error(f"Error in /generate_summary: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        logging.debug("Received request for /generate_wordcloud")
        data = request.json
        logging.debug(f"Request data: {data}")
        if not data or 'comments' not in data:
            return jsonify({'error': 'Missing \"comments\" in request data'}), 400

        comments = data['comments']
        wordcloud = wordcloud_generator.create_wordcloud(comments)
        logging.debug("Wordcloud generated successfully")
        return jsonify({'wordcloud': wordcloud})
    except Exception as e:
        logging.error(f"Error in /generate_wordcloud: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Make sure to run inside "src" folder
    app.run(debug=True)
