from flask import Flask, request, jsonify
from sentiment.analyzer import SentimentAnalyzer
from summary.summarizer import Summarizer
from wordcloud.generator import WordCloudGenerator
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Provide the required model argument
sentiment_analyzer = SentimentAnalyzer(model='path/to/your/model')
summarizer = Summarizer()
wordcloud_generator = WordCloudGenerator()

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    try:
        logging.debug("Received request for /analyze_sentiment")
        data = request.json
        logging.debug(f"Request data: {data}")
        if not data or 'comment' not in data:
            return jsonify({'error': 'Missing "comment" in request data'}), 400
        comment = data.get('comment')
        logging.debug(f"Comment: {comment}")
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
            return jsonify({'error': 'Missing "comments" in request data'}), 400
        comments = data.get('comments')
        logging.debug(f"Comments: {comments}")
        summary = summarizer.generate_summary(comments)
        logging.debug(f"Summary: {summary}")
        return jsonify({'summary': summary})
    except Exception as e:
        logging.error(f"Error in /generate_summary: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.json
        logging.debug(f"Request data: {data}")
        if not data or 'comments' not in data:
            return jsonify({'error': 'Missing "comments" in request data'}), 400
        comments = data.get('comments')
        logging.debug(f"Comments: {comments}")
        wordcloud = wordcloud_generator.create_wordcloud(comments)
        logging.debug("Wordcloud generated successfully")
        return jsonify({'wordcloud': wordcloud})
    except Exception as e:
        logging.error(f"Error in /generate_wordcloud: {str(e)}")
        return jsonify({'error': str(e)}), 500