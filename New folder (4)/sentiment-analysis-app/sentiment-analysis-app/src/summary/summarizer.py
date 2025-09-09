class Summarizer:
    def generate_summary(self, comments):
        from collections import Counter
        import nltk
        from nltk.tokenize import sent_tokenize

        nltk.download('punkt')

        # Tokenize comments into sentences
        sentences = []
        for comment in comments:
            sentences.extend(sent_tokenize(comment))

        # Count frequency of each sentence
        sentence_frequency = Counter(sentences)

        # Generate a summary based on the most common sentences
        most_common_sentences = sentence_frequency.most_common(3)  # Adjust the number for summary length
        summary = ' '.join([sentence for sentence, freq in most_common_sentences])

        return summary.strip()