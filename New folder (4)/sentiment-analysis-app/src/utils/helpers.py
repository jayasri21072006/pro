def clean_text(text):
    import re
    # Remove special characters and digits
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_text(text):
    return text.split()

def remove_stopwords(tokens):
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]