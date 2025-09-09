class WordCloudGenerator:
    def create_wordcloud(self, comments):
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        
        # Combine all comments into a single string
        text = ' '.join(comments)
        
        # Generate the word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        
        # Display the generated image
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')  # Do not show axes
        plt.show()