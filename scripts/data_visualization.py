import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from collections import Counter
import pandas as pd
def plot_sentiment_distribution(df):
    sentiment_counts = df['category'].value_counts()
    labels = sentiment_counts.index
    plt.figure(figsize=(6,6))
    plt.pie(sentiment_counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=['red','yellow','green'])
    plt.title("Sentiment Distribution")
    plt.show()

def plot_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(text))
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=20)
    plt.show()
    
def plot_most_frequent_words(df, column='clean_text', top_n=20):
    # Combine all text
    combined_text = ' '.join(df[column].dropna().tolist()).lower()
    
    # Tokenize into words
    words = combined_text.split()
    
    # Count frequencies
    word_freq = Counter(words)
    
    # Get the most common words
    common_words = word_freq.most_common(top_n)
    
    # Create a dataframe for easy plotting
    freq_df = pd.DataFrame(common_words, columns=['word', 'count'])
    
    # Plot with hue
    plt.figure(figsize=(12, 6))
    sns.barplot(data=freq_df, x='count', y='word', hue='word', palette='viridis', dodge=False, legend=False)
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    plt.title(f'Top {top_n} Most Frequent Words')
    plt.show()

