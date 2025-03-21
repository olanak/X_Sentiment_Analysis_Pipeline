import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Make sure to download necessary resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if pd.isnull(text):
        return ""
    
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    # Remove mentions and hashtags
    text = re.sub(r'\@\w+|\#','', text)
    # Remove punctuation, special characters, and numbers
    text = re.sub(r"[^a-zA-Z\s]", '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Tokenize and remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]
    # Lemmatize
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)

def preprocess_dataframe(df):
    # Work on a copy to avoid SettingWithCopyWarning
    df = df.copy()
    # Drop rows where clean_text is NaN or empty
    df = df.dropna(subset=['clean_text'])
    df = df[df['clean_text'].str.strip() != '']
    # Clean the text safely
    df.loc[:, 'clean_text'] = df['clean_text'].apply(lambda x: clean_text(str(x)))
    # Drop rows again if cleaning results in empty strings
    df = df[df['clean_text'].str.strip() != '']
    return df

