import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv('Tweets.csv')
df.head()

# Convert text to lowercase
df['text'] = df['text'].str.lower()

df['text'] = df['text'].astype(str)  # Convert 'text' column to string data type

df['tokens'] = df['text'].apply(nltk.word_tokenize)  # Tokenization

# Remove stopwords
stopwords = nltk.corpus.stopwords.words('english')
df['tokens'] = df['tokens'].apply(lambda x: [word for word in x if word not in stopwords])