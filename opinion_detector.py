import pandas as pd
from nltk import regexp_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report


dataset = 'SAR14.txt'


# ---------------------------------------- 1. Load the corpus

def load_corpus(filename: str) -> pd.DataFrame:
    corpus_table = pd.read_csv(filename, names=['Review', 'Score'])
    return corpus_table


# ---------------------------------------- 2. Preprocess the data

def add_lemmas(corpus: pd.DataFrame) -> pd.Series:
    tokens = corpus['Review'].apply(tokenize)
    lemmas = tokens.apply(lemmatize)
    lemmas = lemmas.apply(remove_stopwords)
    return lemmas


def tokenize(review: str) -> list:
    pattern = '\w+'
    return regexp_tokenize(review, pattern)


def lemmatize(tokens: list) -> list:
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token.lower()) for token in tokens]


def remove_stopwords(lemmas: list) -> list:
    stop_words = stopwords.words('english')
    return [lemma for lemma in lemmas if lemma not in stop_words]


# ---------------------------------------- 3. Machine-learning preparations

def add_sentiment(corpus: pd.DataFrame) -> pd.Series:
    return corpus['Score'].map(lambda score: 'Positive' if 7 <= score <= 10 else 'Negative')


def vectorize_reviews(lemmas: pd.Series) -> pd.array:
    vectorizer = TfidfVectorizer()
    data = lemmas.map(lambda review: " ".join(review))
    tfidf_matrix = vectorizer.fit_transform(data)
    return tfidf_matrix


# ---------------------------------------- 4. Predict with SDGClassifier

def train_model(x, y) -> SGDClassifier:
    model = SGDClassifier()
    model.fit(x, y)
    return model


# ---------------------------------------- MAIN


def main():
    corpus = load_corpus(dataset)
    corpus['Lemmas'] = add_lemmas(corpus)
    corpus['Sentiment'] = add_sentiment(corpus)
    tfidf_matrix = vectorize_reviews(corpus['Lemmas'])

    x, y = tfidf_matrix, corpus['Sentiment']
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75)

    classifier = train_model(x_train, y_train)
    y_predicted = classifier.predict(x_test)
    print(classification_report(y_test, y_predicted))


if __name__ == '__main__':
    main()
