import gensim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

e2v = gensim.models.KeyedVectors.load_word2vec_format("resources/emojional.bin", binary=True)

def preprocess_text(text, opt = 'Raw'):
    """
    Preprocess the text by removing special characters and converting to lowercase.
    """
    if opt == 'Raw':
        return text.str.lower().replace(r'[^a-zA-Z0-9\s]', ' ', regex=True).str.strip().str.split().apply(lambda tokens: ' '.join(tokens))
    elif opt == 'Lemmatize':
        pass
    elif opt == 'Stem':
        pass
    else:
        raise ValueError("Invalid option for preprocessing.")

def tokenize(text, opt = 'text'):
    if opt == 'text':
        pass
    elif opt == 'emoji':
        pass
    else:
        raise ValueError("Invalid option for tokenization.")

def train(model, X_train, y_train):
    """
    Train the model using the training data.
    """
    model.fit(X_train, y_train)
    return model

def main():
    # load the dataset
    ELCo_df = pd.read_csv('data/ELCo.csv')
    ELCo_df = ELCo_df.drop(columns=['Description'])

    # preprocess the dataset
    ELCo_df['EN'] = preprocess_text(ELCo_df['EN'])

    # map the 'Composition strategy' column to numerical values
    composition_strategy_mapping = {name: idx for idx, name in enumerate(ELCo_df['Composition strategy'].unique())}
    ELCo_df['Composition strategy'] = ELCo_df['Composition strategy'].map(composition_strategy_mapping)

    # split the dataset into train, validate and test sets
    train_df, test_df = train_test_split(ELCo_df, test_size=0.2, random_state=42, stratify=ELCo_df['Composition strategy'])

    # further split the test set into validate and test sets
    train_df, validate_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['Composition strategy'])
    X_train, y_train = train_df.drop(columns=['Composition strategy']), train_df['Composition strategy']
    X_validate, y_validate = validate_df.drop(columns=['Composition strategy']), validate_df['Composition strategy']
    X_test, y_test = test_df.drop(columns=['Composition strategy']), test_df['Composition strategy']

    # feature extraction
    X_EN = tokenize(X_train['EN'], opt = 'text')
    X_EM = tokenize(X_train['EM'], opt = 'emoji')
    X_train_vectorized = hstack([X_EN, X_EM])

    # define the model
    model = LogisticRegression(max_iter=1000, verbose=1)

    # train the model
    model = train(model, X_train_vectorized, y_train)
    
    # validate the model
    X_validate_EN = tokenize(X_validate['EN'], opt = 'text')
    X_validate_EM = tokenize(X_validate['EM'], opt = 'emoji')
    X_validate_vectorized = hstack([X_validate_EN, X_validate_EM])
    y_validate_pred = model.predict(X_validate_vectorized)
    validate_accuracy = np.mean(y_validate_pred == y_validate)
    print(f"Validation Accuracy: {validate_accuracy:.4f}")
    
    # test the model
    X_test_EN = tokenize(X_test['EN'], opt = 'text')
    X_test_EM = tokenize(X_test['EM'], opt = 'emoji')
    X_test_vectorized = hstack([X_test_EN, X_test_EM])
    y_test_pred = model.predict(X_test_vectorized)
    test_accuracy = np.mean(y_test_pred == y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")