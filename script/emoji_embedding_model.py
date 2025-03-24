import gensim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
from transformers import BertTokenizer, BertModel
import emoji

nltk.download('punkt')
nltk.download('wordnet')

e2v = gensim.models.KeyedVectors.load_word2vec_format("resources/emojional.bin", binary=True)

# Initialize the BERT tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Initialize the BERT model
bert_model = BertModel.from_pretrained('bert-base-uncased')

def preprocess_text(text, opt = 'Raw'):
    """
    Preprocess the text by removing special characters and converting to lowercase.
    """
    if opt == 'Raw':
        return text.str.lower().replace(r'[^a-zA-Z0-9\s]', ' ', regex=True).str.strip().str.split().apply(lambda tokens: ' '.join(tokens))
    elif opt == 'Lemmatize':
        lemmatizer = WordNetLemmatizer()
        return text.str.lower().replace(r'[^a-zA-Z0-9\s]', ' ', regex=True).str.strip().apply(
            lambda sentence: ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(sentence)])
        )
    elif opt == 'Stem':
        stemmer = PorterStemmer()
        return text.str.lower().replace(r'[^a-zA-Z0-9\s]', ' ', regex=True).str.strip().apply(
            lambda sentence: ' '.join([stemmer.stem(word) for word in word_tokenize(sentence)])
        )
    else:
        raise ValueError("Invalid option for preprocessing.")

def tokenize(text, opt='text'):
    """
    Tokenize the input text or emoji based on the specified option.
    """
    if opt == 'text':
        def extract_text_embeddings(sentence):
            # Tokenize the sentence
            inputs = bert_tokenizer(sentence, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=512)
            
            # Pass through the BERT model
            outputs = bert_model(**inputs)
            
            # Extract the embeddings (last hidden state)
            embeddings = outputs.last_hidden_state  # Shape: (batch_size, sequence_length, hidden_size)
            
            # Average the embeddings across the sequence length to get a single vector
            return embeddings.mean(dim=1).squeeze().detach().numpy()
        
        return np.array(text.apply(extract_text_embeddings).tolist())
    elif opt == 'emoji':
        def extract_emoji_embeddings(sentence):
            emojis = [char for char in sentence if char in emoji.EMOJI_DATA]
            embeddings = [e2v[em] for em in emojis if em in e2v]
            
            # Average the embeddings if there are any, otherwise return a zero vector
            if embeddings:
                return np.mean(embeddings, axis=0)
            else:
                return np.zeros(e2v.vector_size)
        
        return np.array(text.apply(extract_emoji_embeddings).tolist())
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
    ELCo_df['EN'] = preprocess_text(ELCo_df['EN'], opt='Raw')

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

    # Feature extraction
    X_EN = tokenize(X_train['EN'], opt='text')
    X_EM = tokenize(X_train['EM'], opt='emoji')

    # Combine text and emoji embeddings
    X_train_vectorized = np.concatenate([X_EN, X_EM], axis=1)

    # Define the model
    model = LogisticRegression(max_iter=1000, verbose=1)

    # Train the model
    model = train(model, X_train_vectorized, y_train)
    
    # Validate the model
    X_validate_EN = tokenize(X_validate['EN'], opt='text')
    X_validate_EM = tokenize(X_validate['EM'], opt='emoji')
    X_validate_vectorized = np.concatenate([X_validate_EN, X_validate_EM], axis=1)
    y_validate_pred = model.predict(X_validate_vectorized)
    validate_accuracy = np.mean(y_validate_pred == y_validate)
    print(f"Validation Accuracy: {validate_accuracy:.4f}")
    
    # Test the model
    X_test_EN = tokenize(X_test['EN'], opt='text')
    X_test_EM = tokenize(X_test['EM'], opt='emoji')
    X_test_vectorized = np.concatenate([X_test_EN, X_test_EM], axis=1)
    y_test_pred = model.predict(X_test_vectorized)
    test_accuracy = np.mean(y_test_pred == y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()