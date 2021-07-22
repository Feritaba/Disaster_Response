import sys
import nltk
nltk.download(['punkt', 'wordnet'])
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import sqlalchemy
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import TruncatedSVD
# from sklearn.preprocessing import FunctionTransformer
# from gensim.models import Word2Vec
# from sklearn.ensemble import ExtraTreesClassifier
# from gensim.test.utils import common_texts
# from sklearn.datasets import make_multilabel_classification
# from sklearn.metrics import confusion_matrix
# from sklearn.base import BaseEstimator, TransformerMixin


def load_data(database_filepath):
    engine = sqlalchemy.create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('disaster_msgs', engine)
    X = df.message.values
    Y = df[df.columns[4:]].values
    category_names = df.columns[4:]

    print(df.columns)
    return X, Y, category_names


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    neigh = KNeighborsClassifier(n_neighbors=3)

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('svd', TruncatedSVD(n_components=20, n_iter=100, random_state=42)),
        ('clf', MultiOutputClassifier(neigh))
    ])
    
    # parameters = {
        # 'estimator__clf__estimator__n_estimators': [100, 200],
        # 'vect__ngram_range': ((1, 1),(1,2),(1,3),(1,4))
        # 'clf__estimator__learning_rate': [0.1, 0.3],
        # 'clf__estimator__bootstrap': [True, False],
        # 'vect__ngram_range': ((1, 1),(1,2))
    # }

    # cv = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=2, n_jobs=-1, verbose=3)
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    # target_names = df.columns[4:]
    y_pred = model.predict(X_test)
    for i in range(len(Y_test[0])):
        y_pred1  =y_pred[:,i]
        y_true = Y_test[:,i]
        
        print(category_names[i].upper())

        print(classification_report(y_true, y_pred1))


def save_model(model, model_filepath):
    model_pickle = open(model_filepath,'wb')
    pickle.dump(model, model_pickle)
    model_pickle.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
