import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, Table, MetaData
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import pickle

import joblib

import time
from contextlib import contextmanager

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

example_db_filepath = '../data/DisasterResponse.db'

print(sys.version)
@contextmanager
def timer(process):
    """
    Timer function called to print the time any other process takes

    Args:
        process : str
            String to print with the output

    Returns:
        None
    """
    t0 = time.time()
    yield
    if int(time.time() - t0) < 60:
        print('{} executed in {} seconds\n'.format(
            process, int(time.time() - t0)))
    else:
        print('{} executed in {} minutes\n'.format(
            process, int(time.time() - t0) / 60))


def load_data(database_filepath, category_names):
    """
    Loads in DataFrame object from SQLite3 database
    and splits into X and y for modelling

    Args:
        database_filepath : string
            Relative location and name of .db file
        category_names : List
            List of table columns to filter y by

    Returns:
        X : DataFrame
            X DataFrame containing messages column
        y : DatFrame
            y DataFrame containing category classifcations
    """
    with timer('Load data'):
        engine = create_engine('sqlite:///{}'.format(database_filepath))
        connection = engine.connect()
        df = Table('labelled_messages', MetaData(),
                autoload=True, autoload_with=engine)
        df = pd.read_sql_table('labelled_messages', connection)

        y = df[category_names]
        X = df['message']
        return X, y


def tokenize(text):
    """
    Tokenizes a string of text by dumping punctuation,
    lemmatizing and stemming the string

    Args:
        text : str
            Text string to be tokenized

    Returns:
        clean_tokens : List
            List of tokenized string
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for token in tokens:
        clean_tokens.append(lemmatizer.lemmatize(token).lower().strip())
    clean_tokens = [w for w in clean_tokens if w not in stopwords.words('english')]

    return clean_tokens


def build_model():
    """
    Returns a PipeLine object of various model components

    Args:
        None

    Returns:
        pipeline : sklearn.Pipeline object
            Contains a vectorizer, transformer and classifier estimator
    """
    with timer('Build model'):
        pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(KNeighborsClassifier()))
        ])
        steps = '    Pipeline steps:'
        for index, step in enumerate(pipeline.steps):
            steps += '\n    Step {}: {}'.format(index+1, step[0])
        print(steps)
        return pipeline


def best_gridsearch_estimator(model, X_train, y_train, parameters):
    """
    Returns a PipeLine object with optimised parameters
    based on options in parameters arg and cross validation

    Args:
        model : sklearn model object
            Model to be cross validated
        X_train : DataFrame
            Training set
        y_train : DataFrame
            Expected results
        parameters : Dictionary
            values to test iteratively on

    Returns:
        best_estimator_ : GridSearchCV best_estimator_ object
            Model object containing best parameters from options given
    """
    with timer("Tune model hyperparameters"):
        para_combinations = 1
        for i in parameters.values():
            para_combinations = para_combinations*len(i)
        with timer("Gridsearch Fit on {} hyperparameter combinations"
                .format(str(para_combinations))):
            cv = GridSearchCV(model, scoring='neg_mean_squared_error',
                              param_grid=parameters, verbose=10, 
                              n_jobs=1, return_train_score=True)
            grid_result = cv.fit(X_train, y_train)
            return grid_result.best_estimator_


def evaluate_model(model, X_test, y_test, category_names):
    """
    Prints accuracy, precision, recall and F1 information of
    a given model

    Args:
        model : sklearn model object
            Model to evaluate
        X_test : DataFrame
            Testing set
        y_test : DataFrame
            Expected results
        category_names : list
            values to title y_prediction columns with

    Returns:
        None
    """
    y_preds_array = model.predict(X_test)
    y_preds = pd.DataFrame(y_preds_array)
    y_preds.columns = category_names
    accuracy_sum = 0
    precision_sum = 0
    recall_sum = 0
    f1_sum = 0
    y_col_count = 0
    for col in y_test:
        print("\n****{}****\n".format(col),
              classification_report(y_test[col], y_preds[col]))
        report = classification_report(
            y_test[col], y_preds[col], output_dict=True)
        accuracy_sum += report['accuracy']
        precision_sum += report['weighted avg']['precision']
        recall_sum += report['weighted avg']['recall']
        f1_sum += report['weighted avg']['f1-score']
        y_col_count += 1

    acc_agg = round(accuracy_sum/y_col_count, 5)
    precision_agg = round(precision_sum/y_col_count, 5)
    recall_agg = round(recall_sum/y_col_count, 5)
    f1_agg = round(f1_sum/y_col_count, 5)

    print('----- Average values of entire model -----\n')
    print('F1 Accuracy avg: ', acc_agg)
    print('Precision weighted avg: ', precision_agg)
    print('Recall weighted avg: ', recall_agg)
    print('F1 weighted avg: ', f1_agg)
    print('------------------------------------------\n\n')

def save_model(model, model_filepath, X, y):
    """
    Saves a new model object with the optimised parameters
    as a .pkl file

    Args:
        model : sklearn model object
            Model to save
        model_filepath : str
            Relative location to save .pkl file
        X : DataFrame
            Full X set
        y : DataFrame
            Full y set

    Returns:
        None
    """
    with timer('{} model fit and pickle'.format(model_filepath)):
        model = model.fit(X, y)
        pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        category_names = ['related', 'request', 'offer',
                          'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
                          'security', 'military', 'water', 'food', 'shelter',
                          'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
                          'infrastructure_related', 'transport', 'buildings', 'electricity',
                          'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
                          'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
                          'other_weather', 'direct_report']
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y = load_data(database_filepath, category_names)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        
        print('Building model...')
        model = build_model()

        print('Obtaining tuned model hyperparameters...')
        params = {
            'vect__ngram_range': ((1, 1), (1, 2)),
            #'vect__max_df': (0.5, 0.75, 1.0),
            'vect__max_features': (None, 5000, 10000),
            'tfidf__use_idf': (True, False),
            #'clf__estimator__n_neighbors': (2, 5, 10)
        }
        best_model = best_gridsearch_estimator(model, X_train, y_train, params)

        print('Training tuned model...')
        with timer("Train with tuned hyperparameters"):
            best_model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(best_model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(best_model, model_filepath, X, y)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
