import sys

import joblib
import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine

nltk.download(['punkt', 'wordnet'])


def load_data(database_filepath):
    '''read messages table from prvoded database. Returns 'message' as X, categories as y and names of categories in y'''
    # connect to database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    con = engine.connect()
    # read table messages into dataframe
    df = pd.read_sql_table('messages', con)
    # assign messages to feature Variable
    X = df.message.values
    # remove non category columns from columns list
    category_names = df.columns[~df.columns.isin(
        ['id', 'message', 'original', 'genre'])]
    # assign all category values as targets
    y = df[category_names].values
    return X, y, category_names


def tokenize(text):
    '''tokenizes, lemmatizes and cleans words in provided text'''
    # split text into words
    tokens = word_tokenize(text)
    # initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    # itertate through all tokens
    for tok in tokens:
        # lemmatize and strip whitespace
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''creates a pipeline model for classification of messages'''
    # create classifiers and transformers
    clf = MultiOutputClassifier(RandomForestClassifier())
    estimators = [('vect', CountVectorizer(tokenizer=tokenize)),
                  ('tfidf', TfidfTransformer()),
                  ('clf', clf)]
    # build pipeline
    pipeline = Pipeline(estimators)
    # possible parameters to optimize
    param_grid = {
        'vect__max_df': [0.5, 1.0],
        'tfidf__use_idf': [True, False],
        'tfidf__norm': ('l1', 'l2')
    }
    # find optimal parameters with gridSearchCV
    optimized_pipeline = GridSearchCV(
        pipeline, param_grid, verbose=3, n_jobs=-1
    )
    return optimized_pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''prints model metrics for all categories'''
    # predict on test data
    y_pred = model.predict(X_test)
    # convert to dataframes with category names as columns
    df_y_pred = pd.DataFrame(y_pred, columns=category_names)
    df_y_test = pd.DataFrame(Y_test, columns=category_names)
    # report metrics per category
    for col in df_y_pred.columns:
        print(f"Column {col}:\n"
              f"Classification report for classifier {model}:\n"
              f"{metrics.classification_report(df_y_test[col], df_y_pred[col])}\n")


def save_model(model, model_filepath):
    '''saves the model as a pickle file'''
    # pickle model
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

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
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
