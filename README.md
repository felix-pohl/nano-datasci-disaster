# Disaster Response Pipeline Project

## Summary

Example project showcasing the use of an ETL-pipeline, training a NLP model and integrating the result into a web app.

## Motivation

Disaster response organizations often get thousands of messages following natural disasters. Organizations need ressources to filter these messages and find the most important ones requesting aid. Ressources are scarce following a disaster and could be usesd more effectively by providing aid. To reduce the manual effort this project tries to categories incoming messages for relevance and requested aid to increase efficiency.

## Installation

 * Clone repo
 * Install dependencies using your favored package installer
    * numpy
    * pandas
    * sklearn
    * flask
    * nltk
    * joblib
    * sqlalchemy

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Project File description

* `app`
    * `templates`: Html templates for rendering web app user interface
    * `run.pyv`: creates a http server and reads in cleaned data from database. Performs classification for new messages and prepares visuals.
* `data`
    * `disaster_categories.csv`: training labels for disaster messages
    * `disaster_messages.csv`: training data of original and translated disaster messages
    * `DisasterResponse.db`: prefilled Database with cleaned categorised messages
    * `process_data.py`: reads in trainingdata, cleans, removes duplicates and stores in database
* `models`
    * `train_classifier.py`: reads cleaned data from databse and fits an NLP pipeline onto training data generating a pickled model to use in the web app
