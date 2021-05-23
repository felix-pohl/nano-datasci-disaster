import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
	'''reads provided csv files and returns a dataframe merged on the id column'''
	# load messages dataset
	messages = pd.read_csv(messages_filepath)
	# load categories dataset
	categories = pd.read_csv(categories_filepath)
	# merge datasets and return
	return messages.merge(categories, on='id')


def clean_data(df):
	'''create one-hot-encoding of categories column and removes duplicates in the messages column'''
	# create a dataframe of the individual category columns
	categories = df['categories'].str.split(';', expand=True)
	# select the first row of the categories dataframe
	row = df['categories'].str.split(';', expand=True).iloc[0]
	# convert values into category names by removing number and '-'
	category_colnames = row.str.replace('-\d','')
	# rename the columns of `categories`
	categories.columns = category_colnames
	# convert category values to binary encoding (0,1)
	for column in categories:
		# set each value to be the last character of the string
		categories[column] = categories[column].astype(str).str[-1]
		
		# convert column from string to numeric
		categories[column] = pd.to_numeric(categories[column])
	# remove unnecessary categories column
	df.drop('categories', axis=1, inplace=True, errors='ignore')
	# concat categories as one-hot-encoding and message data
	df = pd.concat([df, categories], axis=1)
	# remove duplicates by 'message' column
	df = df.drop_duplicates(subset=['message'])
	return df


def save_data(df, database_filename):
	'''save df into provided database in table 'messages' '''
	engine = create_engine('sqlite:///{}'.format(database_filename))
	df.to_sql('messages', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()