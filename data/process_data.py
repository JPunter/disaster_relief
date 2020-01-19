import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import os
import time
from contextlib import contextmanager

@contextmanager
def timer(process):
    t0 = time.time()
    yield
    if int(time.time() - t0) < 60:
        print('{} executed in {} seconds\n'.format(
            process, int(time.time() - t0)))
    else:
        print('{} executed in {} minutes\n'.format(
            process, int(time.time() - t0) / 60))


def load_data(messages_filepath, categories_filepath):
    with timer("Load data"):
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        messages = pd.read_csv(messages_filepath)
        categories = pd.read_csv(categories_filepath)
        df = messages.set_index('id').join(
            categories.set_index('id'), on='id').reset_index()
        return df


def clean_data(df):
    with timer("Clean data"):
        print('Cleaning data...')
        # Split categories into separate columns
        cat_df = df['categories'].str.split(';', expand=True)
        print('    Created {} new category columns'.format((cat_df.shape[1] - df.shape[1])))
        # Rename the columns of `categories`
        cat_df.columns = [x.split('-')[0] for x in list(cat_df.iloc[0, :])]
        # Change columns into booleans
        cat_df = cat_df.apply(lambda x: pd.to_numeric(
            x.replace('2', '1').str.split('-', expand=True)[1]))
        print('    Forced all categories into binary values')
        # Drop child_alone as is all 0
        cat_df = cat_df.drop(['child_alone'], axis=1)
        print('    Dropped "Child Alone" category, as all values are 0')
        # Drop the original categories column from `df`
        df_nocat = df.drop(['categories'], axis=1)
        # Concatenate the original dataframe with the new `categories` dataframe
        df_concat = df_nocat.join(cat_df)
        # Drop duplicates
        duplicate_count = sum(df_concat.duplicated())
        df_concat = df_concat.drop_duplicates()
        print('    Removed {} duplicates'.format(duplicate_count))
        return df_concat

def save_data(df, database_filename):
    with timer("Save data"):
        print('Saving data...\n    DATABASE: {}'.format(database_filename))
        db_loc = 'sqlite:///{}'.format(database_filename)
        engine = create_engine(db_loc)
        table_name = 'labelled_messages'
        try:
            df.to_sql(table_name, engine, index=False)
            print("    Table '{}' saved into {} database".format(table_name, database_filename))
        except ValueError as e:
            print("    ERROR: Couldn't save table '{}' due to error: \n    ***{}***".format(
                  table_name, e))


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        df = load_data(messages_filepath, categories_filepath)

        df = clean_data(df)
        
        save_data(df, database_filepath)
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
