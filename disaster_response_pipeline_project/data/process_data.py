import sys
import pandas as pd
import sqlalchemy

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on="id")
    return df

def clean_data(df):
    categories = df.categories.str.split(pat = ';' , expand=True)
    row = categories.iloc[0]
    row = row.str.replace('\d+', '')
    row = row.str.replace('-', '')
    category_colnames = row
    categories.rename(columns = category_colnames, inplace=True)
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str)
        # convert column from string to numeric
        categories[column] = categories[column].str.strip().str[-1]
        categories[column] = categories[column].astype(int)

    df.drop(['categories'], axis = 1, inplace = True)
    df = df.merge(categories, left_index=True, right_index=True)
#     #duplicate = df[df.duplicated()]
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filepath):
    engine = sqlalchemy.create_engine('sqlite:///{}'.format(database_filepath))
    df.to_sql('disaster_msgs', engine, index=False, if_exists = 'replace')  


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
