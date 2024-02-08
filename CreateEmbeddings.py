import os
import pandas as pd
import openpyxl
from openai import OpenAI
import pickle



def excel_to_dataframe(file_path):
    df = pd.read_excel(file_path)
    print(df.columns)
    return df


def combine_genres_and_description(df=None):
    df['CombinedText'] = 'Genres: ' + df['Genres'] + ' | Description: ' + df['Description']
    return df


def generate_embeddings(df, API_KEY):
    client = OpenAI(api_key=API_KEY)
    embeddings = []
    for text in df['CombinedText']:
        response = client.embeddings.create(
            input=[text],
            model="text-embedding-ada-002"
        )
        embeddings.append(response.data[0].embedding)
    return embeddings


def save_embeddings_to_pickle(df, embeddings, file_name):
    if len(embeddings) != len(df):
        raise ValueError("Number of embeddings and titles do not match.")

    embeddings_dict = dict(zip(df['Title'], embeddings))

    with open(file_name, 'wb') as file:
        pickle.dump(embeddings_dict, file)




if __name__ == '__main__':
    df = excel_to_dataframe('imdb_tvshows.xlsx')
    combined_df = combine_genres_and_description(df)
    embeddings = generate_embeddings(combined_df, os.getenv('OpenAI_API_KEY'))
    save_embeddings_to_pickle(combined_df, embeddings, 'tv_show_embeddings.pkl')
