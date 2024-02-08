import logging
import os
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import openpyxl
import pickle
from numpy.linalg import norm
from openai import OpenAI
import webbrowser

client = OpenAI(api_key=os.getenv('OpenAI_API_KEY'))

def get_user_input():
    user_input = input("Which TV shows did you love watching? Separate them by a comma. Make sure to enter more than 1 show ")
    user_input_shows = user_input.split(",")
    return user_input_shows


def excel_to_dataframe(file_path):
    df = pd.read_excel(file_path)
    return df


def get_most_similar_shows(show_df, user_input_show_list):
    def find_closest(show_name):
        scores = show_df['Title'].apply(lambda x: fuzz.ratio(str(show_name), str(x)))
        max_score_index = scores.idxmax()
        return show_df['Title'].iloc[max_score_index]
    similar_shows = {}
    for user_show in user_input_show_list:
        similar_shows[user_show] = find_closest(user_show)
    return list(similar_shows.values())


def load_embeddings_from_pickle(file_name):
    with open(file_name, 'rb') as file:
        embeddings_dict = pickle.load(file)
    return embeddings_dict


def calculate_vector_average(show_list, embeddings_dict):
    embeddings = [embeddings_dict[show] for show in show_list if show in embeddings_dict]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.array([])


def cosine_similarity(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (norm(vec_a) * norm(vec_b))


def find_similar_shows(embeddings_dict, avg_vector, excluded_shows):
    similarities = []
    for show, embedding in embeddings_dict.items():
        if show not in excluded_shows:
            sim = cosine_similarity(avg_vector, embedding) * 100
            similarities.append((show, sim))

    if not similarities:
        return []
    if len(similarities) == 1:
        return similarities

    old_min = min(similarities, key=lambda x: x[1])[1]
    old_max = max(similarities, key=lambda x: x[1])[1]

    rescaled_similarities = [(show, rescale_similarity(sim, old_min, old_max, 0, old_max)) for show, sim in similarities]

    return sorted(rescaled_similarities, key=lambda x: x[1], reverse=True)


def rescale_similarity(similarity, old_min, old_max, new_min, new_max):
    return ((similarity - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min


def print_top_similar_shows(recommendations):
    print("Here are the tv shows that I think you would love:")
    for show, similarity in recommendations[:5]:
        print(f"{show} ({similarity:.2f}%)")


def generate_new_shows(top_5, original_shows):
    aggregated_list = [top_5, original_shows]
    new_shows = []
    for i in range(len(aggregated_list)):
        prompt = f"""
                 You are an experienced Hollywood producer, having great amounts of creativity and ability to create
                 tv shows on the spot.I have a list of shows {aggregated_list[i]}, please invent a new show based on these ones, give it a name
                 and a brief description, return your answer in the following format:
                 Name: X.
                 Description: Y.
                 """
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-3.5-turbo"
            )
            ai_output = chat_completion.choices[0].message.content
            lines = ai_output.strip().split('\n')
            name = ''
            description = 'something intersting, like the name of the show.'
            for line in lines:
                if line.startswith('Name:'):
                    name = line.split(':', 1)[1].strip()
                elif line.startswith('Description:'):
                    description = line.split(':', 1)[1].strip()
            new_shows.append((name, description))
        except Exception as e:
            print(f"Error generating show: {str(e)}")

    return new_shows


def print_created_shows(created_shows):
    print(f"""
                I have also created just for you two shows which I think you would love.
                Show #1 is based on the fact that you loved the input shows that you
                gave me. Its name is {created_shows[0][0]} and it is about {created_shows[0][1]}.
                Show #2 is based on the shows that I recommended for you. Its name is
                {created_shows[1][0]} and it is about {created_shows[1][1]}.
                Here are also the 2 tv show ads. Hope you like them!
          """)


def generate_show_posters(created_shows):
    posters = []
    for i in range(len(created_shows)):
        prompt = f"""
                         You are an experienced Hollywood artist, having great amounts of creativity and ability to create
                         tv shows on the spot.I have a list of fake tv shows and their descriptions {created_shows[i]}, 
                         generate a creative poster based on this information.
                      """
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1792",
            quality="standard",
            n=1,
        )
        posters.append(response.data[0].url)
    return posters


def display_posters(poster_urls):
    for url in poster_urls:
        webbrowser.open(url)


def process_data_and_get_input(file_path):
    dataframe = excel_to_dataframe(file_path)
    most_similar_shows_list = []
    check = False
    while not check:
        user_input = get_user_input()
        most_similar_shows_list = get_most_similar_shows(dataframe, user_input)
        similar_shows_string = ', '.join(most_similar_shows_list)
        done = input(f"Just to make sure, do you mean: {similar_shows_string}? (Y/N)")
        if done.lower() == 'y':
            check = True

    return most_similar_shows_list



def generate_recommendations(embeddings_file, most_similar_shows_list):
    embedding_vectors_dict = load_embeddings_from_pickle(embeddings_file)
    avg_vector = calculate_vector_average(most_similar_shows_list, embedding_vectors_dict)
    recommendations = find_similar_shows(embedding_vectors_dict, avg_vector, most_similar_shows_list)
    top_5_recommendations = [show for show, _ in recommendations[:5]]
    return top_5_recommendations, recommendations


def create_content_and_display_posters(original_shows, recommended_shows):
    generated_shows = generate_new_shows(original_shows, recommended_shows)
    print_created_shows(generated_shows)
    posters_url = generate_show_posters(generated_shows)
    display_posters(posters_url)



if __name__ == "__main__":
    most_similar_shows_list = process_data_and_get_input('imdb_tvshows.xlsx')

    top_5_recommendations, recommendations = generate_recommendations('tv_show_embeddings.pkl', most_similar_shows_list)
    print_top_similar_shows(recommendations)

    create_content_and_display_posters(most_similar_shows_list, top_5_recommendations)








