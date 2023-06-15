from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns
from plotly import tools, subplots
init_notebook_mode(connected=True)

from nltk.corpus import stopwords
from nltk.util import ngrams
import textstat, spacy, nltk
import pandas as pd, os
import numpy as np
import string, re
import random

import pandas as pd
import numpy as np
import re
import glob
import matplotlib.pyplot as plt
import pickle


lookups = {}

def cleanup(text):
    text = text.lower()
    #text = " ".join([c for c in text.split() if c not in stopwords])
    for c in string.punctuation:
        text = text.replace(c, " ")
   # text = " ".join([c for c in text.split() if c not in stopwords])

    words = []
    ignorewords = []
    for wrd in text.split():
        if len(wrd) <= 1:
            continue
        words.append(wrd)
    text = " ".join(words)
    return text


def convert_to_dataframe(title,job_description):
  df = pd.DataFrame(columns=['title', 'job_description_txt'])
  try:
    # Append the extracted data to the DataFrame
    data = {
                'title': title,
                'job_description_txt': job_description
            }
    df = pd.concat([df, pd.DataFrame([data])])
    return df
  except Exception as e:
    print(f"An error occurred: {str(e)}")
    return


def generate_score(weight, num_occurrences, scale):
    if scale == '5scale':
        if num_occurrences > 5:
            score = weight
        elif num_occurrences == 5 or num_occurrences == 4:
            score = 0.75 * weight
        elif num_occurrences == 3 or num_occurrences == 2:
            score = 0.5 * weight
        elif num_occurrences == 1:
            score = 0.1 * weight
        else:
            score = 0
    elif scale == '10scale':
        # Adjust the logic for scale 0-10
        if num_occurrences > 10:
            score = weight
        elif num_occurrences >= 9:
            score = 0.9 * weight
        elif num_occurrences >= 7:
            score = 0.7 * weight
        elif num_occurrences >= 5:
            score = 0.5 * weight
        elif num_occurrences >= 3:
            score = 0.3 * weight
        elif num_occurrences >= 1:
            score = 0.1 * weight
        else:
            score = 0
    elif scale == '100scale':
        # Adjust the logic for scale 0-100
        if num_occurrences > 100:
            score = weight
        elif num_occurrences >= 90:
            score = 0.9 * weight
        elif num_occurrences >= 70:
            score = 0.7 * weight
        elif num_occurrences >= 50:
            score = 0.5 * weight
        elif num_occurrences >= 30:
            score = 0.3 * weight
        elif num_occurrences >= 10:
            score = 0.1 * weight
        else:
            score = 0
    else:
        raise ValueError("Invalid scale")

    return score


def evaluate_dataframe(jd_df):
    load_lookups()
    #words
    jd_df['mas_words'] = jd_df.apply(
        lambda row: check_presence('masculine', row['title'] + ' ' + row['job_description_txt'], 'exact'), axis=1)
    jd_df['fem_words'] = jd_df.apply(
        lambda row: check_presence('feminine', row['title'] + ' ' + row['job_description_txt'], 'exact'), axis=1)
    jd_df['superlatives_wrds'] = jd_df.apply(
        lambda row: check_presence('superlatives', row['title'] + ' ' + row['job_description_txt'], 'exact'), axis=1)
    jd_df['rel_words'] = jd_df.apply(
        lambda row: check_presence('relationships', row['title'] + ' ' + row['job_description_txt'], 'exact'), axis=1)
    jd_df['strict_words'] = jd_df.apply(
        lambda row: check_presence('strict_words', row['title'] + ' ' + row['job_description_txt'], 'exact'), axis=1)
    jd_df['strict_phrases'] = jd_df.apply(
        lambda row: check_presence('strict_phrases', row['title'] + ' ' + row['job_description_txt'], 'exact'), axis=1)
    jd_df['mas_pronouns'] = jd_df.apply(
        lambda row: check_presence('mas_pronouns', row['title'] + ' ' + row['job_description_txt'], 'exact'), axis=1)
    jd_df['fem_pronouns'] = jd_df.apply(
        lambda row: check_presence('fem_pronouns', row['title'] + ' ' + row['job_description_txt'], 'exact'), axis=1)
    jd_df['exclusive_language_wrds'] = jd_df.apply(
        lambda row: check_presence('exclusive_language', row['title'] + ' ' + row['job_description_txt'], 'exact'),
        axis=1)
    jd_df['lgbtq_words'] = jd_df.apply(
        lambda row: check_presence('lgbtq', row['title'] + ' ' + row['job_description_txt'], 'exact'), axis=1)
    jd_df['racial_words'] = jd_df.apply(
        lambda row: check_presence('racial', row['title'] + ' ' + row['job_description_txt'], 'exact'), axis=1)

    #summation
    jd_df['mas_words_sum'] = jd_df.apply(lambda row: sum(row['mas_words'].values()), axis=1)
    jd_df['fem_words_sum'] = jd_df.apply(lambda row: sum(row['fem_words'].values()), axis=1)
    jd_df['superlatives_wrds_sum'] = jd_df.apply(lambda row: sum(row['superlatives_wrds'].values()), axis=1)
    jd_df['rel_words_sum'] = jd_df.apply(lambda row: sum(row['rel_words'].values()), axis=1)
    jd_df['strict_words_sum'] = jd_df.apply(lambda row: sum(row['strict_words'].values()), axis=1)
    jd_df['strict_phrases_sum'] = jd_df.apply(lambda row: sum(row['strict_phrases'].values()), axis=1)
    jd_df['mas_pronouns_sum'] = jd_df.apply(lambda row: sum(row['mas_pronouns'].values()), axis=1)
    jd_df['fem_pronouns_sum'] = jd_df.apply(lambda row: sum(row['fem_pronouns'].values()), axis=1)
    jd_df['exclusive_language_wrds_sum'] = jd_df.apply(lambda row: sum(row['exclusive_language_wrds'].values()), axis=1)
    jd_df['lgbtq_words_sum'] = jd_df.apply(lambda row: sum(row['lgbtq_words'].values()), axis=1)
    jd_df['racial_words_sum'] = jd_df.apply(lambda row: sum(row['racial_words'].values()), axis=1)
    jd_df['total_bias_words'] = jd_df['mas_words_sum'] + jd_df['fem_words_sum'] + jd_df['superlatives_wrds_sum'] + \
                                jd_df['rel_words_sum'] + jd_df['strict_words_sum'] + jd_df['strict_phrases_sum'] + \
                                jd_df['mas_pronouns_sum'] + jd_df['fem_pronouns_sum'] + jd_df[
                                    'exclusive_language_wrds_sum'] + jd_df['lgbtq_words_sum'] + jd_df[
                                    'racial_words_sum']

    #Bias score
    jd_df['bias_score'] = jd_df['total_bias_words'].apply(
        lambda x: 10 if x > 100 else (0 if x == 0 else (x / 100) * 10))

    #prep for machine learning
    jd_df['ml_bias_flag'] = jd_df['bias_score'].apply(lambda x: 1 if x > 4 else 0)

    return jd_df


def check_presence(flag, txt, condition):
    global lookups

    matched = {}
    txt = " " + txt.lower() + " "
    for wrd in lookups[flag]:
        if condition == "exact":
            cnt = txt.count(" " + wrd.lower() + " ")
            if cnt > 0:
                matched[wrd.lower()] = cnt
        elif condition == "startswith":
            cnt = txt.count(" " + wrd.lower())
            if cnt > 0:
                matched[wrd.lower()] = cnt

    return matched




def load_lookups():
    global lookups
    with open("models/biased_words.pkl", 'rb') as model_file:
        lookups = pickle.load(model_file)


import pandas as pd

import pandas as pd

import pandas as pd


def analysis_result_df(jd_df):
    # Create a sample dataframe
    data = {
        'Criteria': ['Masculine Words', 'Feminine Words', 'Superlative words', 'Relationship Words', 'Strict Words',
                     'Strict Phrases', 'Masculine Pronouns', 'Feminine Pronouns', 'Exclusive Language', 'LGBTQ Words',
                     'Racial Words'],
        'Count': [jd_df['mas_words_sum'][0], jd_df['fem_words_sum'][0], jd_df['superlatives_wrds_sum'][0],
                  jd_df['rel_words_sum'][0], jd_df['strict_words_sum'][0], jd_df['strict_phrases_sum'][0],
                  jd_df['mas_pronouns_sum'][0], jd_df['fem_pronouns_sum'][0], jd_df['exclusive_language_wrds_sum'][0],
                  jd_df['lgbtq_words_sum'][0], jd_df['racial_words_sum'][0]],
        'Words': [jd_df['mas_words'].apply(lambda x: ', '.join(str(key) for key in x.keys())).tolist(),
                  jd_df['fem_words'].apply(lambda x: ', '.join(str(key) for key in x.keys())).tolist(),
                  jd_df['superlatives_wrds'].apply(lambda x: ', '.join(str(key) for key in x.keys())).tolist(),
                  jd_df['rel_words'].apply(lambda x: ', '.join(str(key) for key in x.keys())).tolist(),
                  jd_df['strict_words'].apply(lambda x: ', '.join(str(key) for key in x.keys())).tolist(),
                  jd_df['strict_phrases'].apply(lambda x: ', '.join(str(key) for key in x.keys())).tolist(),
                  jd_df['mas_pronouns'].apply(lambda x: ', '.join(str(key) for key in x.keys())).tolist(),
                  jd_df['fem_pronouns'].apply(lambda x: ', '.join(str(key) for key in x.keys())).tolist(),
                  jd_df['exclusive_language_wrds'].apply(lambda x: ', '.join(str(key) for key in x.keys())).tolist(),
                  jd_df['lgbtq_words'].apply(lambda x: ', '.join(str(key) for key in x.keys())).tolist(),
                  jd_df['racial_words'].apply(lambda x: ', '.join(str(key) for key in x.keys())).tolist()]
    }
    df = pd.DataFrame(data)

    # Set 'Criteria' column as the index
    df.set_index('Criteria', inplace=True)

    return df


