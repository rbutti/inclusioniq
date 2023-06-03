import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb


# all the method start with _ sign are used
# in pandas apply function


def generate_word_cloud(job_description):
    # Logic to generate word cloud based on job description
    wordcloud = WordCloud().generate(job_description)

    # Display the word cloud using Matplotlib
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    # Render the plot in Streamlit
    st.pyplot(plt)

def convert_to_dataframe(title,job_description):
  df = pd.DataFrame(columns=['JOB_CLASS_TITLE', 'JOB_DUTIES', 'REQUIREMENTS','ORIGINAL_JOB_DESCRIPTION'])
  try:
    # Extract job details using regex
    # job_title = re.search(r"(.+),", job_description).group(1).strip()
    job_title = title

    # Extract job responsibilities using regex
    responsibilities_match = re.search(r"Responsibilities(.*?)Required Qualifications", job_description, re.DOTALL | re.IGNORECASE)
    job_responsibilities = responsibilities_match.group().strip() if responsibilities_match else None

    qualifications = re.search(r"Required Qualifications(.*?)JPMorgan Chase & Co., one of the oldest financial institutions", job_description, re.DOTALL | re.IGNORECASE)
    required_qualifications = qualifications.group().strip() if qualifications else None

    # Extract salary using regex
    salary_match = re.search(r"Salary(.*)", job_description, re.DOTALL | re.IGNORECASE)
    salary = salary_match.group().strip() if salary_match else None

    # Append the extracted data to the DataFrame
    data = {
                'JOB_CLASS_TITLE': job_title,
                'JOB_DUTIES': job_responsibilities,
                'REQUIREMENTS': required_qualifications,
                'ORIGINAL_JOB_DESCRIPTION': job_description
            }
    df = pd.concat([df, pd.DataFrame([data])])
    return df
  except Exception as e:
    print(f"An error occurred: {str(e)}")
    return

def vectorize_df(df):
    selected_columns = ['JOB_DUTIES', 'REQUIREMENTS', 'JOB_CLASS_TITLE']
    X_only = df[selected_columns]

    # replae nan values
    X_only['JOB_DUTIES'] = X_only['JOB_DUTIES'].fillna('Not Found')
    X_only['REQUIREMENTS'] = X_only['REQUIREMENTS'].fillna('Not Found')
    X_only['JOB_CLASS_TITLE'] = X_only['JOB_CLASS_TITLE'].fillna('Not Found')
    default_preprocessor = CountVectorizer().build_preprocessor()

    def build_preprocessor(field):
        field_idx = list(X_only.columns).index(field)
        return lambda x: default_preprocessor(x[field_idx])

    vectorizer = FeatureUnion([
        ('JOB_DUTIES', TfidfVectorizer(
            stop_words='english',
            preprocessor=build_preprocessor('JOB_DUTIES'))),
        ('REQUIREMENTS', TfidfVectorizer(
            stop_words='english',
            preprocessor=build_preprocessor('REQUIREMENTS'))),
        ('JOB_CLASS_TITLE', TfidfVectorizer(
            preprocessor=build_preprocessor('JOB_CLASS_TITLE')))
    ])
    X_train = vectorizer.fit_transform(X_only.values)
    return X_train

def main():
    # Set page title and layout
    st.set_page_config(page_title='InclusionIQ', layout='wide')

    # Left panel
    st.sidebar.title('InclusionIQ')
    job_title = st.sidebar.text_input('Job Title')
    job_description = st.sidebar.text_area('Job Description')
    submit_button = st.sidebar.button('Submit')

    # Right panel
    st.title('Word Cloud')
    if submit_button:
        df_jobs = convert_to_dataframe(job_title, job_description)
        generate_word_cloud(job_description)


if __name__ == '__main__':
    main()
