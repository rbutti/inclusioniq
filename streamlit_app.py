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
import classifier as cs
import seaborn as sns
import operator
import matplotlib
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Wedge, Rectangle
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Rectangle, Circle
import numpy as np
import matplotlib.cm as cm


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


def analyze_dataframe(df_jobs):
    results = []
    for index, row in df_jobs.iterrows():
        txt = row['ORIGINAL_JOB_DESCRIPTION']
        fname = row['JOB_CLASS_TITLE']
        txt = " ".join(txt.split("\n"))
        doc = cs._gender_bias(txt, fname)
        results.append(doc)

    gender_df = pd.DataFrame(results)
    df_jobs_analysis = df_jobs.merge(gender_df, left_index=True, right_index=True)
    return df_jobs_analysis

def plot_gender_wordcloud(femdict, masdict):
    wordcloud_fem = WordCloud(background_color='white', colormap='cool').generate_from_frequencies(femdict)

    # Create a word cloud for masculine words
    wordcloud_mas = WordCloud(background_color='white', colormap='hot').generate_from_frequencies(masdict)

    # Plot the word clouds
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(wordcloud_fem, interpolation='bilinear')
    plt.title('Feminine Word Cloud')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(wordcloud_mas, interpolation='bilinear')
    plt.title('Masculine Word Cloud')
    plt.axis('off')

    # Show the plot
    plt.tight_layout()
    st.pyplot(plt)

    sorted_x = sorted(femdict.items(), key=operator.itemgetter(1), reverse=True)
    yy = [_[0] for _ in sorted_x][:25][::-1]
    xx = [_[1] for _ in sorted_x][:25][::-1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    sns.barplot(y=yy, x=xx, color='#ff77cd', ax=axes[0])
    axes[0].set_xlabel('')
    axes[0].set_ylabel('')
    axes[0].set_title('Keywords with Feminine Inclination')

    sorted_x = sorted(masdict.items(), key=operator.itemgetter(1), reverse=True)
    yy = [_[0] for _ in sorted_x][:25][::-1]
    xx = [_[1] for _ in sorted_x][:25][::-1]

    sns.barplot(y=yy, x=xx, color='#54d1f7', ax=axes[1])
    axes[1].set_xlabel('')
    axes[1].set_ylabel('')
    axes[1].set_title('Keywords with Masculine Inclination')

    plt.tight_layout()
    st.pyplot(plt)


def degree_range(n):
    start = np.linspace(0, 180, n + 1, endpoint=True)[0:-1]
    end = np.linspace(0, 180, n + 1, endpoint=True)[1::]
    mid_points = start + ((end - start) / 2.)
    return np.c_[start, end], mid_points


def rot_text(ang):
    rotation = np.degrees(np.radians(ang) * np.pi / np.pi - np.radians(90))
    return rotation


def gauge(labels=['LOW', 'MEDIUM', 'HIGH', 'VERY HIGH', 'EXTREME'], colors='jet_r', arrow=1, title=''):
    N = len(labels)

    if arrow > N:
        raise Exception("\nThe category ({}) is greater than the length of the labels ({})".format(arrow, N))

    if isinstance(colors, str):
        cmap = cm.get_cmap(colors, N)
        cmap = cmap(np.arange(N))
        colors = cmap[::-1, :].tolist()
    if isinstance(colors, list):
        if len(colors) == N:
            colors = colors[::-1]
        else:
            raise Exception("\nNumber of colors {} not equal to the number of categories {}\n".format(len(colors), N))

    fig, ax = plt.subplots()

    ang_range, mid_points = degree_range(N)

    labels = labels[::-1]

    patches = []
    for ang, c in zip(ang_range, colors):
        patches.append(Wedge((0., 0.), .4, *ang, facecolor='w', lw=2))
        patches.append(Wedge((0., 0.), .4, *ang, width=0.10, facecolor=c, lw=2, alpha=0.5))

    [ax.add_patch(p) for p in patches]

    for mid, lab in zip(mid_points, labels):
        ax.text(0.35 * np.cos(np.radians(mid)), 0.35 * np.sin(np.radians(mid)), lab,
                horizontalalignment='center', verticalalignment='center', fontsize=14,
                fontweight='bold', rotation=rot_text(mid))

    r = Rectangle((-0.4, -0.1), 0.8, 0.1, facecolor='w', lw=2)
    ax.add_patch(r)

    ax.text(0, -0.05, title, horizontalalignment='center',
            verticalalignment='center', fontsize=12)

    pos = mid_points[abs(arrow - N)]

    ax.arrow(0, 0, 0.225 * np.cos(np.radians(pos)), 0.225 * np.sin(np.radians(pos)),
             width=0.01, head_width=0.02, head_length=0.1, fc='k', ec='k')

    ax.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
    ax.add_patch(Circle((0, 0), radius=0.01, facecolor='w', zorder=11))

    ax.set_frame_on(False)
    ax.axes.set_xticks([])
    ax.axes.set_yticks([])
    ax.axis('equal')

    return fig


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
        df_jobs_analysis = analyze_dataframe(df_jobs)
        femdict, masdict = cs.get_gender_dict(df_jobs_analysis)
        df_jobs_analysis['difference'] = df_jobs_analysis["fem_wc"] - df_jobs_analysis['mas_wc']
        st.write(df_jobs_analysis)
        plot_gender_wordcloud(femdict, masdict)
        df_jobs_analysis['unconscious_bias'] = df_jobs_analysis.apply(lambda x: cs.find_score(x), axis=1)
        st.write("Diversity Score :"+str(df_jobs_analysis['unconscious_bias'][0]))

        cols = ['#007A00', '#0063BF', '#FFCC00', '#e58722', '#d6202f', '#007A00', '#0063BF', '#FFCC00', '#e58722',
                '#d6202f']
        labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        fig1 = gauge(labels=labels, colors=cols, arrow=int(df_jobs_analysis['unconscious_bias'][0]), title='Benefits Specialist (Bias: 8.8 / 10)')
        st.pyplot(fig1)

if __name__ == '__main__':
    main()
