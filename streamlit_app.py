from wordcloud import WordCloud
import pickle
import pandas as pd
import classifier as cs
import seaborn as sns
import operator
import re
import PIL.Image
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Rectangle, Circle
import numpy as np
import matplotlib.cm as cm
import eli5
from matplotlib.collections import PatchCollection
import processor as p
import WordsConverter as w
from sklearn.feature_extraction.text import TfidfVectorizer
from IPython.display import display, HTML


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
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Values to color in heatmap style
    cmap = plt.cm.get_cmap('coolwarm')  # Choose a colormap for the heatmap
    cmap = cmap.reversed()
    color_patches = []  # List to store colors for each patch

    for ang, val in zip(ang_range, range(1, 11)):
        color = cmap((val - 1) / 9)  # Normalize the value to range [0, 1] and get the corresponding color
        patches.append(Wedge((0., 0.), .4, *ang, facecolor='w', lw=2))
        if val in values:
            patches.append(Wedge((0., 0.), .4, *ang, width=0.10, facecolor=color, lw=2, alpha=0.5))
        color_patches.extend([color, color])

    fig, ax = plt.subplots()
    collection = PatchCollection(patches, facecolors=color_patches)
    ax.add_collection(collection)

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
# CSS styles to center the logo
custom_css = """
.sidebar .sidebar-content {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
"""


def main():
    st.set_page_config(page_title='InclusionIQ', layout='wide')
    st.image("images/innovationweeklogo.png", use_column_width=True)
    image = PIL.Image.open("images/InclusionIQ.png")
    # Reduce the size of the image
    new_size = (image.size[0] // 2, image.size[1] // 2)
    resized_image = image.resize(new_size)

    st.sidebar.image(resized_image, use_column_width=False)
    # Left panel
    st.sidebar.markdown("<span style='color:Blue;'>A machine learning tool to enhance diversity and inclusion in hiring processes by detecting and eliminating bias in job descriptions.</span>", unsafe_allow_html=True)
    job_title = st.sidebar.text_input('Job Title')
    job_description = st.sidebar.text_area('Job Description')
    submit_button = st.sidebar.button('**Submit**')

    # Right panel
    if submit_button:

        generate_word_cloud(job_description)

        st.title('Data analysis Report')
        jd_df = p.convert_to_dataframe(p.cleanup(job_title), p.cleanup(job_description))
        jd_df = p.evaluate_dataframe(jd_df)

        #update left nav
        cols = ['#007A00', '#0063BF', '#FFCC00', '#e58722', '#d6202f', '#007A00', '#0063BF', '#FFCC00', '#e58722',
                '#d6202f']
        labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        fig1 = gauge(labels=labels, colors=cols, arrow=int(jd_df['bias_score'][0]),
                     title=f"Bias Score: {jd_df['bias_score'][0]} / 10")
        st.sidebar.pyplot(fig1)

        #show table
        st.header('Tabular Data')
        analysis_df= p.analysis_result_df(jd_df)
        st.dataframe(analysis_df)


        #show wordcloud
        st.header('WordCloud')
        generate_word_cloud(' '.join(analysis_df['Words'].explode()))

        # show jobdescirption
        st.header('Job Description')
        highlighted_paragraph = w.highlight_words(list(jd_df['mas_words'].apply(lambda d: list(d.keys())).sum()),job_description, '#ffffcc')
        highlighted_paragraph = w.highlight_words(list(jd_df['fem_words'].apply(lambda d: list(d.keys())).sum()),
                                                  highlighted_paragraph, '#ffd1dc')
        highlighted_paragraph = w.highlight_words(list(jd_df['superlatives_wrds'].apply(lambda d: list(d.keys())).sum()),
                                                  highlighted_paragraph, '#bfefff')
        highlighted_paragraph = w.highlight_words(list(jd_df['rel_words'].apply(lambda d: list(d.keys())).sum()),
                                                  highlighted_paragraph, '#d9f5d1')
        highlighted_paragraph = w.highlight_words(list(jd_df['strict_words'].apply(lambda d: list(d.keys())).sum()),
                                                  highlighted_paragraph, '#e9d8fd')
        highlighted_paragraph = w.highlight_words(
            list(jd_df['strict_phrases'].apply(lambda d: list(d.keys())).sum()),
            highlighted_paragraph, '#ffe0b2')
        highlighted_paragraph = w.highlight_words(list(jd_df['mas_pronouns'].apply(lambda d: list(d.keys())).sum()),
                                                  highlighted_paragraph, '#f5f5f5')
        highlighted_paragraph = w.highlight_words(list(jd_df['fem_pronouns'].apply(lambda d: list(d.keys())).sum()),
                                                  highlighted_paragraph, '#e0ffff')
        highlighted_paragraph = w.highlight_words(
            list(jd_df['exclusive_language_wrds'].apply(lambda d: list(d.keys())).sum()),
            highlighted_paragraph, '#e6e6fa')
        highlighted_paragraph = w.highlight_words(list(jd_df['lgbtq_words'].apply(lambda d: list(d.keys())).sum()),
                                                  highlighted_paragraph, '#f08080')
        highlighted_paragraph = w.highlight_words(list(jd_df['racial_words'].apply(lambda d: list(d.keys())).sum()),
                                                  highlighted_paragraph, '#d0f0c0')
        st.markdown(highlighted_paragraph, unsafe_allow_html=True)


        st.title('InclusionIQ Model Report')

        with open("models/inclusioniq_model.pkl", 'rb') as model_file:
            model = pickle.load(model_file)
        with open("models/vectorizer_bestmodel.pkl", 'rb') as vectorizer_dump:
            vectorizer = pickle.load(vectorizer_dump)


        prediction_html = eli5.show_prediction(model, doc=job_description, vec=vectorizer,
                                               feature_names=vectorizer.get_feature_names_out(),
                                               top=(20, 20), show_feature_values=True)

        table_style = """
                <style>
                table {
                    border-collapse: collapse;
                    width: 100%;
                }

                th, td {
                    padding: 8px;
                    text-align: left;
                }

                tr:nth-child(even) {
                    background-color: green;
                    color: black;
                }

                tr:nth-child(odd) {
                    background-color: white;
                    color: black;
                }
                </style>
                """

        #Display the tables with custom CSS styling
        st.markdown(table_style, unsafe_allow_html=True)

        res = prediction_html.data.replace("\n", "")
        st.markdown(res, unsafe_allow_html=True)

        table_style = """
        <style>
        table {
            border-collapse: collapse;
            width: 100%;
        }

        th, td {
            padding: 8px;
            text-align: left;
        }

        tr:nth-child(even) {
            background-color: green;
            color: black;
        }

        tr:nth-child(odd) {
            background-color: white;
            color: black;
        }
        </style>
        """


if __name__ == '__main__':
    main()
