import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def generate_word_cloud(job_description):
    # Logic to generate word cloud based on job description
    wordcloud = WordCloud().generate(job_description)

    # Display the word cloud using Matplotlib
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    # Render the plot in Streamlit
    st.pyplot(plt)

def main():
    # Set page title and layout
    st.set_page_config(page_title='InclusionIQ', layout='wide')

    # Left panel
    st.sidebar.title('InclusionIQ')
    job_title = st.sidebar.text_input('Job Title')
    job_description = st.sidebar.text_area('Job Description')

    # Right panel
    st.title('Word Cloud')
    if st.button('Generate Word Cloud'):
        generate_word_cloud(job_description)

if __name__ == '__main__':
    main()
