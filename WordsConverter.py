import random
import re
import streamlit as st
import pickle
# def highlight_words(words, paragraph):
#     highlighted_paragraph = paragraph
#     color_map = {}
#
#     for word in words:
#         if word not in color_map:
#             # Generate a random color in RGB format
#             color = f'\033[38;2;{random.randint(0, 255)};{random.randint(0, 255)};{random.randint(0, 255)}m'
#             color_map[word] = color
#
#         highlighted_word = f'\033[3m{color_map[word]}{word}\033[0m'
#         highlighted_paragraph = highlighted_paragraph.replace(word, highlighted_word)
#
#     return highlighted_paragraph

def highlight_words(words, paragraph):
    highlighted_paragraph = paragraph
    st.write(paragraph)
    st.write(words)

    for word in words:
        pattern = re.compile(f'({word})', re.IGNORECASE)
        highlighted_paragraph = pattern.sub(r'<mark>\1</mark>', highlighted_paragraph)

    st.write(highlighted_paragraph)
    return highlighted_paragraph



# word_list = ['apple', 'banana', 'orange', 'pear']
# paragraph = "I like to eat an apple and a banana. The orange is also a tasty fruit. I prefer pears over apples."
#
# highlighted_text = highlight_words(word_list, paragraph)
# print(highlighted_text)


