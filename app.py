import pickle
import joblib
import pandas as pd
import streamlit as st
from preprocessor import preprocess_text
import matplotlib.pyplot as plt
import nltk
import tensorflow as tf
nltk.download('punkt')
nltk.download('wordnet')

APP_TITLE = 'Sentiment Bot'
st.set_page_config(page_title='Home - Sentiment Bot')


@st.cache_resource
def vec():
    # Load the saved model from a file
    with open('models/tfid_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return vectorizer


text_vectorizer = vec()


# @st.cache_resource
# def get_model():
#     model = tf.keras.models.load_model('models/sentiment_CNN.h5')
#     return model\


# mlp_model = get_model()


@st.cache_resource
def get_model():
    model = joblib.load('models/naive_bayes_model.pkl')
    return model


nb_model = get_model()


def load_csv(path):
    df = pd.read_csv(path)
    return df

df = load_csv('assets/cleaned_tweets.csv')


# Building the front end

def main():
    # Colors:
    # Dark Blue - primaryColor = "#0012bb"
    # Blue - backgroundColor = "#6d7acb"
    # Light Blue - secondaryBackgroundColor = "#05afff"
    # TextColor = "#182D40"

    st.markdown(
        """
        <style>
        .css-1dp5vir {
        background-image: linear-gradient(90deg, rgb(75 81 255), rgb(120 7 165));
        }
        .css-1iktosd{
        background: transparent;
        }
        .css-1y4p8pa {
        padding: 0 1rem 10rem;
        }
        .css-1n76uvr {
        gap: 0;
        }
        .css-1taab8 {
        background-color: #aab1b5;
        }
        .css-13jzekw:hover {
        background-color: #05afff;
        }
        code {
        color: #0012bb;
        }
        .css-janbn0 {
        background-color: rgb(165 174 255 / 50%);
        }
        .css-4oy321 {
        padding: 1rem 0px 1rem 1rem;
        }
        </style>
        """, unsafe_allow_html=True
    )
    st.image("https://github.com/kmedri/sentiment_analyser/blob/main/assets/words.png?raw=true")
    st.title(APP_TITLE)
    st.write('Welcome to Sentiment Bot!')
    st.write('Our application is powered by advanced Machine Learning technologies and designed to understand the underlying sentiment in your text inputs.')
    st.write('Using an innovative combination of TF-IDF vectorization and the Multinomial Naive Bayes algorithm, Sentiment Bot effectively analyses and categorizes the sentiment of your text as positive, negative, or neutral.')
    # col1, col2 = st.columns(2)
    # with col1:
    # Button to clear chat history
    if st.button('Clear chat history'):
        st.session_state.messages = []

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # Get user input
    prompt = st.chat_input('Enter your text here')

    # React to user input
    if prompt:
        # Display user message in chat message container
        st.chat_message('user').markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # Analyze sentiment using your model
        new_text = preprocess_text(prompt)
        new_text = [new_text]
        new_text = text_vectorizer.transform(new_text).toarray()
        sentiment = nb_model.predict(new_text)  # Replace `predict()` with the appropriate method for sentiment analysis

        # Determine sentiment label
        if sentiment[0] == 1:
            response = 'The universe predicts a positive sentiment!'
            print(sentiment)
        elif sentiment[0] == 0:
            response = 'The universe predicts a negative sentiment!'
            print(sentiment[0])
        else:
            response = "I'm not sure about the sentiment."

        # Display assistant response in chat message container
        with st.chat_message('assistant'):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({'role': 'assistant', 'content': response})

    # with col2:
    # Create a Boxplot of Tweet Lengths
    # st.title('Boxplot of Tweet Lengths')

    # # Create a figure and axis for the boxplot
    # fig, ax = plt.subplots(figsize=(10, 6))

    # # Create the boxplot
    # ax.boxplot(df['tweet_length'], vert=False, flierprops=dict(markerfacecolor='r', marker='D'), patch_artist=True)
    # ax.set_title('Boxplot of Tweet Lengths')

    # # Display the boxplot in Streamlit
    # st.pyplot(fig) 

if __name__ == "__main__":
    main()
