import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def load_csv(path):
    df = pd.read_csv(path)
    return df


df = load_csv('assets/cleaned_tweets.csv')


def main():
    st.title('Preprocessing')
    st.write('This page is visualises some of the pre-processed data.')
    # Dataframe
    st.subheader('Dataframe')
    st.write(df)

    # Create a Boxplot of Tweet Lengths
    st.subheader('Boxplot of Tweet Lengths')

    # Create a figure and axis for the boxplot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create the boxplot
    ax.boxplot(df['tweet_length'], vert=False, flierprops=dict(markerfacecolor='r', marker='D'), patch_artist=True)
    ax.set_title('Boxplot of Tweet Lengths')

    # Display the boxplot in Streamlit
    st.pyplot(fig)
    
    # Create a Violinplot of Tweet Lengths
    st.subheader('Violinplot of Tweet Lengths')

    # Create a figure and axis for the violinplot
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    # Create the violinplot
    ax2.violinplot(df['tweet_length'], vert=False, showmedians=True, showextrema=True)

    # Set the title of the violinplot
    ax2.set_title('Violinplot of Tweet Lengths')

    # Display the violinplot in Streamlit
    st.pyplot(fig2)
    
    # Create a Piechart of Tweet sentiment
    st.subheader('Piechart of Tweet sentiment')
    
    # Create a figure and axis for the piechart
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    # Create the piechart
    ax3.pie(df['sentiment'].value_counts(), labels=['Positive', 'Negative'], autopct='%1.1f%%', shadow=True, startangle=90)
    
    # Display the piechart in Streamlit
    st.pyplot(fig3)

if __name__ == "__main__":
    main()