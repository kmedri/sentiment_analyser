import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def load_csv(path):
    df = pd.read_csv(path)
    return df


df = load_csv('assets/cleaned_tweets.csv')


def main():
    st.title('Preprocessing')
    st.write('This page visualises some of the pre-processed data.')
    # Dataframe
    st.subheader('Dataframe')
    st.write(df)

    # Create a Boxplot of Tweet Lengths
    st.subheader('Boxplot of Tweet Lengths')
    st.write('A boxplot, also known as a box-and-whisker plot, is used to display the distribution of a dataset and summarize its key statistical properties. It provides a visual representation of the minimum, first quartile, median, third quartile, and maximum values of the data. Boxplots are particularly useful for identifying outliers, understanding the range and spread of the data, and comparing distributions across different categories or groups.')

    # Create a figure and axis for the boxplot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create the boxplot
    ax.boxplot(df['tweet_length'], vert=False, flierprops=dict(markerfacecolor='r', marker='D'), patch_artist=True)
    ax.set_title('Boxplot of Tweet Lengths')

    # Display the boxplot in Streamlit
    st.pyplot(fig)

    # Create a Violinplot of Tweet Lengths
    st.subheader('Violinplot of Tweet Lengths')
    st.write('A violinplot is similar to a boxplot but provides additional information about the density or distribution of the data. It displays a combination of a boxplot and a kernel density plot, allowing for a more detailed view of the data distribution. The width of the violin at a particular point indicates the density of data at that value. Violinplots are useful for understanding the shape, skewness, and multimodality of the data, as well as comparing distributions between different groups.')

    # Create a figure and axis for the violinplot
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    # Create the violinplot
    ax2.violinplot(df['tweet_length'], vert=False, showmedians=True, showextrema=True)

    # Set the title of the violinplot
    ax2.set_title('Violinplot of Tweet Lengths')

    # Display the violinplot in Streamlit
    st.pyplot(fig2)

    # Create a Piechart of Tweet Sentiment
    st.subheader('Piechart of Tweet sentiment')
    st.write('A pie chart is a circular graph divided into sectors, each representing a proportion or percentage of a whole. It is commonly used to illustrate the composition or distribution of categorical data. Pie charts are particularly effective for displaying data with a small number of categories and comparing the relative sizes of those categories. They provide a quick and intuitive visualization of how the parts contribute to the whole and allow for easy identification of the largest and smallest categories. However, it is important to note that pie charts can become less effective as the number of categories increases or when the differences in proportions are subtle.')

    # Create a figure and axis for the piechart
    fig3, ax3 = plt.subplots(figsize=(10, 6))

    # Create the piechart
    ax3.pie(df['sentiment'].value_counts(), labels=['Positive', 'Negative'], autopct='%1.1f%%', shadow=True, startangle=90)

    # Display the piechart in Streamlit
    st.pyplot(fig3)

if __name__ == "__main__":
    main()