import os
os.system('pip install nltk scikit-learn matplotlib seaborn wordcloud')

import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from azureml.core import Run, Dataset, Datastore
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from datetime import datetime
from azureml.data.dataset_factory import FileDatasetFactory

# Ensure required NLTK data is downloaded
nltk.data.path.append('/root/nltk_data')
nltk.download('stopwords', download_dir='/root/nltk_data')
nltk.download('punkt_tab', download_dir='/root/nltk_data')
stop_words = set(stopwords.words('english'))


def filter_tweets(df, min_likes=10, start_date='2023-04-01', end_date='2023-04-30'):
    # Convert timestamp to datetime 
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    # Filter based on likes and date range 
    filtered_df = df[(df['Likes'] >= min_likes) & (df['Timestamp'] >= start_date) & (df['Timestamp'] <= end_date)] 
    return filtered_df

def preprocess_tweet(tweet):
    word_tokens = word_tokenize(tweet)
    filtered_tweet = [w for w in word_tokens if not w in stop_words]
    return ' '.join(filtered_tweet)

def azureml_main(dataframe1=None, dataframe2=None):
    run = Run.get_context(allow_offline=True)
    ws = run.experiment.workspace
    # dataset = Dataset.get_by_name(ws, name='twitter-dataset-345')
    df = dataframe1
    print("Column Names:", df.columns)
    # Preprocessed Text is the column sent
    df['Cleaned_Tweets'] = df['Preprocessed Text'].apply(preprocess_tweet)

    # Filter tweets based on likes and timestamp 
    df = filter_tweets(df)
    
    output_dir = 'outputs'
    # Create the output directory if it doesn't exist 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate a unique timestamp
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    # Example plot: Word Cloud of Cleaned Tweets
    from wordcloud import WordCloud
    all_words = ' '.join(df['Cleaned_Tweets'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Cleaned Tweets')
    wordcloud_path = os.path.join(output_dir, f'wordcloud_{timestamp}.png')
    plt.savefig(wordcloud_path)
    # Log the word cloud image 
    run.log_image("Word Cloud", path=wordcloud_path)

    # Example plot: Bar plot of tweet lengths using distplot
    df['Tweet_Length'] = df['Cleaned_Tweets'].apply(len)
    plt.figure(figsize=(10, 5))
    sns.distplot(df['Tweet_Length'], kde=True)
    plt.title('Distribution of Tweet Lengths')
    plt.xlabel('Tweet Length')
    plt.ylabel('Frequency')
    tweet_length_distribution_path = os.path.join(output_dir, f'tweet_length_distribution_{timestamp}.png')
    plt.savefig(tweet_length_distribution_path)

    # Log the tweet length distribution image 
    run.log_image("Tweet Length Distribution", path=tweet_length_distribution_path)
    
    return df,  # Ensure the return type is a tuple of dataframes
