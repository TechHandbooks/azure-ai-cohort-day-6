# Twitter Data Set Analysis Mini Project

-- 
### Azure ML Pipeline build using Designer in ML Studio

![ML Pipeline](https://github.com/user-attachments/assets/cc1feebb-57fd-4eb1-8a25-6baa9739c2b2)

---

### Data Collection
Twitter Dataset can be downloaded from Kaggle - [Twitter-Dataset](https://www.kaggle.com/datasets/goyaladi/twitter-dataset)

---
## Data Pre-processing
 Using Prebuild components like `Select Columns in Dataset`, `Remove Duplicate Rows` and `Preprocess Text`, data cleaning process can be efficiently managed.

---

### Custom Preprocessing Filter and Visualization
The Following Code is used as `Excute Python Script` Component in ML Studio Designer
```
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
```

---

### Configure Feature Extraction using N-Gram Extract from Text
There is a pre-built `Extract N-Gram Features from Text` Component which can be leveraged for Feature Extraction.

---

### Custom Feature Extraction using TF-IDF Vectorizer
The Following Code is used as `Excute Python Script` Component in ML Studio Designer for Feature Extraction.
```
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def azureml_main(dataframe1 = None, dataframe2 = None):
    # Convert the input DataFrame to a list of documents
    documents = dataframe1['Cleaned_Tweets'].tolist()  # Replace 'Text' with your column name

    # Initialize the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=100)  # Limit to top 100 features
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Convert the TF-IDF matrix to a DataFrame
    feature_names = vectorizer.get_feature_names()  # Use get_feature_names() method
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

    # Concatenate the original DataFrame with the TF-IDF DataFrame
    result_df = pd.concat([dataframe1, tfidf_df], axis=1)

    return result_df,  # Return the DataFrame with extracted features
```
