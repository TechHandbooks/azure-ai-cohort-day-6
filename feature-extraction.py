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
