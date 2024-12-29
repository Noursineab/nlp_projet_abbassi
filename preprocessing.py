import os
import pandas as pd
import string
import re

# Path to thedataset folder
dataset_path = "csvfiles"

# List the files in the dataset folder to see what we have
files = os.listdir(dataset_path)
print(files)

# Load the dataset
df = pd.read_csv(os.path.join(dataset_path, 'news_dataset.csv'))

# Display the first few rows and some statistiques of the dataset
print(df.head())
print(df.info())
print(df.describe())
print(df.duplicated().sum())

# Drop duplicates
df.drop_duplicates(keep="first", inplace=True)

# Check the dataset after removing duplicates
print(df.describe())
print(df.duplicated().sum())

###removing punctuation
def remove_punctuation(text):
    return "".join([i for i in text if i not in string.punctuation])


df['clean_data'] = df['data'].apply(remove_punctuation)

###converting to lowercase
df['data_lower'] = df['clean_data'].apply(lambda x: x.lower())

### tokenization
def tokenization(text):
    tokens = re.split(r'\W+', text)  # Corrected regular expression
    return [token for token in tokens if token]  


df['data_tokenied'] = df['data_lower'].apply(lambda x: tokenization(x))


###removing stop words
import nltk
stopwords = nltk.corpus.stopwords.words('english')

def remove_stopwords(text):
    output= [i for i in text if i not in stopwords]
    return output

#applying the function
df['no_stopwords']= df['data_tokenied'].apply(lambda x:remove_stopwords(x))

###stemming (it is all commented because it is irrelevant to the project)

##from nltk.stem.porter import PorterStemmer

##porter_stemmer = PorterStemmer()

##def stemming(text):
##    stem_text = [porter_stemmer.stem(word) for word in text]
##    return stem_text

##df['data_stemmed']=df['no_stopwords'].apply(lambda x: stemming(x))


###lemmatization
from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()

# Exception list for words that should not be lemmatized
exceptions = ["us", "i", "u", "bbc", "nfl"]  # this was added due to lemmetizer transforming US to u during test out

# Lemmatizer function
def lemmatizer(text):
    lemm_text = []
    for word in text:
        if word in exceptions:
            lemm_text.append(word)  # Keep the word as it is if it's in the exception list
        else:
            lemm_text.append(wordnet_lemmatizer.lemmatize(word))  # Lemmatize other words
    return lemm_text

df['data_lemmatized']=df['no_stopwords'].apply(lambda x:lemmatizer(x))

# Display the cleaned data
print(df[['data', 'data_lemmatized']].head())

# Replace the 'data' column with the cleaned data ('msg_lemmatized')
df_cleaned = df[['label', 'data_lemmatized']].copy()
df_cleaned.rename(columns={'data_lemmatized': 'data'}, inplace=True)

# Save the cleaned data to a new CSV file (same structure as the original)
df_cleaned.to_csv('csvfiles/cleaned_bbc_data.csv', index=False)
