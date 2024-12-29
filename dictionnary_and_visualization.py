import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud
from typing import List, Optional
import numpy as np

def create_label_visualizations(
    data: pd.DataFrame,
    label: str,
    n_top_words: int = 10,
    min_df: int = 2,
    max_features: Optional[int] = 1000,
    output_dir: str = "csvfiles"
) -> None:

    # Create vectorizers 
    vectorizer_params = {
        'min_df': min_df,
        'max_features': max_features,
        'stop_words': 'english'
    }
    
    count_vectorizer = CountVectorizer(**vectorizer_params)
    tfidf_vectorizer = TfidfVectorizer(**vectorizer_params)
    
    # Get documents for current label
    label_docs = data[data['label'] == label]['data']
    
    if len(label_docs) == 0:
        print(f"No documents found for label: {label}")
        return
    
    # Fit and transform the data
    X_bow = count_vectorizer.fit_transform(label_docs)
    X_tfidf = tfidf_vectorizer.fit_transform(label_docs)
    
    # Create figure with improved styling
    plt.figure(figsize=(20, 10))
    sns.set_style("whitegrid")
    
    # 1. Top N most frequent words (BoW) 
    plt.subplot(2, 2, 1)
    word_freq = X_bow.sum(axis=0).A1
    word_freq_dict = dict(zip(count_vectorizer.get_feature_names_out(), word_freq))
    sorted_word_freq = sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True)
    top_n_words = sorted_word_freq[:n_top_words]
    words, freqs = zip(*top_n_words)
    
    # frequencies normalization
    normalized_freqs = np.array(freqs) / len(label_docs)
    
    bars = sns.barplot(x=normalized_freqs, y=words, palette='Blues_d')
    
    # Add value labels to bars
    for i, v in enumerate(normalized_freqs):
        bars.text(v, i, f'{v:.3f}', va='center')
    
    plt.title(f'Top {n_top_words} most frequent words - {label} (BoW)', pad=20)
    plt.xlabel('Average Frequency per Document')
    plt.ylabel('Words')
    
    # 2. Word Cloud for TF-IDF
    plt.subplot(2, 2, 2)
    tfidf_freq = X_tfidf.sum(axis=0).A1
    tfidf_freq_dict = dict(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_freq))
    
    wordcloud = WordCloud(
        width=1200,
        height=800,
        background_color='white',
        max_words=100,
        colormap='viridis',
        contour_width=3,
        contour_color='steelblue'
    ).generate_from_frequencies(tfidf_freq_dict)
    
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud - {label} (TF-IDF)', pad=20)
    

    plt.suptitle(f'Text Analysis Visualization for {label}', fontsize=16, y=1.05)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f"{output_dir}/visualizations/{label}_analysis.png", 
                bbox_inches='tight', 
                dpi=300)
    plt.close()
    
    # Save feature matrices with compression
    bow_df = pd.DataFrame(X_bow.toarray(), columns=count_vectorizer.get_feature_names_out())
    tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    
    bow_df.to_csv(f"{output_dir}/bow/bow_representation_{label}.csv.gz", 
                  index=False, 
                  compression='gzip')
    tfidf_df.to_csv(f"{output_dir}/tfidf/tfidf_representation_{label}.csv.gz", 
                    index=False, 
                    compression='gzip')


def main():
    # Load the pretreated data
    df = pd.read_csv('csvfiles/cleaned_bbc_data.csv')
    
    # Filtering empty docs
    df_clean = df[df['data'].apply(lambda x: len(str(x)) > 0)]
    
    # Create necessary directories
    import os
    for dir_name in ['visualizations', 'bow', 'tfidf']:
        os.makedirs(f"csvfiles/{dir_name}", exist_ok=True)
    
    # Create visualizations for each unique label
    for label in df_clean['label'].unique():
        print(f"\nCreating visualizations for label: {label}")
        create_label_visualizations(df_clean, label)
    
    print("All visualizations completed!")

if __name__ == "__main__":
    main()