import os
import pandas as pd

#Path to our dataset (downloaded from https://www.kaggle.com/datasets/alfathterry/bbc-full-text-document-classification)
news_folder_path = 'news' 
data = []
labels = []

#files contain different encodings so need to put that in consideration when reading
def read_file(file_path):
    encodings = ['utf-8', 'ISO-8859-1', 'windows-1252']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read().strip()  
        except UnicodeDecodeError:
            continue
    # incase of all encodings failing we return empty string
    print(f"Error decoding file: {file_path}")
    return ""

# go through each directory (each different category) and append the articles files with the according label
for category in os.listdir(news_folder_path):
    category_path = os.path.join(news_folder_path, category)
    
   
    if os.path.isdir(category_path):
        
        for filename in os.listdir(category_path):
            file_path = os.path.join(category_path, filename)
            
           
            if filename.endswith(".txt"):
                article = read_file(file_path)
                
                
                if article:
                    data.append(article)
                    labels.append(category)

# dataframe creation
df = pd.DataFrame({'data': data, 'label': labels})

# saving the dataset into csv file
df.to_csv('csvfiles/news_dataset.csv', index=False)

print("CSV file 'news_dataset.csv' has been created.")
