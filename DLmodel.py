import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load and prepare the dataset
def load_and_prepare_data(file_path):
    # Load the cleaned dataset
    df_clean = pd.read_csv(file_path)
    
    # Convert tokenized text back to string if necessary
    texts = df_clean['data'].apply(lambda x: ' '.join(eval(x)))
    labels = df_clean['label']
    
    # Encode labels
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)
    num_classes = len(encoder.classes_)
    print(f"Number of classes: {num_classes}")
    print("Classes:", encoder.classes_)
    
    return texts, labels, num_classes, encoder

# Vectorize the text data
def vectorize_text(train_texts, test_texts):
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(train_texts)
    X_test_tfidf = vectorizer.transform(test_texts)
    return X_train_tfidf, X_test_tfidf, vectorizer

# Create TensorFlow datasets
def create_tf_datasets(X_train, X_test, train_labels, test_labels, batch_size=32):
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train.toarray(), train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test.toarray(), test_labels))
    
    train_dataset = train_dataset.shuffle(10000).batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)
    
    return train_dataset, test_dataset

# Create the model
def create_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    return model

# Plot training history
def plot_training_history(history):
    history_dict = history.history
    
    # Prepare metrics for plotting
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs = range(1, len(acc) + 1)
    
    # Create figure with two subplots
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # Load and prepare data
    texts, labels, num_classes, encoder = load_and_prepare_data('csvfiles/cleaned_bbc_data.csv')
    
    # Split the dataset
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Vectorize text
    X_train_tfidf, X_test_tfidf, vectorizer = vectorize_text(train_texts, test_texts)
    
    # Create TensorFlow datasets
    train_dataset, test_dataset = create_tf_datasets(
        X_train_tfidf, X_test_tfidf, train_labels, test_labels
    )
    
    # Calculate class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    # Create and compile model
    model = create_model(X_train_tfidf.shape[1], num_classes)
    model.summary()
    
    # Train the model
    history = model.fit(
        train_dataset,
        epochs=10,
        validation_data=test_dataset,
        class_weight=class_weight_dict,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
        ]
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Save the trained model
    model_save_path = 'bbc_classification_model.keras'
    model.save(model_save_path)
    print(f"\nModel saved to: {model_save_path}")
    
    # Save the vectorizer and label encoder for future use
    import joblib
    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
    joblib.dump(encoder, 'label_encoder.joblib')
    print("Vectorizer and label encoder saved")

    return model, vectorizer, encoder

if __name__ == "__main__":
    main()