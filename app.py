import streamlit as st
import tensorflow as tf
import joblib
import os
import pandas as pd
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title=" News Classifier by ABBASSI Noursine",
    page_icon="📰",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
        .stButton>button {
            width: 100%;
        }
        .main {
            padding: 2rem;
        }
        .stAlert {
            padding: 1rem;
            margin: 1rem 0;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_transformers():
    """Load the model, vectorizer, and encoder with caching"""
    try:
        model = tf.keras.models.load_model('bbc_classification_model.keras')
        vectorizer = joblib.load('tfidf_vectorizer.joblib')
        encoder = joblib.load('label_encoder.joblib')
        return model, vectorizer, encoder
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        return None, None, None

def predict_category(text, model, vectorizer, encoder):
    """Make predictions on input text"""
    try:
        # Preprocess the text using the saved vectorizer
        text_vectorized = vectorizer.transform([text])
        
        # Make prediction
        prediction = model.predict(text_vectorized.toarray())
        
        # Get prediction probabilities
        probabilities = prediction[0]
        
        # Get the predicted class
        predicted_class_index = prediction.argmax(axis=1)[0]
        predicted_class = encoder.inverse_transform([predicted_class_index])[0]
        
        # Get all class names and their probabilities
        class_names = encoder.classes_
        class_probabilities = {name: float(prob) for name, prob in zip(class_names, probabilities)}
        
        return predicted_class, class_probabilities
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None

def read_text_file(uploaded_file):
    """Read text from uploaded file"""
    try:
        text = uploaded_file.read()
        if isinstance(text, bytes):
            text = text.decode('utf-8')
        return text
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def main():
    st.title("📰  News Article Classifier by ABBASSI Noursine")
    st.markdown("Upload or enter a news article to classify its category")
    
    # Load model and transformers
    model, vectorizer, encoder = load_model_and_transformers()
    if not all([model, vectorizer, encoder]):
        st.error("Failed to load model components. Please check if all model files are present.")
        return

    # Create tabs
    tab1, tab2 = st.tabs(["Enter Text", "Upload File"])
    
    with tab1:
        # Text input option
        st.subheader("Enter Article Text")
        text_input = st.text_area(
            "Enter your news article text here:",
            height=200,
            placeholder="Paste your article text here..."
        )
        
        if st.button("Classify Text", key="classify_text"):
            if text_input.strip():
                with st.spinner("Analyzing text..."):
                    predicted_class, probabilities = predict_category(
                        text_input, model, vectorizer, encoder
                    )
                    if predicted_class and probabilities:
                        display_results(predicted_class, probabilities)
            else:
                st.warning("Please enter some text to classify.")

    with tab2:
        # File upload option
        st.subheader("Upload Article")
        uploaded_file = st.file_uploader(
            "Choose a text file",
            type=['txt', 'csv', 'doc', 'docx'],
            help="Upload a text file containing the news article"
        )
        
        if uploaded_file:
            text = read_text_file(uploaded_file)
            if text:
                st.text_area("File Content:", text, height=200)
                if st.button("Classify File", key="classify_file"):
                    with st.spinner("Analyzing file content..."):
                        predicted_class, probabilities = predict_category(
                            text, model, vectorizer, encoder
                        )
                        if predicted_class and probabilities:
                            display_results(predicted_class, probabilities)

def display_results(predicted_class, probabilities):
    """Display classification results with a nice UI"""
    st.markdown("---")
    st.subheader("Classification Results")
    
    # Display main prediction
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Predicted Category", predicted_class)
    
    # Display probability distribution
    with col2:
        # Create a DataFrame for the probabilities
        prob_df = pd.DataFrame({
            'Category': probabilities.keys(),
            'Probability': [f"{prob:.2%}" for prob in probabilities.values()]
        })
        st.dataframe(prob_df, hide_index=True)
    
    # Show confidence level message
    max_prob = max(probabilities.values())
    if max_prob > 0.8:
        st.success("High confidence prediction! ✨")
    elif max_prob > 0.5:
        st.info("Moderate confidence prediction")
    else:
        st.warning("Low confidence prediction - consider reviewing the text")

if __name__ == "__main__":
    main()