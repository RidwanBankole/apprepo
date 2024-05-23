# Import the libraries
import nltk
import pandas as pd
import numpy as np
from datetime import datetime
import re
import pickle
import streamlit as st
from PIL import Image
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    # If not found, download the resources
    nltk.download(['punkt', 'stopwords'])
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize


# Define stopwords
# stop_words = set(stopwords.words('english'))  
stop_words = nltk.corpus.stopwords.words('english')


# Initialize SnowballStemmer
snowball_stemmer = SnowballStemmer(language='english')


# Unpickle model 
def unpickle_model(file_path):
    # Load the pickled model from the specified file
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
        return model
    

loaded_vectorizer = unpickle_model("models/vectorizer.pkl")
loaded_adaboost = unpickle_model("models/AdaBoost.pkl")




# Function to replace URLs with 'url' placeholder
def replace_urls(text):
    url_pattern = r'https?://\S+'
    return re.sub(url_pattern, 'url', text)


# Function to clean text
def clean_text(text, stop_words):
    # replace url in text
    text = replace_urls(text)
    # Remove punctuation using regular expression
    cleaned_text = re.sub(r"([^A-Za-z\s]+)", '', text)

    # Convert text to lowercase
    cleaned_text = cleaned_text.lower()

    # Remove stopwords
    cleaned_text = ' '.join(word for word in cleaned_text.split() if word not in stop_words)

    return cleaned_text


# Function to lemmatize text
def stem_text(text):

    # clean text
    text = clean_text(text, stop_words)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Stem each word in the text using SnowballStemmer
    snowball_stemmed_words = [snowball_stemmer.stem(word) for word in tokens]

    # Join the lemmatized tokens back into a string
    stemmed_text = ' '.join(snowball_stemmed_words)

    return stemmed_text


# Function to vectorize the text 
def vectorize_text(text_list):
    # clean text
    cleaned_texts = [stem_text(text) for text in text_list if text != ""]

    # vectorize text
    vectorized_text = loaded_vectorizer.transform(cleaned_texts)

    return vectorized_text


def label_sentiment(predict_sentiment):   
    # map the numeric prediction to text 
    if predict_sentiment == 1.0:
        return "Hate Speech"
    else:
        return "Not Hate Speech"
    

def save_feedback(feedback):
    # Prepare the feedback data
    feedback_data = {
        "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Feedback": [feedback]
    }
    feedback_df = pd.DataFrame(feedback_data)

    # Append the feedback to the CSV file
    try:
        previous_feedback = pd.read_csv("feedback/feedback.csv")
        complete_feedback = pd.concat([previous_feedback, feedback_df], axis=0)
        complete_feedback.to_csv("feedback/feedback.csv", index=False)
    except FileNotFoundError:
        feedback_df.to_csv("feedback/feedback.csv", index=False)

def navigate(page_name):
    current_url = st.experimental_get_query_params()
    current_url["page"] = page_name
    st.experimental_set_query_params(**current_url)
    st.experimental_rerun()


def main():
    st.set_page_config(
        page_title="Hate Speech Detector",
        page_icon=":rage:",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    # Load the logo image
    logo = Image.open("images/logo.jpg")

    # Insert the logo and the description 
    col1, col2 = st.columns([9, 1])
    with col1:
        st.image(logo, width=200)  
        st.markdown("""
        <div style="display: flex; align-items: center;">
            <p style="margin: 0; font-size: 10px; align-items: center;">A product from the Research and Development Department</p>
        </div>
        """, unsafe_allow_html=True)
    
    
    st.title("Hate Speech Detector Chatbot")

     # Get the current page from query parameters
    query_params = st.experimental_get_query_params()
    current_page = query_params.get("page", ["main"])[0]

    
    # Text input for user to enter message
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    if "show_feedback" not in st.session_state:
        st.session_state.show_feedback = False
    if "about_app" not in st.session_state:
        st.session_state.about_app = False
    if "about_us" not in st.session_state:
        st.session_state.about_us = False
    
    if current_page == "main":
        # Some explanation about the app and the hate speech detection model
        st.markdown("This app detects whether text inputs contains hate speech or not.")


        user_input = st.text_area("Enter text here:", value=st.session_state.user_input)
        st.session_state.user_input = user_input
        user_input = user_input.split("\n")

        # Create two columns 
        col1, col2 = st.columns(2)

        with col1:
            # Refresh button to clear the text area and hide feedback
            if st.button("↻ Refresh"):
                st.session_state.user_input = ""
                st.session_state.show_feedback = False
                st.session_state.about_app = False
                st.session_state.about_us = False


        predict_sentiment = []
        with col2:        
            # Button to send the message
            if st.button("Detect Hate Speech"):
                try:                
                    with st.spinner("Detecting..."):
                        # Clean and Vectorize user_input
                        word_vectorized = vectorize_text(user_input)
                        
                        # Predict the sentiment of the user_input
                        predict_sentiment = loaded_adaboost.predict(word_vectorized)

                except:
                    st.warning("Please enter some text.")
        
        # Convert the prediction from numeric to word

        for i in range(len(predict_sentiment)):
            st.success(f"Result: {label_sentiment(predict_sentiment[i])}")
        
        # Add a feedback mechanism section
        st.markdown("---")

        # Create two columns 
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Have a Feedback?"):
                st.session_state.show_feedback = True
                st.session_state.about_app = False
                st.session_state.about_us = False


        with col2:
            if st.button("About the App"):
                st.session_state.show_feedback = False
                st.session_state.about_app = True
                st.session_state.about_us = False
                navigate("about_app")


        with col3:
            if st.button("About Us"):
                st.session_state.show_feedback = False
                st.session_state.about_app = False
                st.session_state.about_us = True
                navigate("about_us")


        if st.session_state.show_feedback:
            st.subheader("Feedback")
            feedback = st.text_input("Have feedback or encountered an issue? Let us know!")
            if st.button("Submit Feedback"):
                # Handle feedback submission
                if feedback:
                    save_feedback(feedback)
                    st.success("Thank you for your feedback!")
                else:
                    st.warning("Please enter your feedback before submitting.")

    elif current_page == "about_app":
        st.subheader("About Hate Speech Detector")
        st.markdown("""
                Introducing our Hate Speech Detector app, a tool designed to combat hate speech. 
                In today's digital age, hate speech poses a significant threat to the safety 
                and well-being of individuals and communities, perpetuating intolerance and division. 
                Despite ongoing efforts to address this issue, hate speech often evades detection due 
                to its nuanced and ever-changing nature.
                    
                At the core of our app is a sophisticated machine learning model, specifically an 
                AdaBoost algorithm, trained to identify instances of hate speech within tweets. 
                By harnessing the power of AI, we aim to provide a means to filter 
                out harmful content, fostering a safer and more inclusive online environment for all.

                    
                Strengths of our application include the utilization of an advanced machine learning 
                technique, the AdaBoost algorithm, which ensures a high level of accuracy in identifying 
                hate speech within tweets. Additionally, our user-friendly interface makes the app 
                accessible to a wide range of users, empowering individuals from diverse backgrounds to 
                take action against hate speech with ease.
                    

                """, unsafe_allow_html=True)
        
        st.subheader("Limitations")
        st.markdown("""
                However, it's important to acknowledge the limitations of our app. While our model is 
                trained to detect a wide array of hate speech, it may not capture every instance with 
                100% accuracy. Furthermore, the dynamic nature of online communication means that new 
                forms of hate speech may emerge over time, requiring ongoing updates and enhancements 
                to boost detection capabilities.
                
                """, unsafe_allow_html=True)
        
        st.subheader("Collaborators")
        st.markdown("""
                Ridwan Bankole <br>
                Jacinta Muindi <br>
                Ifeoluwa Osasona <br>
                Samson Alfred <br>                
                
                """, unsafe_allow_html=True)
        if st.button("Back to Main"):
            navigate("main")

    elif current_page == "about_us":    
        st.subheader("About OdumareTech")
        st.markdown("""
                OdumareTech is a tech company driven by the vision to narrow the knowledge 
                gap for IT enthusiasts while providing worthwhile experiences for developing 
                in-demand competencies within the IT industry in Africa.
                    
                By offering hands-on yet tailored technology training designed to help 
                talents thrive in the ever-evolving tech landscape, we aim to develop 
                80,000+ talents across Africa by the year 2030.
                """, unsafe_allow_html=True)
        
        st.subheader("Contact Us")
        st.markdown("""                
                <span style="color: black; font-weight: bold;">Email: </span> contact@odumaretech.com                
                """, unsafe_allow_html=True)
        if st.button("Back to Main"):
            navigate("main")

if __name__ == "__main__":
    main()