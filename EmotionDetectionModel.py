import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pandas as pd
import numpy as np
import matplotlib .pyplot as plt
import nltk
import re
import contractions
import pickle
import json
import requests
import streamlit.components.v1 as com
# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale = 2)
import missingno as msno
import plotly.graph_objects as go
import plotly.express as px

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')



from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from os import path
from PIL import Image
from streamlit_option_menu import option_menu
from lime import lime_text
from lime.lime_text import LimeTextExplainer
from streamlit_lottie import st_lottie
from plotly.subplots import make_subplots
from wordcloud import WordCloud
from contextlib import contextmanager




def LowerCase(text):
    #text=[y.lower() for y in text]
    text = text.lower()
    return text

def ContractionExpansion(text):
  text = contractions.fix(text)
  return text

def Tokenization(text):
    text = nltk.word_tokenize(text)
    return text

def PunctuationWhitespaceRemoval(text):
    #Punctuation
    text = [re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), '', token) for token in text]

    #Whitespace
    text = [re.sub('\s+', '', token) for token in text]

    #Exclude unwanted empty string token
    text = [token for token in text if token.strip()]
    return text

def StopwordsRemoval(text):
    stop_words = set(stopwords.words('english'))
    text = [token for token in text if token.lower() not in stop_words]
    return text


# Initialize the WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Function to convert POS tags from Treebank to WordNet
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun

def Lemmatization(text):
  # Lemmatize the tokenized words in the 'tokenized_review' column with specified POS tags
  text = [lemmatizer.lemmatize(word, get_wordnet_pos(pos_tag)) for word, pos_tag in nltk.pos_tag(text)]
  return text

# Function to remove usernames from the text
def UsernameRemoval(text):
    # Define a regular expression pattern to match Twitter handles
    twitter_handle_pattern = r'@[A-Za-z0-9_]+'
    # Use re.sub to replace Twitter handles with an empty string
    text = re.sub(twitter_handle_pattern, '', text)
    return text

def DigitRemoval(text):
    # Remove standalone digits
    text = re.sub(r'\b\d+\b', '', text) #remove digit

    # Remove words containing digits
    text = re.sub(r'\b\w*\d\w*\b', '', text) #remove worddigit
    return text

# Function to remove URLs from the tweets
def URLRemoval(text):
  url_pattern = r'https?://\S+|www\.\S+'
  text = re.sub(url_pattern, '', text)
  return text

# Function to remove Non-ASCII characters like 'Ã¯Â¿Â'
def Non_ASCIIRemoval(text):
    text = re.sub(r'[^\x00-\x7F\u00EF\u00BF]+', '', text)
    return text

def EmojiRemove(text):
    # Remove emojis using regex
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)

    
def normalized_sentence(sentence):
    sentence= LowerCase(sentence)
    sentence= ContractionExpansion(sentence)
    sentence= EmojiRemove(sentence)
    sentence= UsernameRemoval(sentence)
    sentence= URLRemoval(sentence)
    sentence= Non_ASCIIRemoval(sentence)
    sentence= DigitRemoval(sentence)

    sentence= Tokenization(sentence)
    sentence= StopwordsRemoval(sentence)
    sentence= PunctuationWhitespaceRemoval(sentence)
    sentence= Lemmatization(sentence)

    return sentence

#---------------------------------------------------------------------------------------------------------------------
#Fitted Tokenizer
MAXLEN = 35
tokenizer = Tokenizer(oov_token='UNK')

data = pd.read_csv("FYP_2ndTwitterDataset.csv", names=['Text', 'Emotion_Label'], sep=';')

# Load the tokenizer from file
with open('tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)

model = load_model('best_biLSTM_model.h5', compile=False)

label_encoder = joblib.load('label_encoder.pkl')

emoji = {"anger": "😡", "fear":"😨","joy": "😆","love":"🥰","sadness":"😭","surprise":"😮"}
def load_lottiefile(filepath: str):
   with open(filepath,"r") as f:
      return json.load(f)
def load_lottieurl(url:str):
   r = requests.get(url)
   if r.status_code != 200:
      return None
   return r.json()

explainer = LimeTextExplainer(class_names=['anger', 'fear','joy','love','sadness','surprise'])

# Define a function that takes raw text as input and returns the predicted class probabilities
def predict_proba(texts):
    preprocessed_texts = [normalized_sentence(text) for text in texts]
    sequences = loaded_tokenizer.texts_to_sequences(preprocessed_texts)
    padded_sequences = pad_sequences(sequences, maxlen=MAXLEN, padding='post')
    probabilities = model.predict(padded_sequences)
    return probabilities

def main():
  st.set_page_config(
    page_title="Tweets Emotion Detection (TP063195_FYP)",
    page_icon="😊",  # You can use emojis as icons
    layout="centered",  # You can choose "wide" or "centered"
    initial_sidebar_state="auto",  # You can choose "auto", "expanded", or "collapsed"
  )
  
  with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Tweets Emotion Detection", "About", "Exploration", "Modelling & Result", "Acknowledgement"],
        icons=["chat-right-quote", "info-square", "search","journal-check","person"],  # Fix the extra comma here
        menu_icon="cast",
        default_index=0,
        #orientation="vertical",
        styles={
              "container": {"padding": "0!important", "background-color": "#f2ebea"},
              "icon": {"color": "black", "font-size": "15px"}, 
              "nav-link": {"font-size": "11px", "text-align": "left", "margin":"0px", "--hover-color": "grey"},
              "nav-link-selected": {"background-color": "#F63366"},
          }
    )
  
  if selected== "Tweets Emotion Detection":

    #https://www.youtube.com/watch?v=gk5gS47DcYs&t=246s
    #https://www.youtube.com/watch?v=TXSOitGoINE
    st.title("Tweets Emotion Detection")
    st.subheader("Predict Texts Emotion")
    st.write("""This is the trained deep learning model among all other experimented model.
             Please input any texts into the field below and hit 'ENTER' to predict your entered text's emotion.""")
    com.iframe("https://lottie.host/embed/ac3ec0e7-0d39-49c1-967f-7b6b000bbd96/qpbHMZYIZs.json")
    with st.form (key='emotion_clf_form'):
      raw_text = st.text_area("Enter Your Tweets")
      submit_text = st.form_submit_button(label='ENTER')

    if submit_text and raw_text.strip():  # Check if the entered text is not empty
        col1, col2 = st.columns(2)
        normalized_text = normalized_sentence(raw_text)
        preprocessed_text = loaded_tokenizer.texts_to_sequences([normalized_text])
        preprocessed_text = pad_sequences(preprocessed_text, maxlen=MAXLEN, padding='post')

        if preprocessed_text.any():  # Check if the preprocessed text is not empty
            prediction = label_encoder.inverse_transform(np.argmax(model.predict([preprocessed_text]), axis=-1))[0]

            class_probabilities = zip(label_encoder.classes_, model.predict([preprocessed_text])[0])

            with col1:
                st.success("Your Entered Tweets")
                st.write(raw_text)
                st.success("Preprocessed Tweets")
                st.write(f"{normalized_text}")

            with col2:
                st.success("Predicted Emotion")
                emoji_icon = emoji[prediction]
                st.write("{}:{}".format(prediction, emoji_icon))

                st.success("Distribution of Emotion Prediction Probability")
                labels, probabilities = zip(*class_probabilities)

                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

                fig, ax = plt.subplots()
                bars = ax.barh(labels, probabilities, color=colors)

                for bar, probability in zip(bars, probabilities):
                    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2, f'{probability:.3f}', ha='center', va='center')

                ax.set_xlabel('Probability')
                ax.set_title('Emotion Prediction Probability Distribution')
                st.pyplot(fig)

            st.success("LIME Explanation")
            exp = explainer.explain_instance(raw_text, predict_proba, num_features=6, labels=[0, 1, 2, 3, 4, 5])
            lime_html = exp.as_html()
            st.components.v1.html(lime_html, height=800, width=1000)
        else:
            st.warning("Please enter valid text to predict emotions.")
    elif submit_text:
        st.warning("Please enter text in the input field to predict emotions.")
        
  elif selected=="About":
    st.title("About This Project")
    st.subheader("Text Based Emotion Detection (TBED) on Tweets using Natural Language Processing (NLP)")
    with st.container():
      st.write("Teo Zee Xuan | TP063195 | APU3F2305CS(DA) ")
      st.write("B.Sc. (Hons) Computer Science Specialism in Data Analytics")
      st.write("Asia Pacific University of Technology and Innovation.")
    st.write("---")
    st.write("""
        ## Problem Statements❗
        1.	Social media has become the primary communication tool in today’s digital age and most of the generated data is textual data which left unanalyzed.
        2.	The ability to identify emotions in daily conversation is crucial but difficult. 
        3.	Behavioural problems can lead to misinterpretation of emotions.
        4.  The reviews which carry emotion feelings post on social media platforms can significantly impact a company’s reputation.
        5.  The right of free speech on social media allow anyone to share anything to the public.
        """)
    
    st.write("---")
    st.write("""
        ## Project Description 📑
        **Aim**
             
        The project aims to develop a predictive model which can identify a specific emotion from text-based sources, specifically tweets using Natural Language Processing.
                                    
        **Objectives**
        1.	To obtain an open-source dataset that consists of labelled target variables and the size of the dataset must be suitable to build the selected models.
        2.	To perform Exploratory Data Analysis based on the found dataset and perform suitable pre-processing techniques for textual data before stepping into the model building.
        3.	To train different models and evaluate the models with suitable evaluation measures.
        4.	To compare the model performance results and select the best optimal model that can be deployed for Text Based Emotion Detection.
        
        **Deliverables**
        1.	A user can provide an input by typing a complete sentence (at least a noun and a verb) or an emotion keyword in the input field.
        2.	The backend model must process and generate an emotional result even though the prediction might be irrelevant or incorrect.
        3.	A dashboard analysis of each emotion’s probability distribution and result of LIME explainer are displayed.
        4.	Other additional pages are built for users to have a better understanding of this project.
               
        **Target Users**
        
        Anyone who can convey his/ her thoughts in the digital text format, 
        especially those who have experiences in using any of the existing social media platforms specifically, Twitter.
        1.	Individual Social Media Users (esp. Twitter Users)
        2.	Social Media Platform Owner and Workers
        3.	Market Researchers
        4.	Customer Service Teams
        5.	Psychologists
        6.	Data Scientist

       
       **Constraints**
        1.	The model can only accept English text as input for emotion detection.
        2.	The recommended text length is 35 words.
        3.  The model cannot 100% handle Internet acronyms and sarcasm text.
        4.  The model is not able to detect text that does not carry any expression as there is no ‘neutral’ emotion class in the training dataset.

        """)
    st.write("---")
    st.write("""
        ## Potential Benefits 👍
        **Tangible Benefits**
        1.	Total users of a social media platform with TBED feature can be boosted.
        2.	An increase rate in advertising revenue of a social media platform. 
        3.	An increase rate in overall revenue for an online business.
        
        **Intangible Benefits**
        1.	A good improvement in user experience on social media.
        2.	A quick realization of negative emotional well-being on social media. 
        3.	A new academic research area might be arisen.       

        """)
       
    st.write("---")
    st.write("""
    ## Learn More [ℹ️]
    To Learn More about the Project, Click the links below

    1. [Project Report in PDF](https://drive.google.com/drive/folders/1tXSpTOX0fXDvfCUvWwQnZeGilZ3dNU04?usp=sharing)

    2. [GitHub Repository of the Project](https://github.com/jxuanT/fypdeployemotiondetection)
    """)


  elif selected=="Modelling & Result":
    st.title("Modelling & Result")
    st.subheader("Findings of This Project")
    # Create a dictionary with your data
    data = {
        'Tuned CNN': {
            'Training Time (s)': 1,
            'Accuracy': 0.85,
            'Precision (anger)': 0.83,
            'Recall (anger)': 0.86,
            'F1-Score (anger)': 0.85,
            'Precision (fear)': 0.92,
            'Recall (fear)': 0.77,
            'F1-Score (fear)': 0.84,
            'Precision (joy)': 0.87,
            'Recall (joy)': 0.94,
            'F1-Score (joy)': 0.91,
            'Precision (love)': 0.88,
            'Recall (love)': 0.62,
            'F1-Score (love)': 0.73,
            'Precision (sadness)': 0.92,
            'Recall (sadness)': 0.91,
            'F1-Score (sadness)': 0.91,
            'Precision (surprise)': 0.67,
            'Recall (surprise)': 0.86,
            'F1-Score (surprise)': 0.75,

            'F1-Score (Accuracy)': 0.85,
            'Precision (Macro Average)': 0.80,
            'Recall (Macro Average)': 0.83,
            'F1-Score (Macro Average)': 0.81,
            'Precision (Weighted Average)': 0.86,
            'Recall (Weighted Average)': 0.85,
            'F1-Score (Weighted Average)': 0.85
        },
        'Tuned LSTM': {
            'Precision (anger)': 0.88,
            'Recall (anger)': 0.88,
            'F1-Score (anger)': 0.88,
            'Precision (fear)': 0.88,
            'Recall (fear)': 0.86,
            'F1-Score (fear)': 0.87,
            'Precision (joy)': 0.91,
            'Recall (joy)': 0.94,
            'F1-Score (joy)': 0.92,
            'Precision (love)': 0.82,
            'Recall (love)': 0.69,
            'F1-Score (love)': 0.75,
            'Precision (sadness)': 0.92,
            'Recall (sadness)': 0.96,
            'F1-Score (sadness)': 0.94,
            'Precision (surprise)': 0.86,
            'Recall (surprise)': 0.64,
            'F1-Score (surprise)': 0.74,

            'F1-Score (Accuracy)': 0.90,
            'Precision (Macro Average)': 0.88,
            'Recall (Macro Average)': 0.83,
            'F1-Score (Macro Average)': 0.85,
            'Precision (Weighted Average)': 0.90,
            'Recall (Weighted Average)': 0.90,
            'F1-Score (Weighted Average)': 0.90
        },
        'Tuned BILSTM': {
            'Precision (anger)': 0.91,
            'Recall (anger)': 0.91,
            'F1-Score (anger)': 0.91,
            'Precision (fear)': 0.86,
            'Recall (fear)': 0.95,
            'F1-Score (fear)': 0.90,
            'Precision (joy)': 0.93,
            'Recall (joy)': 0.95,
            'F1-Score (joy)': 0.94,
            'Precision (love)': 0.87,
            'Recall (love)': 0.71,
            'F1-Score (love)': 0.79,
            'Precision (sadness)': 0.95,
            'Recall (sadness)': 0.96,
            'F1-Score (sadness)': 0.96,
            'Precision (surprise)': 0.88,
            'Recall (surprise)': 0.73,
            'F1-Score (surprise)': 0.80,

            'F1-Score (Accuracy)': 0.92,
            'Precision (Macro Average)': 0.90,
            'Recall (Macro Average)': 0.87,
            'F1-Score (Macro Average)': 0.88,
            'Precision (Weighted Average)': 0.92,
            'Recall (Weighted Average)': 0.92,
            'F1-Score (Weighted Average)': 0.92
        },
        'Tuned CNN-BILSTM': {
            'Precision (anger)': 0.83,
            'Recall (anger)': 0.86,
            'F1-Score (anger)': 0.85,
            'Precision (fear)': 0.92,
            'Recall (fear)': 0.77,
            'F1-Score (fear)': 0.84,
            'Precision (joy)': 0.87,
            'Recall (joy)': 0.94,
            'F1-Score (joy)': 0.91,
            'Precision (love)': 0.88,
            'Recall (love)': 0.62,
            'F1-Score (love)': 0.73,
            'Precision (sadness)': 0.92,
            'Recall (sadness)': 0.91,
            'F1-Score (sadness)': 0.91,
            'Precision (surprise)': 0.67,
            'Recall (surprise)': 0.86,
            'F1-Score (surprise)': 0.75,

            'F1-Score (Accuracy)': 0.87,
            'Precision (Macro Average)': 0.85,
            'Recall (Macro Average)': 0.83,
            'F1-Score (Macro Average)': 0.83,
            'Precision (Weighted Average)': 0.88,
            'Recall (Weighted Average)': 0.87,
            'F1-Score (Weighted Average)': 0.87
        },
    }

    # User selects a model from the dropdown
    selected_model = st.selectbox('Select a model:', list(data.keys()))

    # Display the selected model's information in a table
    st.write(f"**{selected_model} Model**")
    
    # Display the classification report in a table
    classification_report_data = {
        'Emotion': ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise'],
        'Precision': [data[selected_model][f'Precision ({emotion})'] for emotion in ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']],
        'Recall': [data[selected_model][f'Recall ({emotion})'] for emotion in ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']],
        'F1-Score': [data[selected_model][f'F1-Score ({emotion})'] for emotion in ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']],
        
    }
    st.table(pd.DataFrame(classification_report_data))
    
    averages_data = {
        'Metric': ['Accuracy', 'Macro Average', 'Weighted Average'],
        'F1-Score': [data[selected_model]['F1-Score (Accuracy)'], data[selected_model]['F1-Score (Macro Average)'], data[selected_model]['F1-Score (Weighted Average)']],
        'Precision': [None,data[selected_model]['Precision (Macro Average)'], data[selected_model]['Precision (Weighted Average)']],
        'Recall': [None,data[selected_model]['Recall (Macro Average)'], data[selected_model]['Recall (Weighted Average)']],
    }

    st.write(f"**{selected_model} Averages**")
    st.table(pd.DataFrame(averages_data).set_index('Metric'))

    st.write("---")
    st.write("""
    ## Best Optimal Model for Deployment
    The best optimal model is Tuned Bi-LSTM. Thus, it is used for model deployment.
    The model architecture is shown as below.
    """)
    # Define a context manager to capture printed output
    @contextmanager
    def capture_output():
        import sys
        from io import StringIO

        new_out, new_err = StringIO(), StringIO()
        old_out, old_err = sys.stdout, sys.stderr

        try:
            sys.stdout, sys.stderr = new_out, new_err
            yield new_out, new_err
        finally:
            sys.stdout, sys.stderr = old_out, old_err

    # Capture the model summary
    with capture_output() as (out, _):
        model.summary()

    # Print the model summary to Streamlit
    st.write('**Tuned Bi-LSTM Model Summary**')
    st.code(out.getvalue())
    






  elif selected == "Exploration":
    st.title("Exploration ")
    st.subheader("Exploration of The Dataset (Post Preprocessing)")
    # Read the CSV file into a DataFrame
    data = pd.read_csv("FYP_2ndTwitterDataset.csv", names=['Text', 'Emotion_Label'], sep=';')
    data = data.drop([5067,6133,6563,1625,7685,3508,9069,
    9687,4602,9786,9131,7723,11273,8775,11525,5400,1802,12562,11013,
    9605,13846,7333,14107,14314,5299,14926,15315,15329,8804,15705,15876,
    8258,16466,2112,16603,17626,2649,16774,18086,18262,18265,18353,1372,
    18502,18575,18586,18917,1793,19026,19275,17230,19887])

    data.reset_index(inplace=True, drop=True)
    
    st.write("---")
    st.write("""
             ## Bar Plot 📊
    Frequency Distribution of Each Emotion""")
    # Plot the distribution of emotions in a bar graph with different colors and values
    emotion_distribution = data['Emotion_Label'].value_counts().reset_index()
    emotion_distribution.columns = ['Emotion', 'Count']

    # Define colors for each emotion
    colors = {
        "anger": "red",
        "fear": "orange",
        "joy": "#eee600",
        "love": "green",
        "sadness": "blue",
        "surprise": "purple",
    }

    # Map colors to emotions
    emotion_distribution['Color'] = emotion_distribution['Emotion'].map(colors)

    fig = px.bar(emotion_distribution, x='Emotion', y='Count', color='Emotion', text='Count',
                 color_discrete_map=colors)
    fig.update_traces(textposition='outside', textfont_size=14)
    fig.update_layout(showlegend=False)

    st.plotly_chart(fig)

    st.write("---")
    st.write("""
             ## WordCloud ☁️
    WordCloud of Each Emotion""")
    # Streamlit selectbox for emotion selection
    selections = ["Anger", "Fear", "Joy", "Love", "Sadness", "Surprise"]
    selected_choice = st.selectbox("Select an emotion:", selections)

    # Convert both the DataFrame column and the user's selection to lowercase
    data['Emotion_Label'] = data['Emotion_Label'].str.lower()
    selected_choice = selected_choice.lower()

    # Filter data for the selected emotion
    filtered_data = data[data['Emotion_Label'] == selected_choice]

    # Apply preprocessing to the filtered data, keeping it as a list of tokens
    filtered_data['Text'] = filtered_data['Text'].apply(normalized_sentence)

    # Combine all text for the selected emotion
    all_text = ' '.join(' '.join(tokens) for tokens in filtered_data['Text'])

    # Generate the WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

    # Display the WordCloud using matplotlib
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {selected_choice.capitalize()} Emotion')
    st.pyplot(plt)
    
    # Show the DataFrame
    st.subheader("DataFrame")
    st.write("Original VS Preprocessed Data of the Selected Emotion")
    #Combine original and preprocessed data in a new DataFrame
    combined_data = pd.DataFrame({
        'Original Text': data['Text'],
        'Preprocessed Text': data['Text'].apply(normalized_sentence),
        'Emotion Label': data['Emotion_Label']
    })

    # Convert both the DataFrame column and the user's selection to lowercase
    combined_data['Emotion Label'] = combined_data['Emotion Label'].str.lower()
    selected_choice = selected_choice.lower()

    # Filter data for the selected emotion
    filtered_combined_data = combined_data[combined_data['Emotion Label'] == selected_choice]

    # Display the filtered DataFrame
    st.dataframe(filtered_combined_data)










  else:
    st.title("Acknowledgement")
    st.subheader("Special Thanks to All Contributors")
    image_path = "AcknowledgementMeetTheGroup.jpg"  
    image_path = Image.open(image_path)
    st.image(image_path)
    st.write("---")
    st.write("""
    ## Shower Thanks and Gratitudes ❤️
    The student is grateful to her
             
    1. FYP Manager: (Dr.) Dewi Octaviani, 
    Lecturer in Faculty of Computing, Engineering & Technology, School of Computing, Computer Science & Software Engineering,
             
    2. FYP Supervisor: (Mr.) Raheem Mafas, 
    Senior Lecturer in Faculty of Computing, Engineering & Technology, School of Computing, and Computer Science & Software Engineering,
             
    3. FYP Second Marker: (Dr.) Preethi Subramanian, 
    Senior Lecturer of School of Computing, Computer Science & Software Engineering . 

    4. Asia Pacific University of Technology & Innovation, Beloved Family and Fellow Friends  
                    
    Their unwavering supports, guidances, knowledge, and dedications have been essential in helping the student accomplish this significant milestone in the academic pursuit.

    """)
    st.write("---")
    st.write("""
    ## Contact Me 📧
    Let's Connect!!
    1. [LinkedIn](https://www.linkedin.com/in/zeexuan88/)
    2. [University Email](mailto:{TP063195@mail.apu.edu.my})       
    3. [Personal Email](mailto:{jjxuan21@gmail.com})
    """)

if __name__ == '__main__':
  main()
