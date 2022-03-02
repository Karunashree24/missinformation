import streamlit as st  ## streamlit
import pandas as pd  ## for data manipulation
import pickle   ## For model loading 
import spacy  ## For NLP tasks 
import time
from io import StringIO  ## for text input and output from the web app
from PIL import Image   ## For image

#Load the model the trained model which saved as a pickle file
def load_model():

#declare global variables
    global nlp
    global textcat

nlp = spacy.load(model_path)  ## will load the model from the model_path "(need to export/save the model and copy the path)"
textcat = nlp.get_pipe(model_file)   ## will load the model file "(put the .pickle model inside)"

# After loading the model, we'll use it to make a prediction on the tweet for classification.
# The prediction function will take the tweet as input and then first vectorize the tweet and then will classify it using our model
# 2 categories: misinformation or not, if the prediction is 1 means misinformation and 0 means affirmative
def predict(tweet):
    print("news = ", tweet)  ## tweet
    news = [tweet]
    txt_docs = list(nlp.pipe(tweet)) 
    scores, _ = textcat.predict(txt_docs)
    print(scores)
    predicted_classes = scores.argmax(axis=1)
    print(predicted_classes)
    result = ['misinformation' if lbl == 1 else 'affirmative' for lbl in predicted_classes]
    print(result)
    return(result)

# The run function will take the input from the user via our app as a text or text file 
# and after pressing the button it will give the output
def run():
    st.sidebar.info('You can either enter the news item online in the textbox or upload a txt file')
    st.set_option('deprecation.showfileUploaderEncoding', False)
    add_selectbox = st.sidebar.selectbox("How would you like to face reality?", ("Online", "Txt file"))
    st.title("Predicting misinformation tweet")
    st.header('This app is created to predict if a tweet is misinformation or not')
    if add_selectbox == "Online":
        text1 = st.text_area('Enter text')
        output = ""
        if st.button("Face reality"):
            output = predict(text1)
            output = str(output[0]) # since its a list, get the 1st item
            st.success(f"The news item is {output}")
            st.balloons()
      elif add_selectbox == "Txt file":
           output = ""
          file_buffer = st.file_uploader("Upload text file for new item", type=["txt"])
         if st.button("Face reality"):
             text_news = file_buffer.read()

# in the latest stream-lit version ie. 68, we need to explicitly convert bytes to text
          st_version = st.__version__ # eg 0.67.0
          versions = st_version.split('.')
          if int(versions[1]) > 67:
               text_news = text_news.decode('utf-8')
            print(text_news)
          output = predict(text_news)
          output = str(output[0])
          st.success(f"The news item is {output}")
          st.balloons()

# Execute the code only if the file was run directly and not imported.
# The module is being used standalone by the user to do corresponding appropriate actions.
if __name__ == "__main__":
load_model()
run()