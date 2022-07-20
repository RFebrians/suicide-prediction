import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
from nltk.stem.porter import PorterStemmer
import pickle
import streamlit.components.v1 as components

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stops=stopwords.words('english')
nltk.download('stopwords')
    
    
st.set_option('deprecation.showPyplotGlobalUse', False)
# app would work faster if you would not read and show the data set
data= pd.read_csv("Suicide_Detection.csv")
numm = data["Unnamed: 0"][len(data)-1]
data.drop("Unnamed: 0", axis=1, inplace = True)

from pathlib import Path

# data in home tab
def home(data):
     # app would work faster if you would not read and show the data set
    if st.checkbox("Show Sample Data"):
        st.table(data.head())
   

# prediction function    
def predict(data):
    st.header("Classify a sentence")
    text = st.text_area("Enter the Sentence to Check if it is suicidal")

    porter=PorterStemmer()
    def tokenizer_porter(text):
        return [porter.stem(word) for word in text.split()]

    text = np.array(tokenizer_porter(text))
    
    def remove_stopwords(lower_tokens):
      filtered_words=[]
      for s in lower_tokens:
        temp=[]
        for token in s:
          if token not in stops:
            temp.append(token)
        filtered_words.append(temp)
      return filtered_words

    f_text = np.array(remove_stopwords([text])[0])
    f_text = " ".join(f_text)
    tfidf_vectorizer = pickle.load(open("tfidf.pickle", "rb"))
    ss2 = tfidf_vectorizer.transform([f_text])

    # load the model from disk
    loaded_model = pickle.load(open("lr_model.sav", 'rb'))
    result = loaded_model.predict(ss2)

    if st.button("Predict"):
        if result[0] == 1:
            st.error("This is a Suicidal sentence")
        else:
            st.success("Good news!\n it is not a Suicidal sentence")
    
    if st.checkbox("Need some Example?"):
        st.write("try this statement:")
        st.write(data["text"][7])


if __name__ == "__main__":
    # root = Path(__file__).parents[1]
    st.title("Psychology - Data Augmentation | Suicide Prediction")
    st.image("https://cms.qz.com/wp-content/uploads/2018/08/suicide-prediction-animated-final.gif?quality=75&strip=all&w=1200&h=630&crop=1",width = 800)
    nav = st.sidebar.radio("Menu",["Home","Dataset", "Background", "Contact"])
    if nav == "Home":
        home(data)
        predict(data)
        

    if nav == "Dataset":
        st.warning("Alpha Development . Heavily Under Progress")
        st.header("Contribute to our dataset")
        text1 = st.text_area("Enter the Sentence")
        label1 = st.selectbox("Select the class",["suicide","non-suicide"],index = 0)
        if st.button("submit"):
             #check if we are not reading the dataset
             add_lst = {"Unnamed: 0": [numm+1], "text":[text1],"class":[label1]}
             add_lst = pd.DataFrame(add_lst)
             add_lst.to_csv(root/"new_Suicide_Detection.csv",mode='a',header = False,index= False)
             st.success("Submitted . Thankyou for your contribution")
    if nav == "Background":
        st.header("Objective")
        st.write("This machine learning created as an alteration series . Using Natural Language Process and Data Augmented .")
        st.write("It include about how we imagined our self on the other side based on positivity effect .")
        st.header("Subjective")
        st.write("This machine learning created as an alteration series . Using Linear Regression and Data augmented .")
        st.write("It include about how we imagined our self on the other side based on positivity effect .")
        st.header("Flowchart")
        components.html("""<img src="https://raw.githubusercontent.com/RFebrians/RFebrians/main/assets/Kaggle/Alteration-Prevention.png" alt="flowchart">""",height=900,width=600)


        st.subheader('Technologies stack')
        
        #components.iframe("https://docs.streamlit.io/en/latest")
        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)
        col7, col8, col9 = st.columns(3)
        with col1:
            st.text("""
            NLTK
            Natural Language Processing
            """)
            components.html("""<img src="https://www.w3schools.com/images/w3schools_green.jpg" alt="W3Schools.com">""")
        with col2:
            st.text("""
            Matplotlib 
            for Plotting
            """)
            components.html("""<img src="https://www.w3schools.com/images/w3schools_green.jpg" alt="W3Schools.com">""")
        with col3:
            st.text("""
            NumPy
            
            """)
            components.html("""<img src="https://www.w3schools.com/images/w3schools_green.jpg" alt="W3Schools.com">""")
        with col4:
            st.text("""
            SciKit-Learn
            
            """)
            components.html("""<img src="https://www.w3schools.com/images/w3schools_green.jpg" alt="W3Schools.com">""")
        with col5:
            st.text("""
            Pandas
            
            """)
            components.html("""<img src="https://www.w3schools.com/images/w3schools_green.jpg" alt="W3Schools.com">""")
        with col6:
            st.text("""
            Streamlit
            
            """)
            components.html("""<img src="https://www.w3schools.com/images/w3schools_green.jpg" alt="W3Schools.com">""")
    
    
    if nav == "Contact":
        st.header("About Developer")
        st.write("Rizki Febriansyah is an undergraduate student with a strong passion on technologies , He does some research on college and work at Open Source Software as React Developer .")
        st.write("He has experience as React Developer and Machine Learning Researcher . Determined to continue moving forward , devoted with some promise to get a brighter tomorrow.")
        st.header("Contact")
        st.markdown("""
        **Mail:** ekikz1997@gmail.com\n
        **Phone:** (+62) 853-1232-9672\n
        **Linkedin:** https://www.linkedin.com/in/rizki-febriansyah97\n
        **GitHub:** https://github.com/RFebrians\n
        **Google Dev:** https://g.dev/zegveld\n
        """)
        
