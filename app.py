import numpy as np
import pandas as pd
import pickle
import streamlit as st

# Load the trained model
pickle_in = open("classifier.pkl", "rb")
classifier = pickle.load(pickle_in)

def main():
    st.title("Bank Authenticator")
    
    # Input fields for user data
    age = st.number_input("Age", min_value=1, max_value=120, value=30)

    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Bank Authenticator ML App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    # Collecting user input
    variance = st.text_input("Variance", "Type Here")
    skewness = st.text_input("Skewness", "Type Here")
    curtosis = st.text_input("Curtosis", "Type Here")
    entropy = st.text_input("Entropy", "Type Here")

    result = ""
    if st.button("Predict"):

        input_data = pd.DataFrame(
             [[float(variance), float(skewness), float(curtosis), float(entropy)]],
              columns=['Variance_Wavelet', "Skewness_Wavelet", "Curtosis_Wavelet", "Image_Entropy"])

        prediction = classifier.predict(input_data)
        
        # Display the result
        if prediction[0] == 0:
            result = "The bank note is authentic."
        else:
            result = "The bank note is not authentic."

    if st.button("About"):
        st.text("This is a simple ML app to predict whether a bank note is authentic or not.")
    st.success(result)


if __name__ == '__main__':
    main()
    

