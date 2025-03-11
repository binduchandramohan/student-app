import streamlit as st 
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder


def load_model():
    # to read the physical pickle file , open it in read binary mode
    with  open("student_lr_final_model.pkl",'rb') as file:
        # file is going to return 3 things... as we stored 3 things
        # save them on 3 different variables..
        model,scaler,le=pickle.load(file)
    return model,scaler,le

def preprocesssing_input_data(data, scaler, le):
    # user wouldnt know what kind of transformation is being done within the code
    # so we need to do the same kind of transformation that we did in the pickle file 
    # this function will take data , scaler object & le object
    # it will use label encoder and transform the column Extracurricular Activities
    data['Extracurricular Activities']= le.transform([data['Extracurricular Activities']])[0]
    df = pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed

def predict_data(data): # data is the input dataset
    # predict function will take data and predict the y value
    # we are calling load_model here which will return 3 values
    model,scaler,le = load_model()
    # then we preprocess the values using the other function
    processed_data = preprocesssing_input_data(data,scaler,le)
    # prediction using the processed_data
    prediction = model.predict(processed_data)
    # returns the prediction..
    return prediction

# where to use this function --> predict_data ? 
# in UI , lets code for that part
def main():
    st.title("student performnce perdiction")
    st.write("enter your data to get a prediction for your performance")
    
    hour_sutdied = st.number_input("Hours studied",min_value = 1, max_value = 10 , value = 5)
    prvious_score = st.number_input("previous score",min_value = 40, max_value = 100 , value = 70)
    extra = st.selectbox("extra curri activity" , ['Yes',"No"])
    sleeping_hour = st.number_input("sleeping hours",min_value = 4, max_value = 10 , value = 7)
    number_of_peper_solved = st.number_input("number of question paper solved",min_value = 0, max_value = 10 , value = 5)
    
    # on the click of a button , we need to predict the output..
    if st.button("predict-your_score"):
        user_data = {
            # map the original column name with the variables we created for UI
            # in a key value pair..
            "Hours Studied":hour_sutdied,
            "Previous Scores":prvious_score,
            "Extracurricular Activities":extra,
            "Sleep Hours":sleeping_hour,
            "Sample Question Papers Practiced":number_of_peper_solved
        }
        prediction = predict_data(user_data)
        st.success(f"your prediciotn result is {prediction}")
    
if __name__ == "__main__":
    main()

# run the app using streamlit run .\stud-per.py
# opens http://localhost:8501/ , and we are able to predict the score..
# choose env as 3.9.13 on right bottom in vs code..
