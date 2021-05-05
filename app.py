# Importing Libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle

#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#

# Page Layout
## Page explands to full width

# st.set_page_config(page_title='Titanic Survivability Prediction',
#     layout='wide')

#------------------------------------------------------------------------------#

# Model Building 
# load the model from disk

filename = r'finalized_model_rf.sav'
loaded_model_rf = pickle.load(open(filename, 'rb')) 

#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#

st.markdown("""
    <p style= 'font-size: 30px; color: tomato; text-align: center'>
        <b>TITANIC PASSENGERS SURVIVABILITY PREDICTION WEB APP</b>
    </p><hr>
"""
,unsafe_allow_html= True)

st.markdown("""
    <p style= 'font-size: 20px; color: blue; text-align: left'>
        The features provided in the <b>Side bar</b> should be filled and hit the <b>Predict</b> button to get the Prediction. 
    </p>

    <p style= 'font-size: 18px; color: green'>
        Please follow the instructions for Age  
        <p style= 'font-size: 15px'>Children: 0 to 15 years</p>
        <p style= 'font-size: 15px'>Young Aged: 16 to 30 years</p>
        <p style= 'font-size: 15px'>Middle Aged: 31 to 45 years</p>
        <p style= 'font-size: 15px'>Middle Aged: 31 to 45 years</p>
        <p style= 'font-size: 15px'>Senior Aged: 46 to 60 years</p>
        <p style= 'font-size: 15px'>Old: 60+ years</p>
    </p>


    """
    ,unsafe_allow_html= True)



with st.sidebar.subheader('Input Parameters'):
    Pclass = st.sidebar.selectbox('Pclass', [1,2,3])
    Sex = st.sidebar.selectbox('Sex', ["Male", "Female"])
    SibSip = st.sidebar.selectbox('Number of Siblings/Spouses abroaded', [0,1,2,3,4,5,8 ])
    Parch= st.sidebar.slider('Number of Parents/Children abroaded', 0, 6, 3, 1)
    Fare= st.sidebar.number_input('Fare in Dollar')
    Age = st.sidebar.selectbox('Age', ["Children", "Young Aged","Middle Aged","Senior Aged","Old"])
    Cabin = st.sidebar.selectbox('Availability of Cabin', ["Available", "Not Available"])
    Embark = st.sidebar.selectbox('Port of Embarkment', ["C", "Q","S"])

Family = SibSip + Parch + 1


#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#

#st.beta_container()
st.beta_columns(spec=1)
col1, col2 = st.beta_columns([3,1])
#col2.subheader('Columnisation')



if col2.button("Predict!"):
    pre_list = []

    pre_list.append(SibSip)
    pre_list.append(Parch)
    pre_list.append(Fare)
    pre_list.append(Family)

    if Pclass == 1:
        pre_list.append(0)
        pre_list.append(0)
    elif Pclass == 2:
        pre_list.append(1)
        pre_list.append(0)
    else:
        pre_list.append(0)
        pre_list.append(1)

    if Sex == "Male":
        pre_list.append(1)
    else:
        pre_list.append(0)

    if Age == "Children":
        pre_list.append(0)
        pre_list.append(0)
        pre_list.append(0)
        pre_list.append(0)
    elif Age == "Young Aged":
        pre_list.append(0)
        pre_list.append(0)
        pre_list.append(0)
        pre_list.append(1)
    elif Age == "Middle Aged":
        pre_list.append(1)
        pre_list.append(0)
        pre_list.append(0)
        pre_list.append(0)
    elif Age == "Senior Aged":
        pre_list.append(0)
        pre_list.append(0)
        pre_list.append(1)
        pre_list.append(0)
    elif Age == "Old":
        pre_list.append(0)
        pre_list.append(1)
        pre_list.append(0)
        pre_list.append(0)

    if Cabin == "Available":
        pre_list.append(0)
    else:
        pre_list.append(1)

    if Embark == "C":
        pre_list.append(0)
        pre_list.append(0)
    elif Embark == "S":
        pre_list.append(0)
        pre_list.append(1)
    else:
        pre_list.append(1)
        pre_list.append(0)

    pre = loaded_model_rf.predict(np.array(pre_list).reshape(1,-1))
    if pre == 1:
        st.success("The Passenger will Survive")
    else:
        st.warning("The Passenger will not Survive")
    

    

    




