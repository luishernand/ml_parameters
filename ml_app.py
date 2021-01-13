# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 14:56:11 2021

@author: Luis Hernandez
"""

# Libraries#
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.datasets import load_boston

#-------------------------------------------#

# Paget layout expand to full with
st.set_page_config(page_title = 'The machine learning app', layout='wide')

#-------------------------------------------#
# model build

def build_model(df):
    X = df.iloc[:, : -1]
    y = df.iloc[:, -1]
    st.markdown('**1.2 Data split**')
    st.write('Traning set')
    st.info(X.shape)
    st.write('target set')
    st.info(y.shape)
    
    st.markdown('**1.3 variable details**')
    st.write('X variable')
    st.info(X.columns.tolist())
    st.write('y variable')
    st.info(y.name)
    
    # Data Sokitting
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = split_size)
    
    # model
    rf = RandomForestRegressor(n_estimators= parameter_n_estimator,
        max_features=parameter_max_feature,
        criterion= parameter_criterion,
        min_samples_leaf= parameter_sample_leaf,
        min_samples_split= parameter_sample_split,
        bootstrap= parameter_bootstrap,
        oob_score= parameter_oob_score,
        n_jobs = parameter_n_jobs)
    rf.fit(X_train, y_train)
    
    st.subheader('2. Model Performace')
    st.markdown('** 2.1 Traning Set')
    y_pred_train = rf.predict(X_train)
    st.write('Coeficient of Determination ($r2$)')
    st.info(r2_score(y_train, y_pred_train))
    
    st.markdown('** 2.2 Test Set')
    y_pred_test = rf.predict(X_test)
    st.write('Coeficient of Determination ($r2$)')
    st.info(r2_score(y_test, y_pred_test))
    
    
    st.write('Error (MSE or MAE)')
    st.info(mean_squared_error(y_test, y_pred_test))
    
    st.subheader(rf.get_params())
    
#----------------------------------------------------#
#Title

st.write('''
         The Machine Learning App
In this implementation, the *RandomForestRegressor()* function is used in this app for build a regression model using the **Random Forest** algorithm.
Try adjusting the hyperparameters!
''')
    
#----------------------------------------------------#

#sidebars
with st.sidebar.header('1. Upload yor CVS Data'):
    upload_file = st.sidebar.file_uploader('upload your input csv file', type = ['csv'])
    st.sidebar.markdown('''
                        [Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
                        ''')
                        
# Sidebar specific settings split size
with st.sidebar.header('2. set parameters'):
    split_size = st.sidebar.slider(' Data split Ratio (% for training set)', 10, 90, 75, 5)
    
with st.sidebar.subheader('2.1 Learning Parameters'):
    parameter_n_estimator = st.sidebar.slider('Number of estimator', 0, 1000, 100, 100)
    parameter_max_feature = st.sidebar.select_slider('Max Features', options = ['auto', 'sqrt', 'log2'])
    parameter_sample_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2,1)
    parameter_sample_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
    
    
    
with st.sidebar.subheader('2.2 General Parameters'):
    parameter_criterion = st.sidebar.select_slider("Select meassure criterion", options = ['mse', 'mae'])
    parameter_bootstrap = st.sidebar.select_slider('Select boostrap Samples', options = [True, False])
    parameter_oob_score = st.sidebar.select_slider('Oob Score', options = [True, False])
    parameter_n_jobs = st.sidebar.select_slider('Select n_jobs', options = [1, -1])
    

#--------------------------------------------------------------------------------------#
# Main panel
## Display dataset

st.subheader('1. Dataset')
if upload_file  is not None:
    df = pd.read_csv(upload_file, index = False)
    st.markdown('**1.1 Glimpse  of dataset')
    st.write(df)
    build_model(df)
else:
    st.info('Awaiting for csv file uploaded')
    if st.button('Press to use exmaple dataset'):
        boston = load_boston()
        X = pd.DataFrame(boston.data, columns = boston.feature_names)
        y = pd.Series(boston.target, name = 'response')
        df = pd.concat([X, y], axis = 1)
        
        st.markdown('The Boston housing dataset is used as example')
        st.write(df.head())
        build_model(df)
    
    
    

    

