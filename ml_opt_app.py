# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 14:37:20 2021

@author: Luis Hernandez
"""
import streamlit as st
import pandas as pd
import numpy as np
import base64
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes

#----------------------------------------------------------------#
# page layout 
st.set_page_config(page_title='The Machine Learning Hyperparameter Optimization App',
    layout='wide')

#-------------------------------------------------------------------#

#Title

st.write('''
         # The Machine Learning Hyperparameter Optimization App  
**(Regression Edition)**  
In this implementation, the *RandomForestRegressor()* function is used in this app for build a regression model using the **Random Forest** algorithm.
''')

#-------------------------------------------------------------------#
#sidebar

st.sidebar.header('Upload your csv Data')
uploaded_file = st.sidebar.file_uploader('Upload yor CSV file', type = ['csv'])
st.sidebar.markdown('''
                    [Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
                    ''')

## Sidebar specifyct parameters
st.sidebar.header('Set parameters')
split_size = st.sidebar.slider('Data split Ratio (% for training set)', 10, 90,80,5)

st.sidebar.subheader('Learning parameters')
parameter_n_estimator = st.sidebar.slider('Number of estimator', 0, 500, (10,50), 50)
parameter_n_estimator_step = st.sidebar.number_input('Step size for n_estimator', 10)
st.write('---')

parameter_max_feature = st.sidebar.slider('Max Features', 1, 50, (1, 3), 1)
st.sidebar.number_input('Step size for max_feature', 1)
st.write('---')


parameter_sample_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2,1)
parameter_sample_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)

st.sidebar.subheader('General Parameters')
parameter_random_state = st.sidebar.slider('Seed Number(random_state)', 0, 1000, 42, 1)
parameter_criterion = st.sidebar.select_slider("Select meassure criterion", options = ['mse', 'mae'])
parameter_bootstrap = st.sidebar.select_slider('Select boostrap Samples', options = [True, False])
parameter_oob_score = st.sidebar.select_slider('Oob Score', options = [True, False])
parameter_n_jobs = st.sidebar.select_slider('Select n_jobs', options = [1, -1])


n_estimators_range = np.arange(parameter_n_estimator[0], parameter_n_estimator[1]+parameter_n_estimator_step, parameter_n_estimator_step)
max_features_range = np.arange(parameter_max_feature[0], parameter_max_feature[1]+1, 1)
param_grid = dict(max_features=max_features_range, n_estimators=n_estimators_range)

#------------------------------------------------------------------------------------#
#Main panel
# Display the dataset
st.subheader('Dataset')


#------------------------------------------------------------------------------------#

# file download
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="model_performance.csv">Download CSV File</a>'
    return href

# Model Build
def build_model(df):
    X = df.iloc[:, : -1]
    y = df.iloc[:, -1]
    st.markdown('A model is being built to predict following **y** variable')
    st.info(y.name)
    
    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split_size)
    
    
    #model
    rf = RandomForestRegressor(n_estimators= parameter_n_estimator,
        max_features=parameter_max_feature,
        criterion= parameter_criterion,
        min_samples_leaf= parameter_sample_leaf,
        min_samples_split= parameter_sample_split,
        bootstrap= parameter_bootstrap,
        oob_score= parameter_oob_score,
        n_jobs = parameter_n_jobs)
    
    
    #Optimize parameters GridCV
    grid = GridSearchCV(estimator = rf, param_grid= param_grid, cv = 5)
    grid.fit(X_train, y_train)
    
    st.subheader('Model performace')
    y_pred = grid.predict(X_test)
    
    st.write('Coeficient of determiantion ($R^2$):')
    st.info(r2_score( y_test, y_pred))
    
    st.write('Error (MSE or MAE):')
    st.info(mean_squared_error( y_test, y_pred))
    
    st.write("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
    
    st.subheader('Model Parameters')
    st.write(grid.get_params())
    
    
    #-----Process grid data-----#
    grid_results = pd.concat([pd.DataFrame(grid.cv_results_["params"]),pd.DataFrame(grid.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
    # Segment data into groups based on the 2 hyperparameters
    grid_contour = grid_results.groupby(['max_features','n_estimators']).mean()
    # Pivoting the data
    grid_reset = grid_contour.reset_index()
    grid_reset.columns = ['max_features', 'n_estimators', 'Accuracy']
    grid_pivot = grid_reset.pivot('max_features', 'n_estimators')
    x = grid_pivot.columns.levels[1].values
    y = grid_pivot.index.values
    z = grid_pivot.values

    #-----Plot-----#
    layout = go.Layout(
            xaxis=go.layout.XAxis(
              title=go.layout.xaxis.Title(
              text='n_estimators')
             ),
             yaxis=go.layout.YAxis(
              title=go.layout.yaxis.Title(
              text='max_features')
            ) )
    fig = go.Figure(data= [go.Surface(z=z, y=y, x=x)], layout=layout )
    fig.update_layout(title='Hyperparameter tuning',
                      scene = dict(
                        xaxis_title='n_estimators',
                        yaxis_title='max_features',
                        zaxis_title='Accuracy'),
                      autosize=False,
                      width=800, height=800,
                      margin=dict(l=65, r=50, b=65, t=90))
    st.plotly_chart(fig)

    #-----Save grid data-----#
    x = pd.DataFrame(x)
    y = pd.DataFrame(y)
    z = pd.DataFrame(z)
    df = pd.concat([x,y,z], axis=1)
    st.markdown(filedownload(grid_results), unsafe_allow_html=True)
    
#------------------------------------------------------------------------------------#
# file upload
if uploaded_file  is not None:
    df = pd.read_csv(uploaded_file, index = False)
    st.write(df)
    build_model(df)
else:
    st.info('Awaiting for csv file uploaded')
    if st.button('Press to use exmaple dataset'):
        diabetes = load_diabetes()
        X = pd.DataFrame(diabetes.data, columns = diabetes.feature_names)
        y = pd.Series(diabetes.target, name = 'response')
        df = pd.concat([X, y], axis = 1)
        
        st.markdown('The Boston housing dataset is used as example')
        st.write(df.head())
        build_model(df)



 


