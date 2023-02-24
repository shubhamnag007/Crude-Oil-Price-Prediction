import streamlit as st
import datetime
import pandas as pd
import numpy as np
import pickle
import darts
import torch
from darts import TimeSeries
import matplotlib.pyplot as plt

# Load the pickled model
with open('model5.pkl', 'rb') as file:
    model = pickle.load(file)

# Create a function to make predictions
def predict(num_days):
    # Make a prediction using the model and the input data
    Price = model.predict(num_days)
    return Price

# Define the Streamlit app
def app():
    # This is the tiltle of the App

    
    st.title("Crude oil Price Prediction")
    
    st.write("The following model app is based on facebook prophet Model where it is trained on crude oil price data from  1986-01-02 to 2020-12-30 using darts library functions.You can make predictions for any date after 2020-12-30 and the accuracy of the prediction will fall with increasing dates away from 2020-12-30")  
    
    # Create a date picker widget for selecting the date
    # Define the base date
    base_date = datetime.date(2020, 12, 31)
    selected_date = st.date_input("Select a date", min_value=base_date)
    # Calculate the number of days between the selected date and the base date
    base_date = datetime.date(2020, 12, 31)
    delta = selected_date - base_date
    num_days = delta.days + 1
    # Create a button widget to trigger the prediction
    if st.button("Predict"):
        # Make a prediction using your trained model and the number of days
        price = predict(num_days)
        Price = TimeSeries.pd_dataframe(price)
        Cost = Price.tail(1)
        # Display the resulting integer value
        st.write(f"The predicted price for crude oil on {selected_date} is given below")
        st.dataframe(Cost)
        st.write(f"The below table gives crude oil price from 2020-12-31 to {selected_date}")
        #st.dataframe(Price)
        #st.dataframe(Price.style.highlight_max(axis=0))
        #st.dataframe(Price.style.highlight_max(axis=1))
        st.dataframe(Price, use_container_width=True)
        #st.table(Price)



if __name__ == '__main__':
    app()
