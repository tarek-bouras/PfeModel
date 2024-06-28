import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from keras.models import load_model
from datetime import datetime, timedelta

# Load the trained model
model = joblib.load('./lstm_model.pkl')

# Function to prepare the data
def prepare_data(df, time_steps=60):
    data = df['quantity'].values.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    x_test = []
    for i in range(time_steps, len(scaled_data)):
        x_test.append(scaled_data[i - time_steps:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    return x_test, scaler

# Function to forecast the next 60 days
def forecast(model, x_test, scaler, time_steps=60, future=60):
    forecast_data = x_test[-1]  # Use the last sequence of the test set for forecasting
    forecast_predictions = []
    for _ in range(future):
        prediction = model.predict(forecast_data.reshape(1, time_steps, 1))
        forecast_predictions.append(prediction[0, 0])
        forecast_data = np.append(forecast_data[1:], prediction[0, 0]).reshape(-1, 1)
    forecast_predictions = np.array(forecast_predictions).reshape(-1, 1)
    forecast_predictions = scaler.inverse_transform(forecast_predictions)
    
    return forecast_predictions

# Streamlit UI
st.title('Product Sales Forecasting')

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=['date'])
    st.write(df.tail())  # Display the tail of the dataframe

    family = st.selectbox("Select a family", df['family'].unique())

    if st.button('Predict'):
        df_family = df[df['family'] == family]
        
        # Ensure df_family is not empty
        if df_family.empty:
            st.write("No data available for the selected family.")
        else:
            # Prepare data
            x_test, scaler = prepare_data(df_family)
            
            # Forecast
            forecast_predictions = forecast(model, x_test, scaler)
            
            # Prepare forecast dataframe
            last_date = df_family['date'].max()
            forecast_dates = [last_date + timedelta(days=i) for i in range(1, 61)]
            forecast_df = pd.DataFrame({'date': forecast_dates, 'forecasted_quantity': forecast_predictions.flatten()})
            
            # Plot using Plotly with green line
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['forecasted_quantity'], mode='lines', name='Forecasted Quantity', line=dict(color='green')))
            fig.update_layout(title=f'Sales Forecast for {family}', xaxis_title='Date', yaxis_title='Quantity Sold')
            st.plotly_chart(fig)
            
            st.write(forecast_df)
