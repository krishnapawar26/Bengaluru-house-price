import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Bengaluru House Price Predictor", page_icon="🏠")
st.title("🏠 Krishna's Real Estate AI")
st.write("Enter the details below to get an estimated price.")

# --- 2. LOAD & CLEAN DATA (Cached for Speed) ---
@st.cache_data
def load_and_train_model():
    # Load Data
    df = pd.read_csv("Bengaluru_House_Data.csv")
    df = df[['location', 'size', 'total_sqft', 'bath', 'price']]
    
    # Cleaning Logic (Same as before)
    
    df = df.dropna()
    
    def clean_bhk(x):
        try:
            return int(x.split(' ')[0])
        except:
            return None
            
    def convert_sqft_to_num(x):
        tokens = x.split('-')
        if len(tokens) == 2:
            return (float(tokens[0]) + float(tokens[1])) / 2
        try:
            return float(x)
        except:
            return None

    df['bhk'] = df['size'].apply(clean_bhk)
    df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)
    df = df.dropna()

    # Location Grouping
    df['location'] = df['location'].apply(lambda x: x.strip())
    location_stats = df['location'].value_counts()
    location_stats_less_than_10 = location_stats[location_stats <= 10]
    df['location'] = df['location'].apply(lambda x: 'other' if x in location_stats_less_than_10 else x)

    # Train Model
    dummies = pd.get_dummies(df.location)
    df_final = pd.concat([df, dummies.drop('other', axis='columns')], axis='columns')
    
    X = df_final.drop(['price', 'location', 'size'], axis='columns')
    y = df_final.price
    
    model = LinearRegression()
    model.fit(X, y)
    
    return model, X.columns, df['location'].sort_values().unique()

# Load the model and column names
model, columns, location_options = load_and_train_model()

# --- 3. USER INPUTS (The Website Part) ---
col1, col2 = st.columns(2)

with col1:
    location = st.selectbox("Select Location", location_options)
    sqft = st.number_input("Total Sqft Area", min_value=300, max_value=50000, value=1000)

with col2:
    bhk = st.number_input("BHK (Bedrooms)", min_value=1, max_value=20, value=2)
    bath = st.number_input("Bathrooms", min_value=1, max_value=20, value=2)

# --- 4. PREDICTION LOGIC ---
if st.button("Estimate Price", type="primary"):
    # Prepare input array
    loc_index = -1
    if location in columns:
        loc_index = np.where(columns == location)[0][0]

    x = np.zeros(len(columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    # Predict
    price = model.predict([x])[0]
    
    # Show Result
    if price < 0:
        st.error("Error: The inputs are unrealistic for this model.")
    else:
        st.success(f"Estimated Price: ₹ {price:.2f} Lakhs")
        st.balloons()