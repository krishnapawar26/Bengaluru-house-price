import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# --- PAGE CONFIG ---
st.set_page_config(page_title="Bengaluru House Price AI", page_icon="🏠", layout="wide")

# --- CUSTOM CSS (For a Pro Look) ---
st.markdown("""
<style>
    .big-font { font-size:20px !important; color: #555; }
    .metric-card { background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.title("🏠 Bengaluru Real Estate AI Dashboard")
st.markdown('<p class="big-font">Predict prices and analyze market trends instantly.</p>', unsafe_allow_html=True)

# --- LOAD DATA & TRAIN MODEL ---
@st.cache_data
def load_data():
    df = pd.read_csv("Bengaluru_House_Data.csv")
    # ... (Keep your cleaning logic exactly same as before) ...
    df = df[['location', 'size', 'total_sqft', 'bath', 'price']]
    df = df.dropna()
    
    def clean_bhk(x):
        try: return int(x.split(' ')[0])
        except: return None
    
    def convert_sqft(x):
        tokens = x.split('-')
        if len(tokens) == 2: return (float(tokens[0]) + float(tokens[1])) / 2
        try: return float(x)
        except: return None

    df['bhk'] = df['size'].apply(clean_bhk)
    df['total_sqft'] = df['total_sqft'].apply(convert_sqft)
    df = df.dropna()
    
    # Remove basic outliers
    df = df[~(df.total_sqft/df.bhk < 300)]
    
    # Location Grouping
    df.location = df.location.apply(lambda x: x.strip())
    location_stats = df['location'].value_counts()
    location_stats_less_than_10 = location_stats[location_stats<=10]
    df.location = df.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
    
    return df

df = load_data()

# Train Model
dummies = pd.get_dummies(df.location)
df_final = pd.concat([df, dummies.drop('other', axis='columns')], axis='columns')
X = df_final.drop(['price', 'location', 'size'], axis='columns')
y = df_final.price
model = LinearRegression()
model.fit(X, y)

# --- SIDEBAR INPUTS ---
st.sidebar.header("🔍 Property Details")
location = st.sidebar.selectbox("Select Location", df['location'].unique())
bhk = st.sidebar.slider("BHK (Bedrooms)", 1, 10, 2)
bath = st.sidebar.slider("Bathrooms", 1, 10, 2)
sqft = st.sidebar.number_input("Total Sqft", 300, 20000, 1200)

# --- PREDICTION LOGIC ---
if st.sidebar.button("Predict Price", type="primary"):
    # 1. Predict
    loc_index = np.where(X.columns == location)[0][0] if location in X.columns else -1
    x = np.zeros(len(X.columns))
    x[0] = sqft; x[1] = bath; x[2] = bhk
    if loc_index >= 0: x[loc_index] = 1
    
    price = model.predict([x])[0]
    
    # 2. Display Result (Main Area)
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"### 💰 Estimated Price")
        st.success(f"₹ {price:.2f} Lakhs")
        price_per_sqft = (price * 100000) / sqft
        st.info(f"Price/Sqft: ₹ {price_per_sqft:.0f}")

    with col2:
        # 3. THE NEXT LEVEL CHART: "Where do I stand?"
        st.markdown(f"### 📊 Market Trends in {location}")
        
        # Filter data for this location only
        loc_df = df[df['location'] == location]
        
        if len(loc_df) > 5:
            fig, ax = plt.subplots(figsize=(8, 4))
            # Plot all houses in this location
            sns.scatterplot(x=loc_df['total_sqft'], y=loc_df['price'], ax=ax, color="blue", alpha=0.3, label="Other Properties")
            
            # Plot the USER'S prediction (Red Star)
            ax.scatter(x=sqft, y=price, color='red', s=200, marker='*', label='Your Prediction')
            
            ax.set_xlabel("Total Sqft")
            ax.set_ylabel("Price (Lakhs)")
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning("Not enough data to show market trends for this location.")
else:
    st.info("👈 Adjust the sidebar options and click Predict!")
