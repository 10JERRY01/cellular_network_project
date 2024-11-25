import pandas as pd
import numpy as np

# Load dataset
file_path = "train.csv"  # replace with the actual path to the downloaded file
data = pd.read_csv(file_path)

# Check the first few rows of the dataset
print(data.head())

# Overview of dataset
print(data.info())

# Summary statistics
print(data.describe())


# Check for missing values
print(data.isnull().sum())

# Fill or drop missing values (example: fill NaN in 'SNR' with mean)
data['SNR']=data['SNR'].mean()
# Convert categorical data to consistent format
data['Environment'] = data['Environment'].str.lower()

#Exploratory Data Analysis (EDA)
#Signal Strength vs Distance
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['Distance to Tower (km)'], y=data['Signal Strength (dBm)'], hue=data['Environment'])
plt.title('Signal Strength vs Distance to Tower')
plt.xlabel('Distance to Tower (km)')
plt.ylabel('Signal Strength (dBm)')
plt.show()

#SNR Across Environments

sns.boxplot(x=data['Environment'], y=data['SNR'])
plt.title('SNR Distribution by Environment')
plt.show()

#Feature Engineering
# Convert Timestamp to datetime and extract new features
data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')
data['Hour'] = data['Timestamp'].dt.hour
data['Day'] = data['Timestamp'].dt.day

# Derive Signal Quality
data['Signal Quality'] = data['Signal Strength (dBm)'] + data['SNR']

from sklearn.preprocessing import LabelEncoder

# Initialize label encoder
le = LabelEncoder()

# Apply to columns
data['Environment'] = le.fit_transform(data['Environment'])
data['Call Type'] = le.fit_transform(data['Call Type'])
data['Incoming/Outgoing'] = le.fit_transform(data['Incoming/Outgoing'])

print(data)

#Build Predictive Model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Select features and target
features = ['Distance to Tower (km)', 'SNR', 'Attenuation']
X = data[features]
y = data['Signal Strength (dBm)']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))


#Deploy and Visualize Insights
import streamlit as st

# Categorical feature options
environment_options = ["home", "urban", "open"]
call_type_options = ["data", "voice"]
incoming_outgoing_options = ["incoming", "outgoing"]

# Streamlit app layout
st.title("Cellular Network Performance Prediction")

# User input for categorical features
environment = st.selectbox("Select Environment", environment_options)
call_type = st.selectbox("Select Call Type", call_type_options)
incoming_outgoing = st.selectbox("Select Call Direction", incoming_outgoing_options)

# Display the user-selected values
st.write("Selected Features:")
st.write(f"Environment: {environment}")
st.write(f"Call Type: {call_type}")
st.write(f"Incoming/Outgoing: {incoming_outgoing}")

st.title("Cellular Network Performance Analysis")
st.write("Explore network insights and predictions.")

distance = st.slider("Distance to Tower (km)", min_value=0.0, max_value=10.0, step=0.1)
snr = st.slider("SNR", min_value=0.0, max_value=50.0, step=0.1)
attenuation = st.slider("Attenuation (dB)", min_value=0.0, max_value=100.0, step=1.0)

# Encoding categorical features
feature_map = {
    "Environment": {"home": 0, "urban": 1, "open": 2},
    "Call Type": {"data": 0, "voice": 1},
    "Incoming/Outgoing": {"incoming": 0, "outgoing": 1},
}

# Encode user inputs
encoded_environment = feature_map["Environment"][environment]
encoded_call_type = feature_map["Call Type"][call_type]
encoded_incoming_outgoing = feature_map["Incoming/Outgoing"][incoming_outgoing]

# Display encoded values
st.write("Encoded Features:")
st.write(f"Environment: {encoded_environment}")
st.write(f"Call Type: {encoded_call_type}")
st.write(f"Incoming/Outgoing: {encoded_incoming_outgoing}")

# Example: Collect all features in a DataFrame
features = {
    "Environment": [encoded_environment],
    "Call Type": [encoded_call_type],
    "Incoming/Outgoing": [encoded_incoming_outgoing],
    "Signal Strength (dBm)": [-80],  # Example input
    "Distance to Tower (km)": [0.5],  # Example input
    "SNR": [30],  # Example input
    "Attenuation": [5],  # Example input
}

input_df = pd.DataFrame(features)
st.write("Input Features for Prediction:")
st.write(input_df)


# Prediction
new_data = pd.DataFrame({'Distance to Tower (km)': [distance], 'SNR': [snr], 'Attenuation': [attenuation]})
predicted_signal = model.predict(new_data)[0]
st.write("Predicted Signal Strength (dBm):", predicted_signal)

