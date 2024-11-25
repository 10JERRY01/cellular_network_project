# cellular_network_project
Aim to analyze and optimize cellular network performance by understanding relationships among various metrics, predicting signal strength, and identifying factors that impact user experience.
Project Documentation: Network Performance Optimization Analysis
1. Project Overview
This project focuses on optimizing network performance using predictive analytics. The goal is to predict key network metrics, enabling better resource allocation and improving the user experience in cellular networks. The project leverages a dataset with features such as Signal Strength (dBm), Distance to Tower (km), SNR, Attenuation, and categorical factors like Environment, Call Type, and Incoming/Outgoing.

The final deliverable is a deployed Streamlit application where users can input network conditions and receive predictions for network performance metrics.

2. Dataset Description
Source
The dataset was sourced from Kaggle: Cellular Network Performance Data.

Columns
Column Name	Description	Unit
Timestamp	Timestamp of the observation in the format YYYY-MM-DD HH:MM:SS.	DateTime (object)
Signal Strength (dBm)	Measures the intensity of the received signal, ranging from -50 dBm (strongest) to -120 dBm (weakest).	dBm
SNR	Signal-to-Noise Ratio: A measure of signal clarity, representing the ratio of signal power to noise power.	Unitless (ratio)
Call Duration (s)	Duration of the call.	Seconds
Environment	Describes the surrounding environment: home, urban, or open.	Categorical
Attenuation	Measures signal loss due to distance, interference, or obstacles.	Decibels (dB)
Distance to Tower (km)	Distance of the mobile device to the nearest cellular tower.	Kilometers
Tower ID	Unique identifier for each tower.	Categorical (ID)
User ID	Unique identifier for each user.	Categorical (ID)
Call Type	Indicates the type of call: data or voice.	Categorical
Incoming/Outgoing	Indicates the call direction: incoming or outgoing.	Categorical
3. Methodology
3.1. Preprocessing
Datetime Conversion: Converted the Timestamp column to a datetime format.
Categorical Encoding: One-hot encoding was applied to categorical features (Environment, Call Type, Incoming/Outgoing).
Feature Scaling: Scaled numerical features (e.g., Signal Strength, Distance to Tower, SNR) for compatibility with models like MLP Regressor.
3.2. Model Implementation
The following models were implemented:

Random Forest Regressor
Metrics for Evaluation
Mean Squared Error (MSE): Measures the average squared difference between predicted and actual values.


4. Streamlit Application
Features
User Inputs:
Signal Strength (slider)
Distance to Tower (slider)
SNR (slider)
Attenuation (slider)
Categorical options for Environment, Call Type, and Incoming/Outgoing.
Predictions: Outputs the predicted network metric based on user inputs.
Deployment
The application was deployed using Streamlit Cloud, enabling user interaction and visualization of predictions in real-time.

5. Results and Insights
Key Insights
Signal Strength and Distance: Strong negative correlation; the farther the device from the tower, the weaker the signal.
SNR and Call Quality: High SNR values correlate with better call quality and stability.
Attenuation: Significant contributor to signal loss, heavily influenced by environmental conditions.
Business Impact
Enhanced network coverage planning by predicting signal reliability under varying conditions.
Improved resource allocation by identifying areas with frequent weak signals.
6. Tools and Technologies
Programming Language: Python
Libraries:
Data Analysis: pandas, numpy
Modeling: sklearn, catboost
Visualization: matplotlib, seaborn
App Development: streamlit
Deployment: Streamlit Cloud
7. Future Work
Real-Time Data: Integrate real-time network data for dynamic predictions.
Scalability: Extend the app to analyze larger datasets with distributed computing platforms.
8. Appendix
References
Kaggle Dataset: Cellular Network Performance Data.
Python Libraries: Streamlit Documentation
