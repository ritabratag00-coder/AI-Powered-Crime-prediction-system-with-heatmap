# AI-Powered-Crime-prediction-system-with-heatmap
This repository folder comprises of the project that works on crime prediction system with the help of heatmap ,
The application helps the user informed about the crime prediction analysis based on available recorded official data of crimes , with this users 
can assess the probabilistic assessment of the crime , it doesn't however shows the certainty of the crime that will be taking place .
# AI-Sentinels — Crime Prediction Dashboard

A Streamlit-based crime analytics and AI prediction dashboard for India.

# Setup

Download the zip file , extract it into a folder , in the address baer type cmd , then,
in command terminal type -

1.pip install -r requirements.txt
2.python -m streamlit run app.py

(Note :- Make sure you have installed required tools , pip install streamlit folium streamlit-folium plotly scikit-learn pandas numpy)











## Features
- Interactive heatmap on India (CartoDB dark tiles)
- Filters: city, hour, crime type, domain, gender, age, weapon
- AI Risk Predictor (Random Forest, 120 estimators)
- 6 analytics charts: hourly bar, crime type donut, city bar, weapon bar, age histogram, hour×weekday heatmap
- City risk index in sidebar
- Expandable raw data table
