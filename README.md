# Stock_Predictor_App
Built a Streamlit web app to predict and visualize stock prices using historical data. Integrated a Keras deep learning model, moving average analysis, and yfinance API for real-time data with interactive visualizations and UI enhancements.


Stock Market Predictor
Overview:
An interactive web app built with Streamlit and Keras, designed to visualize and predict stock prices using historical data and machine learning models.

Key Features:

1. Allows users to input stock symbols and custom date ranges

2. Downloads live stock data using Yahoo Finance (yfinance)

3. Uses a trained deep learning model (Keras) to predict future prices

4. Visualizes price trends and moving averages (50, 100, 200-day)

5. Compares predicted vs actual prices with intuitive plots

6. Styled with a custom logo and layout using HTML/CSS in Streamlit

Tech Stack:

- Python, Streamlit, Keras, NumPy, Pandas, Matplotlib, Seaborn

- yfinance (for stock data), scikit-learn (for scaling), PIL (for logo display)

How to Run:

Install required packages:
pip install streamlit yfinance keras matplotlib pandas pillow

Run the app:
streamlit run app.py
Ensure the model file Stock Prediction Model.keras is placed in the project directory.

Note:
You can also visualize moving average comparisons and model predictions via a dropdown.
