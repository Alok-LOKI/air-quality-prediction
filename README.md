# Air-Quality-Prediction

open code in google colab: 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]([https://colab.research.google.com/github/Anindya-Arnob/air-quality-prediction/blob/main/Airquality_pred.ipynb](https://colab.research.google.com/github/Alok-LOKI/air-quality-prediction/blob/main/Airquality_pred.ipynb))

Berlin Air Quality Prediction using LSTM: 
  This project provides an end-to-end machine learning pipeline to predict air quality levels (specifically PM2.5) in Berlin. It integrates cloud-based data storage with deep learning time-series forecasting.

Table of Contents: 
  1. Project Overview
  2. Key Features
  3. Tech Stack
  4. Dataset & Features
  5. Workflow
  6. Installation
  7. Model Performance

Project Overview: 
  Urban air quality monitoring is critical for public health. This project focuses on predicting PM2.5 concentrations—a key indicator of air pollution—by analyzing its relationship with other pollutants like PM10, Nitrogen Dioxide (NO2), and Ozone (O3). The system utilizes a Long Short-Term Memory (LSTM) network, which is uniquely suited for finding patterns in historical atmospheric data to forecast future conditions.

Key Features: 
  1. Automated Data Ingestion: Seamlessly connects to a Supabase PostgreSQL database to fetch live or historical sensor data.
  2. Robust Preprocessing Pipeline: Implements advanced data cleaning, including linear interpolation for missing values and temporal sorting.
  3. Time-Series Forecasting: Utilizes an LSTM architecture capable of capturing long-term dependencies in sequential data.
  4. Dynamic Visualization: Uses Plotly to generate interactive charts comparing predicted air quality against ground-truth measurements.

Tech Stack: 
  1. Data Source: Supabase (PostgreSQL).
  2. Data Science: Python, Pandas, NumPy, Scikit-Learn.
  3. Deep Learning: TensorFlow, Keras (LSTM).
  4. Visualization: Plotly, Matplotlib.

Dataset & Features: 
  1. The model utilizes a multi-variate dataset consisting of the following features:
  2. pm25 (Target Variable): Fine particulate matter concentrations (µg/m³).
  3. pm10: Coarse particulate matter.
  4. no2: Nitrogen Dioxide levels.
  5. o3: Ozone levels.
Note: Carbon Monoxide (co) was initially present but dropped during feature selection to optimize model performance.

Workflow: 
1. Data Retrieval from Supabase: 
  The system uses the supabase-py client to query the berlin table. It fetches up to 4,000 rows of data across all pollutant columns, converting the raw JSON response into a structured Pandas DataFrame for analysis.
2. Detailed Data Processing: 
  Before training, the data undergoes a rigorous transformation process:
Cleaning: 
  1. Column names are stripped of leading/trailing whitespaces (e.g., converting " pm25" to "pm25").
  2. Type Conversion: Non-numeric markers (like '-') are replaced with NaN, and all pollutant columns are converted to numeric floats.
  3. Missing Value Imputation: * Linear Interpolation: Used to fill gaps between known data points.
  4. Boundary Filling: ffill (forward fill) and bfill (backward fill) ensure no NaN values remain at the start or end of the series.
  5. Temporal Sorting: Data is sorted chronologically by date and set as a DatetimeIndex.
  6. Feature Scaling: All features are normalized between 0 and 1 using MinMaxScaler to ensure stable gradients during neural network training.



4. Machine Learning (LSTM Model): 
The forecasting is handled by a sequential deep learning model: 
Architecture: 
  A stacked LSTM layer configuration designed to process sequences of historical data.
Data Preparation: 
  The time-series data is converted into a supervised learning format where a specific window of past days is used to predict the next day's PM2.5 level.
Training Strategy: 
  * Optimizer: Adam optimizer for efficient weight updates.
Loss Function: 
  Mean Squared Error (MSE).
Early Stopping: 
  A callback monitors validation loss and stops training once performance plateaus to prevent overfitting.



Installation: 
  To set up the environment, install the following packages:
Bash: 
  "pip install pandas numpy scikit-learn tensorflow supabase plotly Supabase Configuration"

  

Ensure your credentials are set in the notebook to enable data fetching: 
Python: 
  url = "YOUR_SUPABASE_URL"
  key = "YOUR_SUPABASE_KEY"

  

Model Performance: 
  The performance is evaluated by comparing the model's predictions against a withheld test set. The results are visualized through an interactive Plotly line chart, showing the overlap between "Actual PM2.5" and   "Predicted PM2.5". This visual validation, combined with MSE loss tracking, confirms the model's ability to follow Berlin's air quality trends.



License: 
  This project is open-source. For any inquiries or contributions, please open an issue in the repository.
