import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# Load the Flask app
app = Flask(__name__)

# Load the trained model
model = pickle.load(open('rf_classifier.pkl', 'rb'))

# Define mappings for month names and days of the week
month_mapping = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}

dow_mapping = {
    'Monday    ': 1, 'Tuesday   ': 2, 'Wednesday ': 3, 'Thursday  ': 4,
    'Friday    ': 5, 'Saturday  ': 6, 'Sunday    ': 7
}

# Define a mapping for crime category labels
label_mapping = {
    0: "Assault", 1: "Break and Enter", 2: "Auto Theft", 3: "Robbery", 4: "Theft Over"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input data from the form
    input_data = {
        'REPORT_YEAR': int(request.form['REPORT_YEAR']),
        'REPORT_MONTH': request.form['REPORT_MONTH'],
        'REPORT_DOY': int(request.form['REPORT_DOY']),
        'REPORT_DAY': int(request.form['REPORT_DAY']),
        'REPORT_DOW': dow_mapping.get(request.form['REPORT_DOW'], 0),
        'REPORT_HOUR': int(request.form['REPORT_HOUR']),
        'OCC_YEAR': int(request.form['OCC_YEAR']),
        'OCC_MONTH': request.form['OCC_MONTH'],
        'OCC_DAY': int(request.form['OCC_DAY']),
        'OCC_DOY': int(request.form['OCC_DOY']),
        'OCC_DOW': dow_mapping.get(request.form['OCC_DOW'], 0),
        'OCC_HOUR': int(request.form['OCC_HOUR']),
        'DIVISION': request.form['DIVISION'],
        'PREMISES_TYPE': request.form['PREMISES_TYPE'],
        'HOOD_158': int(request.form['HOOD_158']),
        'HOOD_140': int(request.form['HOOD_140'])
    }

    # Convert month names to numerical values
    input_data['REPORT_MONTH'] = month_mapping.get(input_data['REPORT_MONTH'], 0)
    input_data['OCC_MONTH'] = month_mapping.get(input_data['OCC_MONTH'], 0)

    # Create DataFrame from input data
    input_df = pd.DataFrame(input_data, index=[0])

    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Encode 'DIVISION' column
    input_df['DIVISION'] = label_encoder.fit_transform(input_df['DIVISION'])

    # Define a mapping dictionary for premises type to numerical values
    premises_mapping = {
        'Transit': 0, 'Commercial': 1, 'Outside': 2, 'House': 3, 'Apartment': 4, 'Educational': 5, 'Other': 6
    }

    # Apply the mapping to convert premises type to numerical values
    input_df['PREMISES_TYPE'] = input_df['PREMISES_TYPE'].map(premises_mapping)

    # Make predictions
    prediction = model.predict(input_df)

    # Map prediction to crime category
    predicted_label = label_mapping.get(prediction[0], "Unknown")

    return render_template('index.html', prediction_text=f'Prediction of Crime Category is {predicted_label}')

if __name__ == '__main__':
    app.run(debug=True)
