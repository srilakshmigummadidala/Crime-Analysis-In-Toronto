import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Import the dataset
df = pd.read_csv('Major_Crime_Indicators_Open_Data.csv')

# Display the number of missing values for each column
print("Number of missing values for each column:")
print(df.isnull().sum())

# Drop rows with missing values
df = df.dropna()

# Dropping columns which are not relevant and duplicate information with other columns

# OFFENSE column is having duplicate information from MCI_CATEGORY column which is our target Variable
# LOCATION_TYPE is having duplicate information from PREMISES_TYPE.
# NEIGHBOURHOOD_158, NEIGHBOURHOOD_140 have same labeling information in HOOD_158,HOOD_140

df = df.drop(columns=['X', 'Y', 'EVENT_UNIQUE_ID', 'REPORT_DATE', 'OCC_DATE', 'OBJECTID', 'UCR_CODE', 'UCR_EXT',
                      'LOCATION_TYPE', 'OFFENCE', 'NEIGHBOURHOOD_158', 'NEIGHBOURHOOD_140', 'LONG_WGS84',
                      'LAT_WGS84'], axis=1)

# Filter the DataFrame to remove data from years 2003 to 2013
df = df[df['OCC_YEAR'] >= 2014]

# Define mappings for month names and days of the week
month_mapping = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}

dow_mapping = {
    'Monday    ': 1, 'Tuesday   ': 2, 'Wednesday ': 3, 'Thursday  ': 4,
    'Friday    ': 5, 'Saturday  ': 6, 'Sunday    ': 7
}

# Apply mappings to convert month names and days of the week to numerical values
df['REPORT_MONTH'] = df['REPORT_MONTH'].map(month_mapping)
df['OCC_MONTH'] = df['OCC_MONTH'].map(month_mapping)
df['OCC_DOW'] = df['OCC_DOW'].map(dow_mapping)
df['REPORT_DOW'] = df['REPORT_DOW'].map(dow_mapping)

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Apply LabelEncoder to 'DIVISION' column
df['DIVISION'] = label_encoder.fit_transform(df['DIVISION'])

# Define mappings for premises type and MCI category to numerical values
premises_mapping = {
    'Transit': 0, 'Commercial': 1, 'Outside': 2, 'House': 3, 'Apartment': 4, 'Educational': 5, 'Other': 6
}

MCI_mapping = {
    'Assault': 0, 'Break and Enter': 1, 'Auto Theft': 2, 'Robbery': 3, 'Theft Over': 4
}

# Apply mappings to convert premises type and MCI category to numerical values
df['PREMISES_TYPE'] = df['PREMISES_TYPE'].map(premises_mapping)
df['MCI_CATEGORY'] = df['MCI_CATEGORY'].map(MCI_mapping)

# Initialize OrdinalEncoder
encoder = OrdinalEncoder()

# Reshape the data to be a 2-dimensional array
report_year = df['REPORT_YEAR'].values.reshape(-1, 1)
occ_year = df['OCC_YEAR'].values.reshape(-1, 1)

# Fit and transform the data
encoded_report_year = encoder.fit_transform(report_year)
encoded_occ_year = encoder.fit_transform(occ_year)

# Replace the original REPORT_YEAR column with the encoded values
df['REPORT_YEAR'] = encoded_report_year
df['OCC_YEAR'] = encoded_occ_year

df['REPORT_YEAR'] = df['REPORT_YEAR'].astype(int)
df['OCC_YEAR'] = df['OCC_YEAR'].astype(int)
df['OCC_DAY'] = df['OCC_DAY'].astype(int)
df['OCC_DOY'] = df['OCC_DOY'].astype(int)

# Drop rows with 'NSA' values
df = df[(df['HOOD_158'] != 'NSA') & (df['HOOD_140'] != 'NSA')]

# Convert 'HOOD_158' to numeric (excluding 'NSA' values)
df['HOOD_158'] = pd.to_numeric(df['HOOD_158'])
df['HOOD_140'] = pd.to_numeric(df['HOOD_140'])

# Splitting the dataset into training and testing sets
X = df.drop(columns=['MCI_CATEGORY'])  # Features
y = df['MCI_CATEGORY']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=150)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Classification report
print(classification_report(y_test, y_pred))

# Save the trained model
pickle.dump(rf_classifier, open("rf_classifier.pkl", "wb"))
