import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# File paths
train_file = '/Users/nithinreddy/Desktop/internship hemachandra/Credit_Card_Fraud_Detection/Credit_Card_Fraud_Detection_Datasets/fraudTrain.csv'
test_file = '/Users/nithinreddy/Desktop/internship hemachandra/Credit_Card_Fraud_Detection/Credit_Card_Fraud_Detection_Datasets/fraudTest.csv'

# Load datasets (train and test)
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

# Limit datasets to 600 rows
train_df = train_df.sample(n=600, random_state=42)
test_df = test_df.sample(n=600, random_state=42)

# Drop non-numeric columns from both datasets
train_numeric = train_df.select_dtypes(include=[np.number])
test_numeric = test_df.select_dtypes(include=[np.number])

# Check if 'is_fraud' is in the columns
if 'is_fraud' not in train_numeric.columns:
    raise ValueError("The target column 'is_fraud' is missing in the training data.")

# Separate features and target for training and testing
X_train = train_numeric.drop(columns=['is_fraud'])
y_train = train_numeric['is_fraud']

X_test = test_numeric.drop(columns=['is_fraud'], errors='ignore')  # No target in test, so use all features

# Split training data further for validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train the model using Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_split, y_train_split)

# Predictions on validation set
val_predictions = model.predict(X_val_split)

# Classification report on validation set
print("Validation Report:\n", classification_report(y_val_split, val_predictions))

# Predict on the test set
test_predictions = model.predict(X_test)

# Save test predictions into the test dataframe
test_df['is_fraud'] = test_predictions

# Save fraudulent transactions from both train and test datasets
train_fraud = train_df[train_df['is_fraud'] == 1]
test_fraud = test_df[test_df['is_fraud'] == 1]

# Save the fraud details to CSV files
train_fraud.to_csv('fraudulent_transactions_train.csv', index=False)
test_fraud.to_csv('fraudulent_transactions_test.csv', index=False)

print("Fraudulent transactions saved to 'fraudulent_transactions_train.csv' and 'fraudulent_transactions_test.csv'.")
# For Datasets Use