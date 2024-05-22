# receives 1 request record for prediction as a REST API
# pre-process
# load the model in a file
# return prediction for the record
from flask import Flask, request, jsonify
import pandas as pd
import logging
import argparse
import joblib
import os
import json
from my_classifier import my_classifier
from sklearn.preprocessing import StandardScaler
#from sklearn.base import BaseEstimator
#from sklearn.metrics import accuracy_score
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from models.read_model import preprocess_data


# Set up logging
LOG_FILE_PATH = '.\logs'
LOG_FILE = LOG_FILE_PATH + "\\" + os.path.basename( __file__.replace('.py', '.log'))
LOG_FORMAT = f"%(asctime)s - %(levelname)s - %(message)s"

# Set up log directory
if not os.path.exists(LOG_FILE_PATH):
    os.makedirs(LOG_FILE_PATH)

# Configure logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG, format=LOG_FORMAT)

# Constants
HOST = '127.0.0.1'
PORT = 5001
app = Flask(__name__)
appHasRunBefore:bool = False
encoders  = joblib.load('..\models\model_encoders_v1.pkl')
model = joblib.load('..\models\model3_v1.pkl')

'''def preprocess_data_old(df, cat_columns):
    if df is None:
        return None, "Input DataFrame is None"
    additional_cat_features = ['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender', 'PreferedOrderCat', 'MaritalStatus']

    # Check if cat_columns is not empty before adding additional features
    if cat_columns is None:
        cat_columns = additional_cat_features
    else:
        # Convert cat_columns to a list if it's not already
        if not isinstance(cat_columns, list):
            cat_columns = list(cat_columns)

    # Check if any of the cat_columns are not present in the DataFrame
    missing_columns = [col for col in cat_columns if col not in df.columns]
    if missing_columns:
        error_message = f"Columns {missing_columns} are not present in the DataFrame"
        return None, error_message

    # Convert numerical features to appropriate data types
    numerical_features = ['CustomerID', 'Churn', 'Tenure', 'PreferredLoginDevice', 'CityTier', 
                      'WarehouseToHome', 'PreferredPaymentMode', 'Gender', 'HourSpendOnApp', 
                      'NumberOfDeviceRegistered', 'PreferedOrderCat', 'SatisfactionScore', 
                      'MaritalStatus', 'NumberOfAddress', 'Complain', 'OrderAmountHikeFromlastYear', 
                      'CouponUsed', 'OrderCount', 'DaySinceLastOrder', 'CashbackAmount']
    for feature in numerical_features:
        # Check if the value is not already numeric
        if feature not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[feature]):
            logging.debug(f"Converting {feature} to float")
            df[feature] = pd.to_numeric(df[feature], errors='coerce')

    # Convert categorical features to one-hot encoding
    one_hot_encoded_features = pd.get_dummies(df[cat_columns])

    # Ensure column names match the expected format
    one_hot_encoded_features.columns = [col.replace(' ', '').replace('&', '').replace('__', '_') for col in one_hot_encoded_features.columns]

    # Combine numerical and one-hot encoded features
    processed_data = pd.concat([df[numerical_features], one_hot_encoded_features], axis=1)

    # Scale numerical features
    scaler = StandardScaler()
    processed_data[numerical_features] = scaler.fit_transform(processed_data[numerical_features])

    return processed_data, one_hot_encoded_features.columns.tolist()  # Return processed data and column names'''




# Configure logging
logging.basicConfig(filename='app.log', level=logging.DEBUG)

""" class my_classifier(BaseEstimator):
    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y=None):
        self.estimator.fit(X, y)
        return self

    def predict(self, X, y=None):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def score(self, X, y, sample_weight=None):
        # Here, you can define your custom scoring logic
        # For example, you can use accuracy_score
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred, sample_weight=sample_weight)

 """
 #@app.before_first_request
def setup():
    global appHasRunBefore
    global model, encoders
    if not appHasRunBefore:
        # Load the model from a file
        logging.info('Entering setup')
        print("Loading model: models\model2_v1.pkl")
        with open('..\models\model2_v1.pkl', 'rb') as f:
            model = joblib.load(f)
        
        # Load the encoders
        encoders  = joblib.load('..\models\model2_encoders_v1.pkl')
        appHasRunBefore = True 
        logging.info('Exiting setup')


def is_fraud_transaction(suspected_fraud_case):
    OrderAmountHikeFromlastYear = suspected_fraud_case['OrderAmountHikeFromlastYear']
    PreferedOrderCat = suspected_fraud_case['PreferedOrderCat']

    # Implement your logic to detect fraud transaction
    # For now, let's assume every transaction is not a fraud
    if OrderAmountHikeFromlastYear > 1000 and PreferedOrderCat == 'Laptop & Accessory':
        return True  # Flag transactions with high amount at high-risk merchants as fraud
    else:
        return False

@app.route('/')
def hello_world():
    global appHasRunBefore
    logging.debug('Entering hello_world')
    if appHasRunBefore:
       setup()
       appHasRunBefore = True 
    app.logger.info("Hello from Flask!")  # Use app.logger instead of print
    return "Hello, World!"

@app.route('/api/v1/fraudDetection', methods=['POST'])
def fraud_detection():
    global appHasRunBefore
    global model, encoders
    if appHasRunBefore:
        setup()
        appHasRunBefore = True 
    #global enc
    #logging.debug('Entering fraud detection')
    logging.info('Entering fraud detection')
    
    try:
        suspected_fraud_case = request.get_json()
        # Check if JSON data is present
        if suspected_fraud_case is None:
            return {"error": "No JSON data provided"}, 400
       
        print(f"req suspected_fraud_case: type = {type(suspected_fraud_case)}, value = {suspected_fraud_case}")
        # Predefined order of columns
        ordered_columns = ['CustomerID', 'Tenure', 'PreferredLoginDevice', 'CityTier',
                        'WarehouseToHome', 'PreferredPaymentMode', 'Gender', 'HourSpendOnApp',
                        'NumberOfDeviceRegistered', 'PreferedOrderCat', 'SatisfactionScore',
                        'MaritalStatus', 'NumberOfAddress', 'Complain',
                        'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount',
                        'DaySinceLastOrder', 'CashbackAmount']
        #predicted_column = 'Churn'
        # Create the DataFrame (subset data_dict based on ordered_columns)
        #df = pd.DataFrame.from_dict([suspected_fraud_case], columns=ordered_columns)

        suspected_fraud_case = [suspected_fraud_case]
        # Convert the JSON data to a DataFrame
        logging.debug(f"converting dict into df : {suspected_fraud_case}")
        df = pd.DataFrame(suspected_fraud_case) 
        logging.debug(f"converting df into ordered: {df}")
        df = df[ordered_columns]
        logging.debug(f"converted df : {df}")
        logging.info(f"req df before fit : {df}")
        df = preprocess_data(df, encoders)
        
        
        # Make predictions
        print(f"pre-processed df: {df}")
        predictions = model.predict(df)
        result = predictions.tolist()
        # Print the value of result
        logging.info("Result: %s", result)
        #le = encoders[predicted_column]
        #logging.info(f"le:  {le.classes_}")
        #decoded_res = le.inverse_transform(result)
        #logging.info("Decoded Result: %s", decoded_res)
        
        return jsonify({'isFraudTransaction': result[0] })
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)
        return jsonify({'error': str(e)}), 500
    finally:
        logging.info('Exiting fraud detection')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default=HOST)
    parser.add_argument('--port', type=int, default=PORT)
    args = parser.parse_args()
    app.run(debug=True, host=args.host, port=args.port,use_reloader=False)
