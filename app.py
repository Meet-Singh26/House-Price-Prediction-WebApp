import json
import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import warnings

# Suppress scikit-learn warnings for a cleaner console output
warnings.filterwarnings(action='ignore', category=UserWarning)

app = Flask(__name__)

# Global variables to hold the model and column data
__model = None
__data_columns = None
__locations = None

def load_saved_artifacts():
    """
    Loads the saved model and column information from disk.
    This function is called once when the server starts.
    """
    global __data_columns
    global __locations
    global __model

    print("Loading saved artifacts...start")
    try:
        with open("columns.json", "r") as f:
            __data_columns = json.load(f)['data_columns']
            # The first 3 columns are 'total_sqft', 'bath', 'bhk'. The rest are locations.
            __locations = __data_columns[3:]
    except FileNotFoundError:
        print("Error: 'columns.json' not found. Please ensure the file is in the project directory.")
        __data_columns = []
        __locations = []

    try:
        with open('banglore_home_prices_model.pickle', 'rb') as f:
            __model = pickle.load(f)
        print("Loading saved artifacts...done")
    except FileNotFoundError:
        __model = None
        print("Error: Model file 'banglore_home_prices_model.pickle' not found.")
        print("Please generate this file from your notebook and place it in the server directory.")

def get_estimated_price(location, sqft, bhk, bath):
    """
    Predicts the house price based on the input features using the loaded model.
    """
    if not __model or not __data_columns:
        return "Model or column data is not loaded correctly. Check server logs for errors."

    try:
        # Find the index for the location column
        loc_index = __data_columns.index(location.strip().lower())
    except ValueError:
        # If location is not in our columns, it is handled as 'other', which has no specific column
        loc_index = -1

    # Create a zero array with the same number of features as the model expects
    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        # Set the value for the specific location to 1 (one-hot encoding)
        x[loc_index] = 1

    # Predict the price and round it to 2 decimal places
    return round(__model.predict([x])[0], 2)

@app.route('/')
def index():
    """
    Renders the main page of the web application and passes the location
    list directly to the HTML template.
    """
    # Pass the locations list, or an empty list if it's None, to prevent template errors
    return render_template('index.html', locations=__locations or [])

@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():
    try:
        total_sqft = float(request.form.get('total_sqft', 0))
        location = request.form.get('location', '').strip()
        bhk = int(request.form.get('bhk', 0))
        bath = int(request.form.get('bath', 0))
    except Exception as e:
        return jsonify({'error': f'Invalid input: {e}'}), 400

    estimated_price= get_estimated_price(location, total_sqft, bhk, bath)
    
    return jsonify({'estimated_price': estimated_price})


if __name__ == '__main__':
    # Load the model and column data once when the server starts
    load_saved_artifacts()
    print("Starting Python Flask Server For Home Price Prediction...")
    app.run(debug=True)

# Required for Vercel
def handler(environ, start_response):
    return app(environ, start_response)
