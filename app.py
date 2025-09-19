import json
import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import warnings
import os

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
    
    # Get the directory where the script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        columns_path = os.path.join(current_dir, "columns.json")
        with open(columns_path, "r") as f:
            __data_columns = json.load(f)['data_columns']
            # The first 3 columns are 'total_sqft', 'bath', 'bhk'. The rest are locations.
            __locations = __data_columns[3:]
    except FileNotFoundError:
        print("Error: 'columns.json' not found. Please ensure the file is in the project directory.")
        __data_columns = []
        __locations = []
    except Exception as e:
        print(f"Error loading columns.json: {e}")
        __data_columns = []
        __locations = []

    try:
        model_path = os.path.join(current_dir, 'banglore_home_prices_model.pickle')
        with open(model_path, 'rb') as f:
            __model = pickle.load(f)
        print("Loading saved artifacts...done")
    except FileNotFoundError:
        __model = None
        print("Error: Model file 'banglore_home_prices_model.pickle' not found.")
        print("Please generate this file from your notebook and place it in the server directory.")
    except Exception as e:
        __model = None
        print(f"Error loading model: {e}")

def get_estimated_price(location, sqft, bhk, bath):
    """
    Predicts the house price based on the input features using the loaded model.
    """
    if not __model or not __data_columns:
        print("Model or column data is not loaded correctly")
        return "Model or column data is not loaded correctly. Check server logs for errors."

    try:
        # Normalize location name for comparison
        location_normalized = location.strip().lower()
        
        # Find the index for the location column
        loc_index = -1
        for i, col in enumerate(__data_columns):
            if col.lower() == location_normalized:
                loc_index = i
                break
    except Exception as e:
        print(f"Error finding location index: {e}")
        loc_index = -1

    try:
        # Create a zero array with the same number of features as the model expects
        x = np.zeros(len(__data_columns))
        x[0] = float(sqft)
        x[1] = float(bath)
        x[2] = float(bhk)
        
        if loc_index >= 0:
            # Set the value for the specific location to 1 (one-hot encoding)
            x[loc_index] = 1

        # Predict the price and round it to 2 decimal places
        prediction = __model.predict([x])[0]
        return round(prediction, 2)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return f"Error during prediction: {e}"

@app.route('/')
def index():
    """
    Renders the main page of the web application and passes the location
    list directly to the HTML template.
    """
    # Ensure artifacts are loaded
    if __model is None or __data_columns is None:
        load_saved_artifacts()
    
    # Pass the locations list, or an empty list if it's None, to prevent template errors
    return render_template('index.html', locations=__locations or [])

@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():
    """
    API endpoint to predict home price based on POST request data.
    """
    # Ensure artifacts are loaded
    if __model is None or __data_columns is None:
        load_saved_artifacts()
    
    try:
        # Handle both JSON and form data
        if request.is_json:
            data = request.get_json()
            total_sqft = float(data.get('total_sqft', 0))
            location = data.get('location', '').strip()
            bhk = int(data.get('bhk', 0))
            bath = int(data.get('bath', 0))
        else:
            total_sqft = float(request.form.get('total_sqft', 0))
            location = request.form.get('location', '').strip()
            bhk = int(request.form.get('bhk', 0))
            bath = int(request.form.get('bath', 0))
            
        # Validate inputs
        if total_sqft <= 0 or bhk <= 0 or bath <= 0:
            return jsonify({'error': 'Invalid input: All values must be positive'}), 400
            
    except (ValueError, TypeError) as e:
        return jsonify({'error': f'Invalid input format: {e}'}), 400
    except Exception as e:
        return jsonify({'error': f'Error processing input: {e}'}), 400

    try:
        estimated_price = get_estimated_price(location, total_sqft, bhk, bath)
        
        if isinstance(estimated_price, str) and "Error" in estimated_price:
            return jsonify({'error': estimated_price}), 500
            
        return jsonify({'estimated_price': estimated_price})
    except Exception as e:
        return jsonify({'error': f'Prediction error: {e}'}), 500

@app.route('/get_location_names', methods=['GET'])
def get_location_names():
    """
    API endpoint to get available location names.
    """
    # Ensure artifacts are loaded
    if __model is None or __data_columns is None:
        load_saved_artifacts()
        
    return jsonify({'locations': __locations or []})

# Load artifacts when the module is imported
load_saved_artifacts()

# For local development
if __name__ == '__main__':
    print("Starting Python Flask Server For Home Price Prediction...")
    app.run(debug=True)

# For Vercel deployment
app = app