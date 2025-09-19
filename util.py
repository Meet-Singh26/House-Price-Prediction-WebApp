import pickle
import json
import numpy as np
import os

# --- Global Variables ---
__locations = None
__data_columns = None
__model = None

# --- Core Functions ---

def get_estimated_price(location, sqft, bhk, bath):
    """
    Predicts the price based on the loaded model and input features.
    """
    global __data_columns
    global __model

    try:
        # Find the index for the location column
        loc_index = __data_columns.index(location.lower())
    except ValueError:
        # If location is not found, it's considered 'other'
        loc_index = -1

    # Create a zero vector with the same length as the data columns
    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    # Set the location index to 1 if it was found
    if loc_index >= 0:
        x[loc_index] = 1

    # Predict the price and round it to 2 decimal places
    return round(__model.predict([x])[0], 2)

def load_saved_artifacts():
    """
    Loads the trained model, location data, and column info from files
    into the global variables. This function runs once when the app starts.
    """
    print("Loading saved artifacts...start")
    global __data_columns
    global __locations
    global __model

    # Get the absolute path to the directory of the current script
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Construct absolute paths for the data files
    columns_path = os.path.join(dir_path, "columns.json")
    model_path = os.path.join(dir_path, "banglore_home_prices_model.pickle")

    try:
        with open(columns_path, "r") as f:
            __data_columns = json.load(f)['data_columns']
            # The first 3 columns are sqft, bath, bhk. Locations start from index 3.
            __locations = __data_columns[3:]

        if __model is None:
            with open(model_path, 'rb') as f:
                __model = pickle.load(f)

        print("Loading saved artifacts...done")
    except FileNotFoundError as e:
        print(f"Error loading artifacts: {e}")
        # Initialize as empty lists to prevent crashes
        __data_columns = []
        __locations = []

def get_location_names():
    """
    Returns the list of loaded location names.
    """
    return __locations

# --- Main Execution ---

# Load artifacts immediately when this module is imported.
load_saved_artifacts()
