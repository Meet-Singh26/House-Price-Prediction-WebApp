from flask import Flask, request, jsonify, render_template
import util

app = Flask(__name__)

@app.route('/')
def home():
    """
    Renders the main page with a dropdown list of locations.
    """
    locations = util.get_location_names()
    return render_template('index.html', locations=locations if locations else [])

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the prediction request from the frontend.
    """
    try:
        data = request.json
        location = data['location']
        sqft = float(data['sqft'])
        bath = int(data['bath'])
        bhk = int(data['bhk'])
        
        estimated_price = util.get_estimated_price(location, sqft, bhk, bath)
        
        return jsonify({
            'success': True,
            'prediction': estimated_price
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

# The model and artifacts are loaded in util.py when the app starts.
# No need for an __name__ == '__main__' block for Vercel deployment.

