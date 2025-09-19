# **Bengaluru House Price Prediction 🏡**

This project is a web application that predicts house prices in Bengaluru, India, based on features like location, square footage, and the number of bedrooms and bathrooms. The prediction is powered by a machine learning model trained on a comprehensive dataset of Bengaluru real estate listings.

## **🚀 Live Demo**

https://meetsingh-house-price-prediction-webapp.vercel.app

## **🧠 The Machine Learning Model**

The core of this project is a **Linear Regression model**. The entire data science pipeline, from data cleaning and feature engineering to model training and evaluation, was conducted in a Colab Notebook (House\_Price\_Pred.ipynb).

### **1\. Data Cleaning & Preprocessing**

The initial dataset contained various inconsistencies that were handled systematically:

* **Dropping Irrelevant Features**: Columns like area\_type, society, balcony, and availability were removed to simplify the model.  
* **Handling Missing Values**: Rows with null values were dropped to ensure data quality.  
* **Feature Extraction**: A new bhk (Bedrooms, Hall, Kitchen) feature was engineered from the size column (e.g., "2 BHK" \-\> 2).  
* **Data Transformation**: The total\_sqft column, which contained range values (e.g., "1133 \- 1384"), was converted into a single numeric value by taking the average.

### **2\. Feature Engineering & Outlier Removal**

To improve model accuracy, several feature engineering and outlier detection techniques were applied:

* **Price Per Square Foot**: This feature was created to help identify and remove outliers based on market standards for different locations.  
* **Dimensionality Reduction**: Locations with fewer than 10 data points were grouped into an "other" category to reduce noise and prevent overfitting.  
* **Outlier Removal**:  
  * **Domain-Based**: Atypical properties, such as those with less than 300 sq. ft. per bedroom, were filtered out.  
  * **Statistical (Standard Deviation)**: Properties with a price\_per\_sqft outside one standard deviation of the mean for their location were removed.  
  * **BHK-Based**: Properties where a smaller BHK configuration was priced higher than a larger one in the same location were identified and removed as outliers.

### **3\. Model Training**

* **Encoding**: One-Hot Encoding was applied to the location column to convert categorical data into a numerical format for the model.  
* **Model Selection**: A **Linear Regression** model was chosen for its simplicity and interpretability in predicting continuous values like price.  
* **Evaluation**: The model achieved an R-squared score of approximately **84.5%** on the test set, indicating a strong correlation between the selected features and the house price.

## **🛠️ Tech Stack**

* **Machine Learning**: Scikit-learn, Numpy, Pandas  
* **Web Framework**: Flask (Python)  
* **Frontend**: HTML, Tailwind CSS, JavaScript  
* **Deployment**: Vercel

## **🚀 How to Run Locally**

1. **Clone the repository:**  
   git clone 

(https://github.com/Meet-Singh26/house-price-prediction-webapp.git)  
cd house-price-prediction-webapp

2. **Create a virtual environment and install dependencies:**  
   python \-m venv venv  
   source venv/bin/activate  \# On Windows, use \`venv\\Scripts\\activate\`  
   pip install \-r requirements.txt

3. **Run the Flask application:**  
   python app.py

   The application will be available at http://127.0.0.1:5000.