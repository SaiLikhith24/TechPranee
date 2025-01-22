from flask import Flask, request, jsonify
import pandas as pd
import pickle
from model import preprocess_data, train_models, predict_new_input

app = Flask(__name__)

@app.route('/')
def home():
    return  jsonify({"message": "Welcome to the Machine Learning API."}), 200

# Upload endpoint
@app.route('/upload', methods=['POST'])
def upload_dataset():
    try:
        # Get the uploaded file
        file = request.files['file']
        if not file:
            return jsonify({"error": "No file uploaded"}), 400
        
        # Save the uploaded file
        data = pd.read_csv(file)
        data.to_csv('uploaded_data.csv', index=False)
        return jsonify({"message": "File uploaded successfully and saved for training"}), 200
    
    except Exception as e:
        return jsonify({"error": f"Failed to upload file: {str(e)}"}), 500

# Train endpoint
@app.route('/train', methods=['POST'])
def train_model():
    try:
        # Load the uploaded dataset
        try:
            data = pd.read_csv('uploaded_data.csv')
        except FileNotFoundError:
            return jsonify({"error": "No dataset uploaded. Please upload a file first."}), 400
        
        # Preprocess the dataset
        X_train, X_val, y_train, y_val, scaler = preprocess_data(data)
        
        # Train the models and get the best one
        train_accuracy = train_models(X_train, X_val, y_train, y_val, scaler)
        
        # Save the best model and scaler for future use
        with open('best_model.pkl', 'rb') as f:
            best_model = pickle.load(f)
        
        # Calculate test accuracy (using the same dataset for testing)
        test_accuracy = best_model.score(X_val, y_val)
        
        return jsonify({
            "message": "Model trained successfully",
            "Accuracy": test_accuracy
        }), 200
    
    except Exception as e:
        return jsonify({"error": f"Error during model training: {str(e)}"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get new data for prediction
        new_data = request.json
        if not all(col in new_data for col in ["temperature", "run_time"]):
            return jsonify({"error": "Input data must contain 'temperature', and 'run_time' columns."}), 400
        
        # Ensure the input data is structured correctly as a list of dictionaries
        new_data_list = [new_data]  # Wrap the data in a list to avoid scalar issue
        
        # Convert new data to DataFrame and make predictions
        new_data_df = pd.DataFrame(new_data_list)
        prediction = predict_new_input(new_data_df)
        print(prediction)
        print(type(prediction))
        #return jsonify({"No error" : "success"})
        return jsonify({"Downtime Flag": prediction.tolist()}), 200
    
    except Exception as e:
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
