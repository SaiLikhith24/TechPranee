# TechPranee


# Predictive Analysis for Manufacturing Operations

## Project Overview
This repository contains a project aimed at predictive analysis for manufacturing operations. The project includes:
- A machine learning model for predictive analysis.
- API endpoints to handle dataset uploads, model training, and predictions.

## Repository Structure
- `model.py`: Contains the code for building and training the machine learning model.
- `app.py`: Handles API requests and serves as the backend application.
- `requirements.txt`: Specifies the required Python libraries.
- `dataset.csv`: A sample dataset for training and testing the model.

## Setup Instructions

### Prerequisites
- Python 3.7 or higher
- `pip` package manager

### Steps to Set Up
1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python app.py
   ```
   The application will start and run on `http://127.0.0.1:5000` by default.

## API Endpoints

### `/`
- **Method:** GET
- **Description:** Displays a welcome message.

### `/upload`
- **Method:** POST
- **Description:** Uploads a dataset for training the machine learning model.
- **Input:** A CSV file with the key `file`.
- **Responses:**
  - Success: `{ "message": "File uploaded successfully and saved for training" }`
  - Failure (No file): `{ "error": "No file uploaded" }`
  - Failure (Upload error): `{ "error": "Failed to upload file: <error-message>" }`

### `/train`
- **Method:** POST
- **Description:** Trains the machine learning model using the uploaded dataset.
- **Responses:**
  - Success: `{ "Accuracy": <accuracy_value>, "message": "Model trained successfully" }`
  - Failure (No dataset): `{ "error": "No dataset uploaded. Please upload a file first." }`
  - Failure (Training error): `{ "error": "Error during model training: <error-message>" }`

### `/predict`
- **Method:** POST
- **Description:** Predicts the outcome for new data.
- **Input:** JSON object containing `temperature` and `run_time`.
- **Responses:**
  - Success: `{ "Downtime Flag": "yes"/"no" }`
  - Failure (Missing columns): `{ "error": "Input data must contain 'temperature', and 'run_time' columns." }`
  - Failure (Prediction error): `{ "error": "Error during prediction: <error-message>" }`

## Testing the API

### Using Postman
1. **Base URL:** `http://127.0.0.1:5000`

2. **Endpoints:**
   - **Welcome Message:**
     - Method: GET
     - URL: `/`
     - Response: Welcome message.
   
   - **Upload Dataset:**
     - Method: POST
     - URL: `/upload`
     - Body: Form-data with a key `file` and a CSV file as the value.
     - Response: Success or error message.

   - **Train Model:**
     - Method: POST
     - URL: `/train`
     - Response: Training accuracy or error message.

   - **Predict:**
     - Method: POST
     - URL: `/predict`
     - Body: Raw JSON in the format:
       ```json
       {
         "temperature": 75,
         "run_time": 120
       }
       ```
     - Response: Prediction result or error message.

### Example Postman Collection
You can create a Postman collection with the following endpoints and test each one with the appropriate inputs and methods.

## Notes
- Ensure the dataset you upload is in CSV format and contains relevant columns for training.
- Handle all API responses to verify successful interaction.
- Feel free to extend the model or API functionality as needed.



