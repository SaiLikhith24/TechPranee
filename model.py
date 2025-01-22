import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



# Data preprocessing will be used for both training and testing datasets
def preprocess_data(data):
    data = data[["machine_id", "temperature", "run_time", "down_time_flag"]].dropna()
    X = data[["temperature", "run_time"]]
    y = data["down_time_flag"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler

# Train models and find the best one
# Train models and find the best one
def train_models(X_train, X_val, y_train, y_val, scaler):
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42)
    }
    
    best_model = None
    highest_accuracy = 0
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        val_accuracy = model.score(X_val, y_val)
        
        if val_accuracy > highest_accuracy:
            highest_accuracy = val_accuracy
            best_model = model
    
    # Save the best model and scaler
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    # Save the scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return highest_accuracy



# Function to predict downtime for new inputs
def predict_new_input(new_data):
    """
    Predicts downtime_flag for the given new data.
    Args:
        new_data (DataFrame): New data with columns ["machine_id", "temperature", "run_time"].
    Returns:
        List: Predictions for each input row in new_data.
    """
    try:
        # Load the saved model and scaler
        with open('best_model.pkl', 'rb') as f:
            best_model = pickle.load(f)
        
        # Ensure the new data has required columns
        if not all(col in new_data.columns for col in ["temperature", "run_time"]):
            raise ValueError("Input data must contain 'temperature', and 'run_time' columns.")
        
        # Scale the new data
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        new_data_scaled = scaler.transform(new_data[["temperature", "run_time"]])
        
        # Make predictions
        predictions = best_model.predict(new_data_scaled)
        return predictions
    
    except FileNotFoundError:
        print("Error: Model or scaler file not found. Train the model first.")
        return None
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None
