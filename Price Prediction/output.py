import joblib
import pandas as pd
import numpy
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model("complex_bidirectional_lstm_model.h5")

# Load label encoders
label_encoders = {}
categorical_cols = ['state', 'district', 'market', 'commodity', 'variety']
for col in categorical_cols:
    label_encoders[col] = joblib.load(f"label_encoders_{col}.pkl")

# Load the scaler
scaler = joblib.load("scaler.pkl")

# Function to preprocess user input and make predictions
def predict_price(state, district, market, commodity, variety, min_price, max_price):
    # Create a DataFrame from user input
    data = pd.DataFrame([[state, district, market, commodity, variety,  min_price, max_price]],
                        columns=['state', 'district', 'market', 'commodity', 'variety', 'min_price', 'max_price'])

    # Encode categorical features
    for col, encoder in label_encoders.items():
        data[col] = encoder.transform(data[col])

    # Scale numerical features
    data[['min_price', 'max_price']] = scaler.transform(data[['min_price', 'max_price']])

    # Make predictions
    x = data.to_numpy()
    x = x.reshape((x.shape[0], x.shape[1], 1))
    predicted_price = model.predict(x)

    return predicted_price

# Take input from the user
state = input("Enter the state: ")
district = input("Enter the district: ")
market = input("Enter the market: ")
commodity = input("Enter the commodity: ")
variety = input("Enter the variety: ")
min_price = int(input("Enter the minimum price: "))
max_price = int(input("Enter the maximum price: "))

# Example usage:
predicted_price = predict_price(state, district, market, commodity, variety, min_price, max_price)
print("Predicted Price:", predicted_price)
