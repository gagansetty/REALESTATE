import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

# Streamlit app title
st.title("Real Estate Price Prediction App")

# Load Excel file directly (no upload)
try:
    df = pd.read_excel("data_real_estate_data.xlsx")
    df.columns = df.columns.str.strip()  # Clean column names

    st.subheader("Cleaned Column Names")
    st.write(df.columns.tolist())

    # Check for required columns
    required_cols = ['HouseID', 'Area', 'Bedrooms', 'Bathroom', 'Price']
    if not all(col in df.columns for col in required_cols):
        st.error("One or more required columns are missing in the Excel file.")
    else:
        # Prepare features and target
        X = df[['Area', 'Bedrooms', 'Bathroom']]
        y = df['Price']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.subheader("Model Evaluation")
        st.write(f"Mean Squared Error: {mse:.2f}")
        st.write(f"R² Score: {r2:.2f}")

        # HouseID-based prediction
        st.subheader("Predict Price by HouseID")
        house_id = st.text_input("Enter HouseID:")

        if house_id:
            house_row = df[df['HouseID'].astype(str).str.lower() == house_id.lower()]

            if not house_row.empty:
                features = house_row[['Area', 'Bedrooms', 'Bathroom']]
                actual_price = house_row['Price'].values[0]
                predicted_price = model.predict(features)[0]

                st.write(f"### HouseID: {house_id}")
                st.write(f"Area: {features['Area'].values[0]}")
                st.write(f"Bedrooms: {features['Bedrooms'].values[0]}")
                st.write(f"Bathroom: {features['Bathroom'].values[0]}")
                st.write(f"Actual Price: {actual_price}")
                st.write(f"Predicted Price: {predicted_price:.2f}")
            else:
                st.warning(f"No house found with HouseID '{house_id}'.")

except FileNotFoundError:
    st.error("The Excel file 'data_real_estate_data.xlsx' was not found. Please make sure it exists in the same directory as this script.")
