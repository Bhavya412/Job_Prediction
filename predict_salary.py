import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/salary_predictor.pkl")

def predict_salary(input_data: dict, increment_per_year: int = 100000) -> float:
    """
    Predict salary and apply ₹1L increment per year of experience.
    """
    df_input = pd.DataFrame([input_data])
    
    # Predict salary with 0 years of experience
    df_base = df_input.copy()
    df_base['Years of Experience'] = 0
    base_salary = model.predict(df_base)[0]

    # Adjust based on actual experience
    years = input_data.get("Years of Experience", 0)
    final_salary = base_salary + (years * increment_per_year)
    
    return round(final_salary, 2)
