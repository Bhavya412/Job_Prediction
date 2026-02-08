
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load data
df = pd.read_csv("data/Jobs_NYC_Postings.csv", low_memory=False)
df.columns = df.columns.str.strip()

# Fill missing experience
df['Years of Experience'] = df['Years of Experience'].fillna(0)

# Drop rows missing important fields
df = df.dropna(subset=[
    "Business Title", "Agency", "Work Location", "Job Category",
    "Salary Range From", "Salary Range To"
])

# Compute average salary
df['Average Salary'] = (df['Salary Range From'] + df['Salary Range To']) / 2

# Features and target
X = df[["Business Title", "Agency", "Work Location", "Job Category", "Years of Experience"]]
y = df["Average Salary"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
categorical_features = ["Business Title", "Agency", "Work Location", "Job Category"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ],
    remainder='passthrough'  # pass Years of Experience
)

# Pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train
model.fit(X_train, y_train)

# Save
joblib.dump(model, "models/salary_predictor.pkl")
print("✅ Model trained and saved  to models/salary_predictor.pkl")

