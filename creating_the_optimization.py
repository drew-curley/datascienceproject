import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Example: Load CSV
data = pd.read_csv("data.csv")

# Separate target (evaluation) and features
y = data["evaluation"]  # Assuming the first column is "evaluation"
X = data.drop("evaluation", axis=1)

# Identify data types
float_cols = X.select_dtypes(include=["float"]).columns
int_cols = X.select_dtypes(include=["int"]).columns
bool_cols = X.select_dtypes(include=["bool"]).columns

# Preprocessing for each type
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), float_cols),        # Scale floats
        ("cat", OneHotEncoder(handle_unknown="ignore"), int_cols),  # One-hot encode integers if categorical
        ("bool", "passthrough", bool_cols),          # Pass through booleans as-is (or convert manually)
    ]
)

# Create a pipeline with a Random Forest model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(random_state=42))
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
score = model.score(X_test, y_test)
print(f"Model R^2 Score: {score:.2f}")
