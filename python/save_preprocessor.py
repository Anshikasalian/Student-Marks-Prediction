
import pandas as pd
import joblib
from pathlib import Path

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

BASE_DIR = Path(__file__).resolve().parents[1]

df = pd.read_csv(BASE_DIR / "data" / "exam_marks.csv")

X = df.drop(columns=["math score"])

num_features = X.select_dtypes(
    exclude="object"
).columns.tolist()

cat_features = X.select_dtypes(
    include="object"
).columns.tolist()

preprocessor = ColumnTransformer([
    ("OneHotEncoder", OneHotEncoder(), cat_features),
    ("StandardScaler", StandardScaler(), num_features)
])

preprocessor.fit(X)

save_path = BASE_DIR / "notebook" / "preprocessor.pkl"

joblib.dump(preprocessor, save_path)

print("Preprocessor saved successfully!")
print("Saved at:", save_path)
print("Numerical features:", num_features)
print("Categorical features:", cat_features)
print("Input columns:", X.columns.tolist())
