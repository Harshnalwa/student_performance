import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load Dataset
df = pd.read_csv("Student_Performance_DT - Student_Performance_DT (2).csv")

print(df.head())

# Check missing values
print(df.isnull().sum())

# Convert categorical data into numerical
df = pd.get_dummies(df, columns=["Internet_Access", "Extracurricular"], drop_first=True)

# Encode target: Pass = 1, Fail = 0
df["Final_Result"] = df["Final_Result"].map({"Pass": 1, "Fail": 0})

# Features and Target
X = df.drop("Final_Result", axis=1)
y = df["Final_Result"]

# Normalize Features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Save processed features and target so the training script can load them
X_processed = pd.DataFrame(X_scaled, columns=X.columns)
X_processed.to_csv("X_processed.csv", index=False)
y.to_csv("y_processed.csv", index=False)

print("Preprocessing complete. Saved X_processed.csv, y_processed.csv")
print("Feature columns:", list(X.columns))