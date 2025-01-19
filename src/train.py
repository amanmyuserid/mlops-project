import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import mlflow
from mlflow.models.signature import infer_signature

# Load data
data = pd.read_csv('data/iris.csv')
X = data.drop(columns=['species'])
y = data['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow tracking
mlflow.start_run()
mlflow.log_param("model", "LogisticRegression")

model = LogisticRegression()
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
mlflow.log_metric("accuracy", accuracy)

# Create an input example (first row of X_train)
input_example = X_train.iloc[:1].to_dict(orient='records')

# Infer the model signature
signature = infer_signature(X_train, model.predict(X_train))

# Save model with input example and signature
mlflow.sklearn.log_model(
    model,
    "model",
    input_example=input_example,
    signature=signature
)

mlflow.end_run()

print("Model training complete. Accuracy:", accuracy)
