import mlflow
from mlflow.models import infer_signature

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ============================================
# 1️⃣ Load the Iris dataset
# ============================================
X, y = datasets.load_iris(return_X_y=True)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================
# 2️⃣ Define model hyperparameters
# ============================================
params = {
    'solver': 'lbfgs',
    'max_iter': 99245,
    'multi_class': 'auto',
    'random_state': 420,
}

# ============================================
# 3️⃣ Train the Logistic Regression model
# ============================================
lr = LogisticRegression(**params)
lr.fit(x_train, y_train)

# Predict and calculate accuracy
y_pred = lr.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.4f}")

# ============================================
# 4️⃣ Setup MLflow tracking server
# ============================================
mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("interactive cars dsml b2")

# ============================================
# 5️⃣ Log parameters, metrics & model
# ============================================
with mlflow.start_run() as run:
    # Log hyperparameters
    mlflow.log_params(params)

    # Log metric
    mlflow.log_metric("accuracy", accuracy)

    # Add some descriptive tags (for UI filtering)
    mlflow.set_tags({
        "project": "iris-classification",
        "developer": "Atik",
        "description": "Basic Logistic Regression model for Iris dataset"
    })

    # Infer model signature
    signature = infer_signature(x_train, lr.predict(x_train))

    # Log model and register
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="iris_model_dsml",
        signature=signature,
        input_example=x_train,
        registered_model_name="tracking-102"
    )

print("\n🎉 Model successfully logged and registered in MLflow!")

# ============================================
# 6️⃣ Load model back from MLflow
# ============================================
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

# Make predictions again
predictions = loaded_model.predict(x_test)

# ============================================
# 7️⃣ Display results
# ============================================
iris_feature_names = datasets.load_iris().feature_names
result = pd.DataFrame(x_test, columns=iris_feature_names)
result["actual_class"] = y_test
result["predicted_class"] = predictions

print("\n🧾 Sample Prediction Results:")
print(result.head())

