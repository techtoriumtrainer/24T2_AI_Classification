import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from joblib import dump, load

# Load dataset
df = pd.read_csv("iris.csv")

# Split dataframe into input and output
X = df.drop(["variety"], axis=1)
y = df["variety"]

# Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Initialize models
knn = KNeighborsClassifier(n_neighbors=3)
dt = DecisionTreeClassifier()
svm = SVC(kernel='linear')

# Train models against training dataset
knn.fit(X_train, y_train)
dt.fit(X_train, y_train)
svm.fit(X_train, y_train)

# Test models against test dataset
y_pred_knn = knn.predict(X_test)
y_pred_dt = dt.predict(X_test)
y_pred_svm = svm.predict(X_test)

# Evaluate model performance to see which model(s) are most accurate
# KNN
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("KNN Accuracy:", accuracy_knn)
print("KNN Classification Report:\n", classification_report(y_test, y_pred_knn))

# Decision Tree
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Accuracy:", accuracy_dt)
print("Decision Tree Classification Report:\n", classification_report(y_test, y_pred_dt))

# SVM
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy:", accuracy_svm)
print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))

# Create dictionary that stores accuracy
models = {'KNN' : knn, 'Decision Tree' : dt, 'SVM' : svm}
accuracy_scores = {'knn': accuracy_knn, 'dt': accuracy_dt, 'svm': accuracy_svm}

# Plot accuracy comparison
plt.figure(figsize=(10, 6))
plt.bar(accuracy_scores.keys(), accuracy_scores.values(), color=['blue', 'green', 'red'])
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison of Different Models')
plt.ylim([0, 1])
for i, v in enumerate(accuracy_scores.values()):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
plt.show()

# Find the most accurate model
most_accurate_model = max(accuracy_scores, key=accuracy_scores.get)
print("Most Accurate Model:", most_accurate_model)

# Save the most accurate model
dump(eval(most_accurate_model), "best_classification_model.joblib")

# Load the saved model
loaded_model = load("best_classification_model.joblib")

# Manually enter data to test the model
input_data = []
print("Enter the values for the following features:")
for feature in df.columns[:-1]:
    value = float(input(f"{feature}: "))
    input_data.append(value)

# Scale the input data
scaled_input_data = scaler.transform([input_data])

# Predict using the loaded model
prediction = loaded_model.predict(scaled_input_data)
print("Predicted class:", prediction[0])