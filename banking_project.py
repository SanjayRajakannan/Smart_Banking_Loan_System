import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

data = pd.read_csv("loan_data.csv")
data.fillna(data.mean(numeric_only=True), inplace=True)
data.fillna(data.mode().iloc[0], inplace=True)
encoder = LabelEncoder()
object_columns = data.select_dtypes(include=['object', 'string']).columns

for column in object_columns:
    data[column] = encoder.fit_transform(data[column])

print(data.head())
X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]

print("\nInput shape:", X.shape)
print("Output shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling completed!")
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

print("Model trained successfully!")
predictions = model.predict(X_test_scaled)

print("\nFirst 10 predictions:")
print(predictions[:10])

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy =", accuracy * 100, "%")

results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': predictions
})

print("\nActual vs Predicted:")
print(results.head(10))

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train_scaled, y_train)

dt_predictions = dt_model.predict(X_test_scaled)

dt_accuracy = accuracy_score(y_test, dt_predictions)

print("\nDecision Tree Accuracy =", dt_accuracy * 100, "%")

# Random Forest
rf_model = RandomForestClassifier(random_state=42)

rf_model.fit(X_train_scaled, y_train)

rf_predictions = rf_model.predict(X_test_scaled)

rf_accuracy = accuracy_score(y_test, rf_predictions)

print("Random Forest Accuracy =", rf_accuracy * 100, "%")

# KNN
knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train_scaled, y_train)

knn_predictions = knn_model.predict(X_test_scaled)

knn_accuracy = accuracy_score(y_test, knn_predictions)

print("KNN Accuracy =", knn_accuracy * 100, "%")

# PCA
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_train_scaled)

print("\nPCA completed!")
print("Original features:", X_train_scaled.shape[1])
print("Reduced features:", X_pca.shape[1])

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)

clusters = kmeans.fit_predict(X_train_scaled)

print("\nK-Means clustering completed!")
print("First 10 customer segments:")
print(clusters[:10])

import pickle

# Save Logistic Regression model
pickle.dump(model, open("model.pkl", "wb"))

print("\nModel saved successfully!")