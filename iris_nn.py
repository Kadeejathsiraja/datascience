# Step 3: Import Libraries
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Step 4: Load Dataset
df = pd.read_csv("iris.csv")

print("\nFirst 10 rows:\n", df.head(10))
print("\nDataset Info:\n")
print(df.info())
print("\nSummary Statistics:\n", df.describe())

# Step 5: Data Cleaning
print("\nMissing values:\n", df.isnull().sum())

# Drop missing values (if any)
df = df.dropna()

# Step 6: Encode Categorical Data
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

print("\nEncoded species:\n", df.head())

# Step 7: Split Features and Target
X = df.drop('species', axis=1)
y = df['species']

# Step 8: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# Step 9: Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 10: Build Neural Network
model = Sequential()

# Input + Hidden Layer 1
model.add(Dense(10, activation='relu', input_shape=(X_train.shape[1],)))

# Hidden Layer 2
model.add(Dense(8, activation='relu'))

# Output Layer
model.add(Dense(3, activation='softmax'))

# Step 11: Compile Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Step 12: Train Model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=8,
    verbose=1
)

# Step 13: Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", accuracy)

# Step 14: Predictions
predictions = model.predict(X_test)

# Convert probabilities to class labels
predicted_classes = np.argmax(predictions, axis=1)

print("\nPredicted Classes:\n", predicted_classes[:10])
print("\nActual Classes:\n", y_test.values[:10])