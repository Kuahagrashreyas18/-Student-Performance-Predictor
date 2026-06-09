# Step 1 - Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2 - Load Dataset
df = pd.read_csv('student-mat.csv/student-mat.csv', sep=';')
print("Dataset loaded successfully!")
print("Shape:", df.shape)
print(df.head())

# Step 3 - EDA
print("\nMissing Values:\n", df.isnull().sum())
print("\nBasic Stats:\n", df.describe())

# Step 4 - Visualizations
sns.countplot(x='G3', data=df)
plt.title('Final Grade Distribution')
plt.show()

sns.boxplot(x='studytime', y='G3', data=df)
plt.title('Study Time vs Final Grade')
plt.show()

# Step 5 - Preprocessing
df['result'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)
features = ['age', 'studytime', 'failures', 'absences', 'G1', 'G2']
X = df[features]
y = df['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTraining size:", X_train.shape)
print("Testing size:", X_test.shape)

# Step 6 - Train Model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
print("\nModel Trained Successfully ✅")

# Step 7 - Evaluate
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Step 8 - Predict New Student
new_student = pd.DataFrame([[17, 2, 0, 3, 12, 13]], columns=features)
prediction = model.predict(new_student)
if prediction[0] == 1:
    print("\n✅ Student is likely to PASS")
else:
    print("\n❌ Student is likely to FAIL")