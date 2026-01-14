import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import os


file_name = 'data.csv'
if not os.path.exists(file_name):
    print(f" CRITICAL ERROR: I cannot find '{file_name}'.")
    print(f" Please make sure you downloaded the file and renamed it to '{file_name}'.")
    print(f" The file must be in this folder: {os.getcwd()}")
    exit() 

df = pd.read_csv(file_name)
print(" Dataset Loaded Successfully!")
if 'Unnamed: 32' in df.columns:
    df = df.drop('Unnamed: 32', axis=1)

if 'id' in df.columns:
    df = df.drop('id', axis=1)

target_column = 'diagnosis' 

if target_column not in df.columns:
    
    for col in df.columns:
        if col.lower().strip() == 'diagnosis':
            target_column = col
            break
    
    if target_column not in df.columns:
        print(f" ERROR: I cannot find a column named 'diagnosis' in your CSV.")
        print(f" Your columns are: {df.columns.tolist()}")
        exit()

df['target'] = df[target_column].map({'M': 1, 'B': 0})
print(" Diagnosis column converted to numbers (M=1, B=0)")
X = df.drop([target_column, 'target'], axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("------------------------------------------------")
print(f" Final Model Accuracy: {accuracy * 100:.2f}%")
print("------------------------------------------------")
print("Classification Report:\n", classification_report(y_test, predictions))
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Breast Cancer Prediction)')
plt.ylabel('Actual Diagnosis')
plt.xlabel('Predicted Diagnosis')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print(" Graph 1 Saved: confusion_matrix.png")
plt.show()
plt.figure(figsize=(10, 8))
sns.heatmap(pd.DataFrame(X_train, columns=X.columns).iloc[:, :10].corr(), annot=True, fmt='.1f', cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
print(" Graph 2 Saved: correlation_heatmap.png")
plt.show()