import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
from scipy.stats import ttest_ind
data = load_wine()
wine_df = pd.DataFrame(data.data, columns=data.feature_names)
wine_df['target'] = data.target
print(wine_df.isnull().sum())
wine_df = wine_df.dropna()
print(wine_df.info())       
print(wine_df.shape[0])    
print(wine_df.shape[1])     
print(wine_df.columns)      
print(wine_df['target'].value_counts())
print(wine_df['alcohol'].describe())
FEATURES = ['alcohol', 'malic_acid', 'flavanoids', 'color_intensity', 'hue', 'proline']
X = wine_df[FEATURES]
y = wine_df['target']
plt.figure(figsize=(20, 6))
sns.countplot(x='alcohol', data=wine_df)
plt.title('Distribution of Alcohol Content')
plt.savefig("Distribution of Alcohol Content.png")
plt.close()
plt.figure(figsize=(10, 6))
sns.histplot(wine_df['alcohol'], bins=30, kde=True)
plt.title('Histogram of Alcohol Content')
plt.xlabel('Alcohol Content (%)')
plt.savefig("Histogram of Alcohol Content.png")
plt.close()
plt.figure(figsize=(8, 6))
sns.boxplot(x=wine_df['alcohol'])
plt.title('Boxplot of Alcohol Content')
plt.xlabel('Alcohol Content (%)')
plt.savefig("Boxplot of Alcohol Content.png")
plt.close()
plt.figure(figsize=(10, 6))
sns.countplot(x='target', data=wine_df)
plt.title('Distribution of Wine Classes')
plt.xlabel('Wine Class')
plt.savefig("Distribution of Wine Classes.png")
plt.close()
plt.figure(figsize=(10, 6))
sns.scatterplot(x='alcohol', y='flavanoids', hue='target', data=wine_df)
plt.title('Alcohol Content vs Flavanoids with Target Classes')
plt.xlabel('Alcohol Content (%)')
plt.ylabel('Flavanoids')
plt.savefig("Alcohol Content vs Flavanoids with Target Classes.png")
plt.close()
class_0_alcohol = wine_df.loc[wine_df['target'] == 0, 'alcohol']
class_1_alcohol = wine_df.loc[wine_df['target'] == 1, 'alcohol']
t_stat, p_value = ttest_ind(class_0_alcohol, class_1_alcohol, alternative='two-sided', equal_var=False)
print("T-test (two-sided) p-value:", p_value)
print(wine_df.head())
print(wine_df.describe())
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])
pipe.fit(X_train, y_train)
preds = pipe.predict(X_test)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds, average="weighted")
recall = recall_score(y_test, preds, average="weighted")
f1 = f1_score(y_test, preds, average="weighted")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(classification_report(y_test, preds))
joblib.dump(pipe, "wine_cultivar_model.pkl")
print("Model saved successfully!")
