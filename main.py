import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('/content/customer_segmentation.csv')

features_seg = df.select_dtypes(include=[np.number]).dropna(axis=1)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_seg)

kmeans = KMeans(n_clusters=3, random_state=42)
segments = kmeans.fit_predict(scaled_features)

pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_features)

plt.figure(figsize=(8, 6))
plt.scatter(pca_components[:, 0], pca_components[:, 1], c=segments, cmap='viridis')
plt.title('Customer Segmentation (PCA + KMeans)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Segment')
plt.show()
#1st objective
if 'Response' in df.columns:
    X = df.drop(columns=['Response'])
    y = df['Response']
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Customer Response')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
#2nd objective
if 'CLV' in df.columns:
    X = df.drop(columns=['CLV'])
    y = df['CLV']
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title("Actual vs Predicted CLV")
    plt.xlabel("Actual CLV")
    plt.ylabel("Predicted CLV")
    plt.show()

#3rd objective
if 'Churn' in df.columns:
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_probs = clf.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Churn Prediction')
    plt.legend(loc="lower right")
    plt.show()

#4th objective
if 'CustomerID' in df.columns and 'ProductID' in df.columns and 'Rating' in df.columns:
    pivot_table = df.pivot_table(index='CustomerID', columns='ProductID', values='Rating')
    similarity = pivot_table.corr(method='pearson')

    target_product = similarity.columns[0]
    sim_scores = similarity[target_product].sort_values(ascending=False)[1:6]

    sim_scores.plot(kind='bar', color='teal')
    plt.title(f'Top 5 Recommendations for Product {target_product}')
    plt.xlabel('ProductID')
    plt.ylabel('Similarity Score')
    plt.show()
