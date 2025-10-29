import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1️⃣ Load the dataset
df = pd.read_csv("data/sentiment140.csv", encoding='latin-1', header=None)
df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']

# Keep only needed columns
df = df[['text', 'target']]

# Map 0 = negative, 4 = positive
df['target'] = df['target'].replace({4: 1})  # 1 means positive, 0 negative

print("Dataset loaded:", df.shape)
print(df.head())

# 2️⃣ Split data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['target'], test_size=0.2, random_state=42)

# 3️⃣ TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4️⃣ Train Logistic Regression Model
model = LogisticRegression(max_iter=200)
model.fit(X_train_tfidf, y_train)

# 5️⃣ Evaluate
y_pred = model.predict(X_test_tfidf)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6️⃣ Save Model and Vectorizer
joblib.dump(model, "data/sentiment_model.pkl")
joblib.dump(vectorizer, "data/tfidf_vectorizer.pkl")

print("\n✅ Model training complete! Saved as 'sentiment_model.pkl' and 'tfidf_vectorizer.pkl'")
