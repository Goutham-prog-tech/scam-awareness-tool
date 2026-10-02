"""Train the ML model (Level 2): TF-IDF features + Logistic Regression."""
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

df = pd.read_csv("data/messages.csv")
df["y"] = (df["label"] == "SCAM").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["y"], test_size=0.25, random_state=42, stratify=df["y"]
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test),
                            target_names=["LEGITIMATE", "SCAM"]))

# Retrain on all data before saving
model.fit(df["text"], df["y"])
joblib.dump(model, "model.joblib")
print("Saved model.joblib")