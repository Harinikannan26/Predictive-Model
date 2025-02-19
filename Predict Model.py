# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 23:23:43 2025

@author: harin
"""

import tkinter as tk
from tkinter import messagebox
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download stopwords
nltk.download("stopwords")
from nltk.corpus import stopwords

# Sample dataset (text messages and labels)
data = {
    "text": [
        "Win a free iPhone now!",
        "Your account has been compromised. Click this link to recover.",
        "Meeting at 5 PM today. Don't be late.",
        "Congratulations! You won a lottery worth $10000!",
        "Hey, are you coming to the party tonight?",
        "Urgent! Your bank details are required for verification.",
        "Let's catch up for coffee tomorrow.",
        "Claim your free gift by clicking this link!",
        "Call me when you are free.",
        "You have been selected for a special discount. Claim now!"
    ],
    "label": ["spam", "spam", "ham", "spam", "ham", "spam", "ham", "spam", "ham", "spam"]
}

# Convert labels to binary (spam = 1, ham = 0)
labels = [1 if label == "spam" else 0 for label in data["label"]]

# Train the model
vectorizer = TfidfVectorizer(stop_words=stopwords.words("english"))
X_train, X_test, y_train, y_test = train_test_split(data["text"], labels, test_size=0.2, random_state=42)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Check model accuracy
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Function to predict if a message is spam or not
def predict():
    user_input = text_input.get("1.0", tk.END).strip()
    if not user_input:
        messagebox.showerror("Error", "Enter a message for prediction!")
        return

    input_tfidf = vectorizer.transform([user_input])
    prediction = model.predict(input_tfidf)[0]

    result_label.config(text=f"Prediction: {'Spam' if prediction == 1 else 'Ham'}", fg="red" if prediction == 1 else "green")

# Create GUI
root = tk.Tk()
root.title("Spam Email Classifier")
root.geometry("500x350")

label = tk.Label(root, text="Spam Email Detection", font=("Arial", 16))
label.pack(pady=10)

text_input = tk.Text(root, height=5, width=50, font=("Arial", 12))
text_input.pack(pady=10)

predict_button = tk.Button(root, text="Predict", command=predict, font=("Arial", 12), bg="lightgreen")
predict_button.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()
