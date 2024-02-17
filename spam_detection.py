import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Open the CSV file with the appropriate encoding
with open('spam.csv', 'r', encoding='latin-1') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    data = list(reader)

# Extract the text and label columns
X = [row[1] for row in data]
y = [row[0] for row in data]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a CountVectorizer to convert text into a matrix of token counts
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
# print(X_train_counts)
# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_counts, y_train)

# Transform the test data into token counts
X_test_counts = vectorizer.transform(X_test)
# print(X_test_counts)
# Make predictions on the test data
predictions = classifier.predict(X_test_counts)
# print(predictions)
# Calculate the accuracy of the model
accuracy = 100*metrics.accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
