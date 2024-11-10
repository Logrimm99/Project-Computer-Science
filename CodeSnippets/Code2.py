# Import required libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

train_data, train_labels = "", ""

# Convert text data to numerical format using TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,3))
train_data_vec = vectorizer.fit_transform(train_data)

# Define classifiers to compare
classifiers = {
    'Naive Bayes': MultinomialNB(),
    'Support Vector Machine': SVC(kernel='linear'),  # Linear kernel is often best for text classification
    'Logistic Regression': LogisticRegression(max_iter=1000, multi_class='ovr'),  # Change multi_class to 'multinomial' for softmax
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# Train each classifier with the train set
for clf_name, clf in classifiers.items():
    clf.fit(train_data_vec, train_labels)
