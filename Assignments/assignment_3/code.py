import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Read the CSV files
df_train = pd.read_csv("/Users/giamihuynh/AUC_TMCI_2021/Assignments/assignment_3/data/Corona_NLP_train.csv", encoding='latin1')
df_test = pd.read_csv("/Users/giamihuynh/AUC_TMCI_2021/Assignments/assignment_3/data/Corona_NLP_test.csv", encoding='latin1')

# Concatenate the training and testing datasets
df = pd.concat([df_train, df_test], ignore_index=True)
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle the rows

# Define a function for text preprocessing
def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)     # Remove URLs
    text = re.sub(r'@[^\s]+', '', text)     # Remove user mentions
    text = re.sub(r'#([^\s]+)', '', text)    # Remove hashtags
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())   # Remove non-alphanumeric characters and convert to lowercase
    tokens = word_tokenize(text)    # Tokenize the text
    stop_words = set(stopwords.words('english'))    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join the tokens back into a single string
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# Apply the preprocessing function to the 'OriginalTweet' column
df['ProcessedTweet'] = df['OriginalTweet'].apply(preprocess_text)
print(df[['OriginalTweet', 'ProcessedTweet']].head())

from sklearn.model_selection import train_test_split

# Split the data into training and validation sets
train_size = 0.85  # 85% for training
train_df, val_df = train_test_split(df_shuffled, train_size=train_size, random_state=42)

# Apply the preprocessing function to the training data
train_df['ProcessedTweet'] = train_df['OriginalTweet'].apply(preprocess_text)
df['ProcessedTweet'] = df['OriginalTweet'].apply(preprocess_text)
val_df['ProcessedTweet'] = val_df['OriginalTweet'].apply(preprocess_text)

# Display the shapes of the datasets
print("Training set shape:", train_df.shape)
print("Validation set shape:", val_df.shape)

from sklearn.metrics import classification_report

def generate_classification_report(predictions, ground_truth, method_name):
    # Generate classification report
    report = classification_report(ground_truth, predictions, output_dict=True)
    
    # Print accuracy
    accuracy = report['accuracy']
    print(f"Accuracy: {accuracy:.2f}")

    # Print precision, recall, and F1 measure
    print("Precision, Recall, and F1 Score:")
    for label, metrics in report.items():
        if label not in ['accuracy', 'macro avg', 'weighted avg']:
            precision = metrics['precision']
            recall = metrics['recall']
            f1_score = metrics['f1-score']
            print(f"Label: {label}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}")

    # Print example of correctly classified datapoint
    print("\nExample of correctly classified datapoint:")
    for pred, true in zip(predictions, ground_truth):
        if pred == true:
            print(f"Predicted: {pred}, Actual: {true}")
            break

    # Print example of wrongly classified datapoint
    print("\nExample of wrongly classified datapoint:")
    for pred, true in zip(predictions, ground_truth):
        if pred != true:
            print(f"Predicted: {pred}, Actual: {true}")
            break

# testing examples
predictions = [0, 1, 0, 1, 1]
ground_truth = [0, 1, 1, 0, 1]
method_name = "Example Method"
generate_classification_report(predictions, ground_truth, method_name)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Define TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Define logistic regression classifier
logistic_regression = LogisticRegression(max_iter=1000)

# Create a pipeline
pipeline = Pipeline([
    ('tfidf', tfidf_vectorizer),
    ('classifier', logistic_regression)
])

# Fit the pipeline on the training data
pipeline.fit(train_df['OriginalTweet'], train_df['Sentiment'])

# Make predictions on the validation data
predictions = pipeline.predict(val_df['OriginalTweet'])

# Generate classification report
generate_classification_report(predictions, val_df['Sentiment'], "TF-IDF Logistic Regression")

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Define the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['ProcessedTweet'])
X_val_tfidf = tfidf_vectorizer.transform(val_df['ProcessedTweet'])  # Transform the validation data using the fitted TF-IDF vectorizer

# Split the data into training and validation sets
X = df['ProcessedTweet']
y = df['Sentiment']  

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

# Import MLPClassifier
from sklearn.neural_network import MLPClassifier

# Initialize the MLPClassifier
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', random_state=42)

# Fit the classifier on the training data
mlp_classifier.fit(X_train_tfidf, y_train)

# Make predictions on the validation data
predictions = mlp_classifier.predict(X_val_tfidf)

# Generate classification report
report = classification_report(y_val, predictions)
print(report)

df_test = pd.read_csv("data/Corona_NLP_test.csv")

# Split the data into training and validation sets
test_size = 0.85  # 85% for training
test_df, val_df = train_test_split(df_shuffled, train_size=train_size, random_state=42)

# Apply the preprocessing function to the training data
df_test['ProcessedTweet'] = df_test['OriginalTweet'].apply(preprocess_text)
df['ProcessedTweet'] = df['OriginalTweet'].apply(preprocess_text)
val_df['ProcessedTweet'] = val_df['OriginalTweet'].apply(preprocess_text)

from sklearn.ensemble import RandomForestClassifier

# Transform the test data using the fitted TF-IDF vectorizer
X_test_tfidf = tfidf_vectorizer.transform(df_test['ProcessedTweet'])

X_test = df_test['ProcessedTweet']
y_test = df_test['Sentiment']

random_forest = RandomForestClassifier()
random_forest.fit(X_test_tfidf, y_test) # Fit the classifier on the TF-IDF transformed test data

# Transform the test data using the fitted TF-IDF vectorizer
X_test_tfidf = tfidf_vectorizer.transform(df_test['ProcessedTweet'])

# Make predictions on the test data using the trained random forest classifier
predictions_random_forest = random_forest.predict(X_test_tfidf)

# Generate classification report for random forest classifier
print("Random Forest Classifier:")
print(classification_report(df_test['Sentiment'], predictions_random_forest))

# Plot confusion matrix for the random forest classifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#  Plot confusion matrix for the random forest classifier
cm = confusion_matrix(df_test['Sentiment'], predictions_random_forest)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Random Forest Classifier')
plt.show()
