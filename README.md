# Financial Sentiment Analysis Dataset

This project aims to advance financial sentiment analysis research by combining two datasets: **FiQA** and **Financial PhraseBank**, into a single, easy-to-use CSV file. This dataset includes financial sentences along with their corresponding sentiment labels.

## Installation

Before running the code, ensure you have the necessary libraries installed. You can install them using pip:

```bash
pip install transformers torch
pip install accelerate
Usage
Load the Dataset
Begin by loading the dataset using Pandas.

Copy
import pandas as pd

data = pd.read_csv('data.csv')
Inspect the Data
You can check the first few rows and basic information about the dataset.

Copy
print(data.head())
print(data.info())
Visualize Sentiment Distribution
Visualize the distribution of sentiments in the dataset.

Copy
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Sentiment', data=data)
plt.title('Class Distribution')
plt.show()
Data Overview
The dataset consists of financial sentences labeled with sentiments (e.g., positive, negative, neutral). You can check the shape and columns of the dataset using:

Copy
print("Shape of data:", data.shape)
print("Columns in data:", data.columns)
Data Visualization
Visualizations help in understanding the distribution of sentiments and identifying any imbalances in the dataset.

Copy
plt.figure(figsize=(8, 6))
sns.countplot(x='Sentiment', data=data)
plt.title('Class Distribution')
plt.show()
Text Preprocessing
Text preprocessing is a crucial step for sentiment analysis. This includes:

Tokenization
Lemmatization
Removal of stopwords and punctuation
The preprocessing function is defined as follows:

Copy
def preprocess_text(sentence):
    # Tokenization and cleaning steps
    ...
Model Training
Two models are trained in this project: XGBoost and Random Forest, followed by a BERT model for sequence classification.

XGBoost
Copy
from xgboost import XGBClassifier

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train_tfidf, y_train, sample_weight=[class_weights_dict[cls] for cls in y_train])
Random Forest
Copy
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weights_dict)
rf_model.fit(X_train_tfidf, y_train)
BERT Model
Copy
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(list(X_train), truncation=True, padding=True)
train_dataset = SentimentDataset(train_encodings, y_train)
Evaluation
After training, XGBoost and Random Forest models were evaluated using accuracy, classification reports, and confusion matrices. The BERT model was evaluated using the Trainer class, accuracy, and classification report.

Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss changes.

License
This project is licensed under the MIT License. See the LICENSE file for details.

