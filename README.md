# Email_Spam_Detection_with_Machine_Learning

Email Spam Detection with Machine Learning
We’ve all been the recipient of spam emails before. Spam mail, or junk mail, is a type of email that is sent to a massive number of users at one time, frequently containing cryptic messages, scams, or most dangerously, phishing content.

In this project, we use Python to build an email spam detector and then leverage machine learning to train the spam detector to recognize and classify emails into spam and non-spam. Let's get started!

Project Overview
This project focuses on creating a robust email spam detection system using machine learning. The implementation involves using popular libraries such as pandas, scikit-learn, matplotlib, seaborn, and more.

Steps Involved
Data Loading and Preprocessing: We load the email dataset using pandas and preprocess the data by converting labels to binary values (0 for ham, 1 for spam).

Feature Extraction with CountVectorizer: We use the CountVectorizer from scikit-learn to convert the text data into numerical features. The dataset is split into training and testing sets.

Naive Bayes Classifier: We train a Multinomial Naive Bayes classifier on the vectorized training data and predict on the test data. We evaluate the model's accuracy and create a confusion matrix.

SVM Classifier with TF-IDF: We utilize the TF-IDF vectorizer to transform the text data and then train a Support Vector Machine (SVM) classifier. The accuracy, confusion matrix, and classification report are generated for evaluation.

Random Forest Classifier: We train a Random Forest classifier on the TF-IDF transformed data and evaluate it using accuracy, confusion matrix, and classification report.

Visualization: We use matplotlib and seaborn to create visualizations, including a confusion matrix heatmap and a bar plot showcasing model accuracies.

Example Predictions: We provide examples of new emails and demonstrate how the trained models can predict whether they are spam or not.

Getting Started
To run the project, follow these steps:

Install required libraries: pip install pandas scikit-learn matplotlib seaborn

Load the dataset and preprocess it.

Train and evaluate the Naive Bayes, SVM, and Random Forest classifiers.

Visualize the results with confusion matrix heatmap and accuracy bar plot.

Use the trained models to predict spam or ham for new emails.
![image](https://github.com/Jsujanchowdary/Email_Spam_Detection_with_Machine_Learning/assets/91127394/acce7413-c3cd-46dc-a8f2-fad3cea902a5)

Example Predictions
Here are some example predictions:

"Ok lar... Joking wif u oni..." Prediction: ham
"Free entry in 2 a wkly comp to win FA Cup final tkts..." Prediction: spam
"Click the link to claim your prize!" Prediction: spam
Contribution
Contributions to this project are welcome! Whether it's improving the code, enhancing the models, or adding more features, feel free to open issues and pull requests.
