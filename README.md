# CODSOFT
#codsoft_intenship_task

Task:1 Movie Genre Classification
This project focuses on predicting the genre of a movie based on its plot description using Machine Learning techniques.

## Problem Statement
The goal of this project is to automatically classify movie genres from textual plot summaries.

## Dataset
- Movie Genre Classification Dataset
- Contains movie titles, plot descriptions, and their genres
- Data used for training and testing the model

## Methodology
- Text data preprocessing
- TF-IDF vectorization to convert text into numerical features
- Train–test split for evaluation
- Multinomial Naive Bayes classifier used for prediction

## Results
- Model achieved an accuracy of approximately **52%**
- Successfully predicts genres such as Drama, Action, Romance, and Horror

## Tools & Technologies
- Python
- Google Colab
- Pandas
- Scikit-learn

## Conclusion
This project demonstrates how machine learning models can be used to classify text data and predict movie genres effectively.



Task:2 Customer Churn Prediction
## Project Overview
This project focuses on predicting customer churn using machine learning.
Customer churn means identifying customers who are likely to stop using a service.
The goal is to help businesses take early action to retain customers.

## Dataset
The dataset contains customer information such as age, gender, geography,
account balance, number of products, and activity status.
The target column "Exited" indicates whether a customer has churned.

## Models Used
- Logistic Regression
- Random Forest

## Results
- Logistic Regression Accuracy: 81.55%
- Random Forest Accuracy: 86.45%

Random Forest performed better and was selected as the final model.

## Conclusion
This project shows how machine learning can be used to predict customer churn
and help businesses reduce customer loss.


Task:3 Spam SMS Detection using Machine Learning
## Project Description
This project builds a machine learning model to classify SMS messages as spam or legitimate (ham). It uses text processing and classification techniques to identify unwanted and fraudulent messages.

## Objective
- Detect spam SMS messages automatically  
- Reduce unwanted and scam messages  
- Apply NLP techniques on real-world text data  

## Dataset
- SMS Spam Collection Dataset  
- Messages are labeled as spam or ham  
- Dataset was extracted from a ZIP file for processing  

## Technologies Used
- Python  
- Pandas  
- Scikit-learn  
- TF-IDF Vectorizer  
- Naive Bayes Classifier  

## Workflow
1. Load and extract the dataset  
2. Preprocess SMS text data  
3. Convert text into numerical features using TF-IDF  
4. Split data into training and testing sets  
5. Train Naive Bayes model  
6. Evaluate model performance  
7. Test with new SMS messages  

## Model
- Multinomial Naive Bayes  
- Efficient for text classification tasks  

## Results
- Successfully classified SMS messages as spam or ham  
- Model shows good accuracy on test data  

## Sample Output
SPAM MESSAGE
HAM (NOT SPAM)

## Conclusion
The spam detection model effectively identifies unwanted SMS messages using machine learning and NLP techniques.



Task:4 Handwritten Text Generation
## Project Overview
This project focuses on generating handwritten-style text using deep learning.
A character-level Recurrent Neural Network (RNN) is used to learn text patterns
and generate new text similar to handwriting.

## Dataset
The project uses a small handwritten-style text dataset created for training.
The text data is processed at the character level to train the model.
Due to limited data, the generated output may contain repetitive characters.

## Model Used
- Recurrent Neural Network (RNN)
- LSTM (Long Short-Term Memory)

## Results
The trained model is able to generate new text based on learned character patterns.
The quality of generated text depends on the size of the training dataset.

## Conclusion
This project demonstrates the use of RNN and LSTM models for text generation.
Increasing the amount of training data can improve the quality of generated text.










