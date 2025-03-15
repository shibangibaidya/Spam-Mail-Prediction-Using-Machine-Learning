# Spam Mail Prediction Using Machine Learning

This project involves building a machine learning model to classify emails as spam or ham (not spam). The model was trained using the logistic regression algorithm and evaluated on performance metrics to ensure its effectiveness. A user-friendly prediction system has been implemented, allowing users to input new email messages and check whether they are classified as spam or ham.

## Model Workflow

1. **Data Preprocessing**:
   - Cleaned and prepared the email dataset for analysis.
   - Text preprocessing steps included tokenization and feature extraction.

2. **Feature Extraction**:
   - Used `TfidfVectorizer` to convert text data into numerical feature vectors.

3. **Model Training**:
   - Trained a logistic regression model on the transformed dataset.
   - Evaluated model performance using metrics like accuracy, precision, recall, and F1-score.

4. **Prediction System**:
   - Users can input new email messages into the system.
   - The system converts text into numerical features and predicts whether the message is spam or ham.

## Model Performance

- **Accuracy (Training Data)**: 0.9884
- **Accuracy (Test Data)**: 0.9795
- **Precision**: 0.9699
- **Recall**: 0.9920
- **F1-Score**: 0.9809

## Tech Stack

**Language:**
- `python`

**Libraries:**
- `numpy`
- `pandas`
- `sklearn` (Scikit-learn)

## How It Works

1. **User Input**:
   - The system accepts an email message from the user as input.

2. **Text-to-Feature Conversion**:
   - The message is converted into a feature vector using the `TfidfVectorizer`.

3. **Prediction**:
   - The pre-trained logistic regression model predicts whether the input message is spam or ham.

4. **Output**:
   - If the message is classified as spam, the system outputs: `This is Spam Mail`.
   - Otherwise, the system outputs: `This is Ham Mail`.

## Authors

- [@shibangibaidya](https://www.github.com/shibangibaidya)

## Deployment

To clone and run this project, use the following commands:

```bash
git clone https://github.com/shibangibaidya/Spam-Mail-Prediction-Using-Machine-Learning.git
cd Spam-Mail-Prediction-Using-Machine-Learning
jupyter notebook spam-mail-prediction.ipynb
