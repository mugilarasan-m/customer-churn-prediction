# Customer Churn Prediction System

An end-to-end Machine Learning project that predicts whether a telecom customer is likely to churn. The system includes data preprocessing, model training and evaluation, hyperparameter tuning, and an interactive Streamlit web application for making real-time predictions.

## Features

* **Data Preprocessing:** Handles missing values and converts categorical features into a format suitable for machine learning models.
* **Machine Learning Pipeline:** Uses Scikit-Learn pipelines and `ColumnTransformer` for efficient preprocessing and model training.
* **Data Leakage Prevention:** Preprocessing steps are fitted only on the training data to follow proper machine learning practices.
* **Multiple Model Evaluation:** Compares several classification algorithms, including:

  * Logistic Regression
  * Decision Tree
  * Random Forest
  * K-Nearest Neighbors
  * Support Vector Machine
  * Gradient Boosting
* **Hyperparameter Tuning:** Uses `GridSearchCV` to optimize the best-performing model.
* **Model Evaluation:** Measures performance using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
* **Interactive Web Application:** Provides a Streamlit interface for predicting customer churn probability and identifying churn risk levels.
* **Retention Recommendations:** Generates data-driven suggestions based on the predicted churn risk.

## Project Structure

```text
customer-churn-prediction/
│
├── data/
│   └── customer_churn.csv          # Dataset
│
├── notebooks/
│   └── churn_analysis.py           # Exploratory Data Analysis
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py            # Data preprocessing and pipeline creation
│   ├── train_model.py              # Model training, tuning, and evaluation
│   └── predict.py                  # Prediction functions
│
├── models/
│   └── customer_churn_model.pkl    # Trained model
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Required dependencies
└── README.md                       # Project documentation
```

## How to Run the Project

Follow the steps below to set up and run the project locally.

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd customer-churn-prediction
```

### 2. Create and Activate a Virtual Environment

**macOS/Linux:**

```bash
python -m venv venv
source venv/bin/activate
```

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

Install all required Python libraries using:

```bash
pip install -r requirements.txt
```

### 4. Add the Dataset

Make sure the dataset is placed in the following location:

```text
data/customer_churn.csv
```

### 5. Run Exploratory Data Analysis (Optional)

To explore the dataset and generate visualizations, run:

```bash
python notebooks/churn_analysis.py
```

### 6. Train the Model

Run the following command to preprocess the data, train and evaluate multiple models, perform hyperparameter tuning, and save the final trained model:

```bash
python -m src.train_model
```

After successful training, the model will be saved at:

```text
models/customer_churn_model.pkl
```

### 7. Test Predictions (Optional)

To test the prediction module from the terminal, run:

```bash
python -m src.predict
```

### 8. Launch the Streamlit Application

Once the model has been trained, start the web application using:

```bash
streamlit run app.py
```

Streamlit will provide a local URL. Open it in your browser to access the Customer Churn Prediction System.

## Application Output

The web application allows users to enter customer information and receive:

* Churn prediction
* Probability of customer churn
* Customer risk level
* Data-driven retention recommendations

This helps identify customers who may be at risk of leaving and supports proactive customer retention strategies.

## Model Evaluation

The models are evaluated using the following classification metrics:

* **Accuracy:** Measures the overall percentage of correct predictions.
* **Precision:** Measures how accurately the model identifies customers predicted to churn.
* **Recall:** Measures how effectively the model identifies actual churners.
* **F1-Score:** Provides a balance between Precision and Recall.
* **ROC-AUC:** Measures the model's ability to distinguish between customers who churn and those who do not.

The best-performing model is selected based on its evaluation results and further optimized using `GridSearchCV`.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Streamlit
* Matplotlib
* Seaborn
* Joblib

## Future Improvements

* Add support for batch predictions using CSV file uploads.
* Improve the Streamlit dashboard with additional visualizations.
* Add feature importance analysis and model explainability.
* Deploy the application to a cloud platform.
* Integrate a database for storing customer predictions and historical results.

## Author

**Mugilarasan M.**

Machine Learning | AI | Data Science
