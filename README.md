# 📌 Finding Donors For Charity

## 🚀 Project Overview  
This project uses machine learning to classify individuals based on income level (`<=50K` or `>50K`) using census data. It applies data preprocessing, feature engineering, model training, and hyperparameter tuning to optimize classification performance.  

## 🛠 Features  
✅ Loads and preprocesses census data (handling missing values, encoding categorical variables)  
✅ Applies feature scaling and transformation for better model performance  
✅ Trains multiple machine learning models (SVM, SGD Classifier, KNN)  
✅ Uses GridSearchCV for hyperparameter tuning  
✅ Evaluates models using accuracy, precision, recall, and F-score  
✅ Analyzes feature importance with Random Forest  

## 📂 File Structure  
```
📂 income-classification  
 ├── 📜 income_classifier.ipynb   # Jupyter Notebook with full analysis  
 ├── 📜 visuals.py                # Helper functions for data visualization  
 ├── 📜 census.csv                # Census dataset  
 ├── 📜 README.md                 # Project documentation  
 ├── 📜 requirements.txt          # List of dependencies  
```

## 🛠 Installation  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/yourusername/income-classification.git
cd income-classification
```

### 2️⃣ Install Dependencies  
Ensure you have Python 3.x installed, then install the required libraries:  
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## 📊 How It Works  

### 📌 Step 1: Load & Preprocess Data  
✅ Reads the census dataset using pandas  
✅ Handles missing values and performs feature encoding  
✅ Scales numerical features using MinMaxScaler  

### 📌 Step 2: Train Machine Learning Models  
✅ Tests multiple models: Support Vector Machine (SVM), Stochastic Gradient Descent (SGD), and K-Nearest Neighbors (KNN)  
✅ Splits data into training and test sets (80/20 split)  

### 📌 Step 3: Optimize Model Performance  
✅ Uses GridSearchCV to fine-tune hyperparameters  
✅ Evaluates models using accuracy and F-score  

### 📌 Step 4: Feature Importance Analysis  
✅ Trains a Random Forest model to identify key features  
✅ Reduces dataset to the most important features for improved efficiency  

### 📌 Step 5: Evaluate Final Model  
✅ Compares original and optimized models  
✅ Tests performance using accuracy and F-score  

## 🔥 Results  
📊 **Final Model Performance**  
- Optimized model achieved higher accuracy and F-score than the baseline  
- Feature selection improved training efficiency without loss in performance  

## 💡 Future Improvements  
🚀 Test deep learning models like Neural Networks  
🚀 Tune additional hyperparameters for better performance  
🚀 Deploy the model as a web application using Flask or FastAPI  

## 📜 License  
This project is open-source and free to use. Contributions are welcome! 🚀  
