# Week4-NSP

1. Data Loading and Initial Exploration

   df = pd.read_csv('IBM HR Analytics Performance.csv')
   Loads the dataset into a pandas DataFrame.

   df.shape, df.columns, df.head(), df.info(), df.describe()
  .shape and .columns show the structure of the dataset.

   .head() previews the first few rows.

    .info() shows column types and nulls.

   .describe() provides statistical summaries of numeric features.

 
df['Attrition'].value_counts(normalize=True)
df['JobRole'].value_counts()
df['OverTime'].value_counts()
Shows class balance of the target (Attrition) and category distributions for key features.

🔹 2. Data Preprocessing and Encoding
python
Copy
Edit
le = LabelEncoder()
df['Attrition'] = le.fit_transform(df['Attrition'])  # Yes → 1, No → 0
Converts target variable from text to binary (0/1).

python
Copy
Edit
df = pd.get_dummies(df, columns=df.select_dtypes(include='object').columns, drop_first=True)
Converts categorical features like JobRole, OverTime, MaritalStatus to numeric using One-Hot Encoding.

🔹 3. Feature Scaling and SMOTE Balancing
python
Copy
Edit
scaler = StandardScaler()
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('Attrition')
df[num_cols] = scaler.fit_transform(df[num_cols])
Standardizes numerical features to have mean = 0 and std = 1. This improves ML model performance.

python
Copy
Edit
X = df.drop('Attrition', axis=1)
y = df['Attrition']
Defines feature matrix X and target vector y.

python
Copy
Edit
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
Balances the dataset using SMOTE (generates synthetic samples for the minority class).

🔹 4. Exploratory Data Analysis (EDA)
python
Copy
Edit
sns.heatmap(df.corr(), cmap='coolwarm')
Plots correlation heatmap to understand relationships between features.

python
Copy
Edit
sns.countplot(x='Attrition', data=df)
Displays count of employees who left vs stayed.

python
Copy
Edit
sns.boxplot(x='Attrition', y='MonthlyIncome', data=pd.concat([X, y], axis=1))
Shows income distribution among employees who left vs stayed.

🔹 5. Model Training and Comparison
python
Copy
Edit
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(eval_metric='logloss')
}
Initializes 3 ML models for training.

python
Copy
Edit
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
Splits the balanced dataset into training and testing sets (80/20).

python
Copy
Edit
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    print(f"{name} F1 Score: {scores.mean():.4f}")
Evaluates each model using 5-fold cross-validation with F1-score as the metric.

🔹 6. Final Model Evaluation (Random Forest)
python
Copy
Edit
best_model = RandomForestClassifier()
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
Trains the final model (Random Forest) and makes predictions.

python
Copy
Edit
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
Prints confusion matrix and classification metrics like precision, recall, F1-score.

🔹 7. Feature Importance Analysis
python
Copy
Edit
importances = pd.Series(best_model.feature_importances_, index=X.columns)
importances.nlargest(10).plot(kind='barh')
Identifies the top 10 features that influenced the model's predictions the most (e.g., OverTime, MonthlyIncome, etc.).

🔹 8. Model Interpretation using SHAP
python
Copy
Edit
import shap
explainer = shap.Explainer(best_model, X_test)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)
SHAP helps explain how each feature contributes to a prediction.

summary_plot visualizes feature impact across all predictions.

