from header import *

def predict_churn_and_deposit(model_churn, model_deposit, scaler_churn, scaler_deposit, input_data_churn, input_data_deposit):
    input_data_churn_df = pd.DataFrame([input_data_churn], columns=['gender', 'age', 'tenure', 'balance', 'products_number', 'credit_card', 'estimated_salary'])
    input_data_churn_scaled = scaler_churn.transform(input_data_churn_df)
    churn_prediction = model_churn.predict(input_data_churn_scaled)
    churn_probability = model_churn.predict_proba(input_data_churn_scaled)[:, 1]

    input_data_deposit_df = pd.DataFrame([input_data_deposit], columns=['age', 'marital', 'education', 'default', 'housing', 'loan', 'month', 'day_of_week', 'duration', 'campaign', 'pdays', 'previous', 'poutcome'])

    input_data_deposit_scaled = scaler_deposit.transform(input_data_deposit_df)
    deposit_prediction = model_deposit.predict(input_data_deposit_scaled)
    deposit_probability = model_deposit.predict_proba(input_data_deposit_scaled)[:, 1]

    return churn_prediction[0], churn_probability[0] * 100, deposit_prediction[0], deposit_probability[0] * 100

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.005,
        depth=12,
        verbose=100,
    )
    model.fit(X_train, y_train, eval_set=(X_test, y_test))
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_proba))
    return model