from  header import *
from data_preprocessing import *
from model import *
from header import *
from gui import *
from model import *

model_file_churn = 'models/catboost_churn_model.cbm'
scaler_file_churn = 'models/scaler_churn.pkl'
preprocessor_file_deposit = 'models/preprocessor_deposit.pkl'

if os.path.exists(model_file_churn) and os.path.exists(scaler_file_churn) and os.path.exists(preprocessor_file_deposit):
    model_churn = CatBoostClassifier()
    model_churn.load_model(model_file_churn)
    scaler_churn = joblib.load(scaler_file_churn)
    preprocessor_deposit = joblib.load(preprocessor_file_deposit)
    model_deposit = CatBoostClassifier()
    model_deposit.load_model('models/catboost_deposit_model.cbm')
    create_churn_and_deposit_ui(model_churn, model_deposit, scaler_churn, preprocessor_deposit)
else:
    X_churn, y_churn, scaler_churn = load_and_preprocess_churn_data()
    model_churn = train_model(X_churn, y_churn)
    model_churn.save_model(model_file_churn)
    joblib.dump(scaler_churn, scaler_file_churn)

    X_deposit, y_deposit, preprocessor_deposit = load_and_preprocess_deposit_data()
    model_deposit = train_model(X_deposit, y_deposit)
    model_deposit.save_model('models/catboost_deposit_model.cbm')
    joblib.dump(preprocessor_deposit, preprocessor_file_deposit)

    create_churn_and_deposit_ui(model_churn, model_deposit, scaler_churn, preprocessor_deposit)