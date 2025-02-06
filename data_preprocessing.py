from header import *

def load_and_preprocess_deposit_data():
    df = pd.read_csv('data/bank-additional-full.csv', sep=';')
    df['y'] = df['y'].map({'yes': 1, 'no': 0})
    features = ['age', 'marital', 'education', 'default', 'housing', 'loan', 'month', 'day_of_week',
                'duration', 'campaign', 'pdays', 'previous', 'poutcome']
    numeric_features = ['age', 'duration', 'campaign', 'pdays', 'previous']

    # Кодируем категориальные столбцы
    df = encode_categorical_columns(df)

    X = df[features]
    y = df['y']
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    return X_resampled, y_resampled, scaler

def load_and_preprocess_churn_data():
    df = pd.read_csv('data/Bank Customer Churn Prediction.csv', sep=',')
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    features = ['gender', 'age', 'tenure', 'balance', 'products_number', 'credit_card', 'estimated_salary']
    X = df[features]
    y = df['churn']
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    return X_resampled, y_resampled, scaler

def encode_categorical_columns(df):
    """
    Преобразует все категориальные признаки в числовые значения.
    """
    print(df.head(10))
    # Преобразование для 'marital'
    df['marital'] = df['marital'].map({'single': 0, 'married': 1, 'divorced': 2, 'unknown': -1}).fillna(-1)
    
    # Преобразование для 'education'
    df['education'] = df['education'].map({
        'basic.4y': 0, 'basic.6y': 1, 'basic.9y': 2,
        'high.school': 3, 'professional.course': 4,
        'university.degree': 5, 'illiterate': 6, 'unknown': -1
    }).fillna(-1)
    
    # Преобразование для 'default'
    df['default'] = df['default'].map({'yes': 1, 'no': 0, 'unknown': -1}).fillna(-1)
    
    # Преобразование для 'housing'
    df['housing'] = df['housing'].map({'yes': 1, 'no': 0, 'unknown': -1}).fillna(-1)
    
    # Преобразование для 'loan'
    df['loan'] = df['loan'].map({'yes': 1, 'no': 0, 'unknown': -1}).fillna(-1)
    
    # Преобразование для 'month'
    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12, 'unknown': -1
    }
    df['month'] = df['month'].map(month_map).fillna(-1)
    
    # Преобразование для 'day_of_week'
    day_of_week_map = {'mon': 0, 'tue': 1, 'wed': 2, 'thu': 3, 'fri': 4, 'unknown': -1}
    df['day_of_week'] = df['day_of_week'].map(day_of_week_map).fillna(-1)
    
    # Преобразование для 'poutcome'
    df['poutcome'] = df['poutcome'].map({'failure': 0, 'nonexistent': 1, 'success': 2, 'unknown': -1}).fillna(-1)

    print(df.head(10))

    return df