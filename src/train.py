import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from src.data_loader import DataLoader
from src.preprocessing import CrowdfundingPreprocessor
from src.features import TextFeatureExtractor

def build_model_pipeline():
    # Khai báo các mô hình cho Ensemble Learning (Stacking)
    cat_model = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1, verbose=0)
    lgbm_model = LGBMClassifier(n_estimators=200, random_state=42)
    xgb_model = XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss', random_state=42)
    
    # Meta-learner
    meta_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    stacking_clf = StackingClassifier(
        estimators=[
            ('cat', cat_model),
            ('lgbm', lgbm_model),
            ('xgb', xgb_model)
        ],
        final_estimator=meta_model,
        cv=5
    )
    
    return stacking_clf

def run():
    print("1. Đang tải dữ liệu...")
    loader = DataLoader("data/turkishCF.csv")
    df = loader.load_data()
    
    print("2. Đang phân tách dữ liệu (Train/Test)...")
    preprocessor = CrowdfundingPreprocessor()
    df_clean = preprocessor.fit_transform(df)
    
    X = df_clean.drop(columns=['success_status'])
    y = df_clean['success_status']
    
    text_extractor = TextFeatureExtractor()
    print("3. Đang trích xuất đặc trưng NLP BERT (có thể mất vài phút)...")
    X_features = text_extractor.fit_transform(X)
    
    # Xóa các cột đối tượng (chuyển Label Encoding cho Categorical)
    cat_cols = X_features.select_dtypes(include=['object']).columns.tolist()
    ordinal_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    if cat_cols:
        X_features[cat_cols] = ordinal_enc.fit_transform(X_features[cat_cols])
    
    # Điền giá trị trống cho features số nếu có
    X_features.fillna(0, inplace=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)
    
    print("4. Đang huấn luyện Ensemble Model (CatBoost + LightGBM + XGBoost -> RandomForest)...")
    model = build_model_pipeline()
    model.fit(X_train, y_train)
    
    print("5. Đánh giá mô hình...")
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    print("6. Đang lưu mô hình...")
    joblib.dump(model, "models/ensemble_model.pkl")
    joblib.dump(text_extractor, "models/text_extractor.pkl")
    joblib.dump(ordinal_enc, "models/ordinal_enc.pkl")
    joblib.dump(cat_cols, "models/cat_cols.pkl")
    print("Hoàn tất!")

if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)
    run()
