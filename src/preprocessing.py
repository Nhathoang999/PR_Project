import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class CrowdfundingPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Giữ lại đúng các feature mà Web App sẽ có
        self.keep_cols = [
            'platform_name', 'category', 'target_amount', 
            'number_of_tags', 'number_of_awards', 
            'number_of_social_media_accounts', 'project_description', 
            'success_status'
        ]
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_copy = X.copy()
        
        # Xóa tất cả các cột không có trong keep_cols
        cols_to_drop = [col for col in X_copy.columns if col not in self.keep_cols]
        X_copy.drop(columns=cols_to_drop, inplace=True)
        
        # Xử lý các giá trị 'unknown', NaN
        X_copy.fillna('unknown', inplace=True)
        for col in X_copy.select_dtypes(include=['object']).columns:
            if col != 'project_description' and col != 'success_status':
                X_copy[col] = X_copy[col].astype(str).str.lower()
                
        # Transform 'success_status' to 1 (successful) and 0 (failed) nếu có
        if 'success_status' in X_copy.columns:
            # Dữ liệu nguyên thủy là tiếng Thổ Nhĩ Kỳ: 'başarılı' -> thành công, 'başarısız' -> thất bại
            X_copy['success_status'] = X_copy['success_status'].apply(lambda x: 1 if str(x).lower().strip() == 'başarılı' else 0)
            
        # Kỹ thuật đặc trưng (Feature Engineering cơ bản)
        if 'target_amount' in X_copy.columns and 'category' in X_copy.columns:
            # Tạo đặc trưng chéo (Cross-feature) giả lập nhóm mục tiêu
            X_copy['target_category_group'] = X_copy['category'] + "_" + pd.qcut(X_copy['target_amount'], 4, labels=['Low', 'Med', 'High', 'VeryHigh'], duplicates='drop').astype(str)
            
        return X_copy
