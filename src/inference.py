import joblib
import pandas as pd
import numpy as np

class CrowdfundingPredictor:
    def __init__(self, model_dir='models'):
        self.model = joblib.load(f"{model_dir}/ensemble_model.pkl")
        self.text_extractor = joblib.load(f"{model_dir}/text_extractor.pkl")
        self.ordinal_enc = joblib.load(f"{model_dir}/ordinal_enc.pkl")
        self.cat_cols = joblib.load(f"{model_dir}/cat_cols.pkl")
        
    def predict_with_advice(self, input_data: dict):
        """
        Dự đoán và đưa ra tư vấn (Prescriptive Analytics).
        input_data: từ điển chứa các thông tin như mô tả, số tiền, thể loại,...
        """
        df = pd.DataFrame([input_data])
        
        # Tiền xử lý
        df.fillna('unknown', inplace=True)
        if 'target_amount' in df.columns and 'category' in df.columns:
            # Giả lập lại biến phân nhóm
            df['target_category_group'] = df['category'] + "_" + "Med" 

        # Xử lý text
        X_feats = self.text_extractor.transform(df)
        
        # Categorical Encoding
        for col in self.cat_cols:
            if col not in X_feats.columns:
                X_feats[col] = 'unknown'
        X_feats[self.cat_cols] = self.ordinal_enc.transform(X_feats[self.cat_cols].astype(str))
        
        X_feats.fillna(0, inplace=True)
        
        # Căn chỉnh lại thứ tự cột cho khớp với model train
        try:
            expected_cols = self.model.feature_names_in_
            for col in expected_cols:
                if col not in X_feats.columns:
                    X_feats[col] = 0
            X_feats = X_feats[expected_cols]
        except AttributeError:
            pass # Nếu phiên bản scikit-learn cũ không có thuộc tính này
            
        prob = self.model.predict_proba(X_feats)[0][1]
        
        # Tạo Business Insights & Tư vấn dựa trên dữ liệu đầu vào
        advice = []
        if prob < 0.5:
            advice.append("Chiến dịch có nguy cơ thất bại cao.")
            if input_data.get('target_amount', 0) > 50000:
                advice.append("- Mục tiêu gọi vốn đang khá cao. Hãy thiết lập các mốc gọi vốn (milestone) nhỏ hơn để dễ đạt được.")
            if len(str(input_data.get('project_description', '')).split()) < 200:
                advice.append("- Mô tả dự án quá ngắn. Các dự án thành công thường có hơn 200 từ giải thích rõ ràng lý do gọi vốn.")
            if input_data.get('number_of_tags', 0) < 3:
                advice.append("- Hãy thêm nhiều thẻ (tags) để chiến dịch dễ dàng tiếp cận được đối tượng mục tiêu hơn.")
        else:
            advice.append("Chiến dịch có khả năng thành công cao!")
            advice.append("- Hãy tiếp tục duy trì tương tác qua Mạng Xã Hội để giữ chân nhà tài trợ.")
            
        return prob, advice
