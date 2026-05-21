import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin

class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, text_column='project_description', model_name='dbmdz/bert-base-turkish-cased'):
        self.text_column = text_column
        self.model_name = model_name
        self.model = None

    def fit(self, X, y=None):
        # Load BERT model cho Tiếng Thổ Nhĩ Kỳ
        # Sử dụng mô hình nhẹ nhàng hơn của sentence-transformers có hỗ trợ đa ngữ / dbmdz nếu muốn chạy nhanh
        # Thay vì BERT thuần, chúng ta dùng parahrase-multilingual hoặc mô hình SentenceTransformer tiếng Thổ Nhĩ Kỳ
        # "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr" là một model phổ biến.
        try:
            self.model = SentenceTransformer('emrecan/bert-base-turkish-cased-mean-nli-stsb-tr')
        except:
            # Fallback về model multilingual ổn định
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        return self

    def transform(self, X):
        X_copy = X.copy()
        
        # Thêm các đặc trưng độ dài
        descriptions = X_copy[self.text_column].astype(str)
        X_copy['desc_char_length'] = descriptions.apply(len)
        X_copy['desc_word_length'] = descriptions.apply(lambda x: len(x.split()))
        
        # Khai thác Embedding
        embeddings = self.model.encode(descriptions.tolist(), show_progress_bar=False)
        
        # Đưa embeddings (vd: 384 hoặc 768 chiều) vào Dataframe
        emb_df = pd.DataFrame(embeddings, index=X_copy.index)
        emb_df.columns = [f'text_emb_{i}' for i in range(embeddings.shape[1])]
        
        # Xóa cột text ban đầu và nối bảng
        X_copy.drop(columns=[self.text_column], inplace=True)
        X_final = pd.concat([X_copy, emb_df], axis=1)
        
        return X_final
