# Phân tích dữ liệu gọi vốn cộng đồng Thổ Nhĩ Kỳ

## Tổng quan Dự án

Dự án này tập trung vào **Nhận dạng mẫu (Pattern Recognition) và mô hình dự đoán** cho các chiến dịch gọi vốn cộng đồng (crowdfunding) tại Thổ Nhĩ Kỳ. Dựa trên tập dữ liệu Turkish Crowdfunding (`turkishCF.csv`), dự án cung cấp một ứng dụng Web (Prescriptive Analytics) giúp người dùng nhập thông tin dự án khởi nghiệp tương lai của họ và nhận về dự đoán **tỉ lệ thành công** cũng như **lời khuyên kinh doanh** để cải thiện kết quả.

## Công nghệ & Thư viện sử dụng

Dự án sử dụng các công nghệ, thư viện khoa học dữ liệu và học máy tiên tiến:
- **Streamlit**: Xây dựng giao diện Web App tương tác nhanh chóng.
- **Ensemble Learning (Stacking)**: Kết hợp sức mạnh của nhiều mô hình mạnh mẽ:
  - **CatBoost**, **LightGBM**, **XGBoost** (Base Estimators).
  - **RandomForestClassifier** (Meta Learner).
- **SentenceTransformers**: Mô hình NLP mạnh mẽ (BERT đa ngữ / Thổ Nhĩ Kỳ) để trích xuất đặc trưng văn bản (Text Embeddings) từ phần mô tả chi tiết của dự án.
- **Scikit-learn, Pandas, NumPy**: Xây dựng pipeline tiền xử lý (Preprocessing), mã hóa biến phân loại (OrdinalEncoder) và quản lý dữ liệu.

## Cấu trúc thư mục

Cấu trúc cây thư mục được tổ chức theo chuẩn Modularizing gọn gàng:

```text
├── app.py                    # Giao diện Web App Streamlit (Main Entry)
├── requirements.txt          # Danh sách các thư viện cần cài đặt
├── README.md                 # Tài liệu hướng dẫn của dự án
│
├── data/
│   └── turkishCF.csv         # Tập dữ liệu gốc chứa dữ liệu gọi vốn của Thổ Nhĩ Kỳ
│
├── models/                   # Chứa các file mô hình đã được huấn luyện (.pkl)
│   ├── ensemble_model.pkl
│   ├── text_extractor.pkl
│   └── ...
│
├── notebooks/
│   └── CODE.ipynb            # Jupyter Notebook dùng để EDA, phân tích dữ liệu ban đầu
│
└── src/                      # Source code (Mã nguồn chính) phân tách theo chức năng
    ├── data_loader.py        # Module đọc và chuẩn hóa tên cột dữ liệu
    ├── features.py           # Module NLP trích xuất đặc trưng văn bản (BERT)
    ├── preprocessing.py      # Module tiền xử lý, rút gọn biến và xử lý NaN
    ├── train.py              # Script huấn luyện (Training pipeline) toàn bộ mô hình
    └── inference.py          # Module hỗ trợ dự đoán và đưa ra lời khuyên cho Web App
```

## Hướng dẫn Cài đặt & Chạy ứng dụng

1. Đảm bảo bạn đã cài đặt Python 3.8+ trên máy tính.
2. Cài đặt các thư viện phụ thuộc:
   ```bash
   pip install -r requirements.txt
   ```
   Hoặc cài thủ công:
   ```bash
   pip install streamlit pandas numpy scikit-learn catboost lightgbm xgboost sentence-transformers joblib
   ```
3. **Huấn luyện mô hình**: Nếu thư mục `models/` chưa có hoặc bạn vừa chỉnh sửa logic tiền xử lý, hãy chạy script huấn luyện:
   ```bash
   python -m src.train
   ```
4. **Khởi chạy Web App**: Chạy giao diện tương tác Streamlit:
   ```bash
   python -m streamlit run app.py
   ```
5. Mở trình duyệt và truy cập vào `http://localhost:8501` để sử dụng Hệ thống Tư vấn Gọi Vốn.

# Link model: https://drive.google.com/drive/folders/1tlgbrhz9zcyVHlzI7YB9ghOH3imtOo0B?usp=drive_link
