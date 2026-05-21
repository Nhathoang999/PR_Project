# Dự án Nhận dạng Mẫu: Phân tích Dữ liệu Gọi vốn Cộng đồng Thổ Nhĩ Kỳ

## Tổng quan Dự án

Dự án này tập trung vào **Nhận dạng mẫu (Pattern Recognition) và mô hình dự đoán** cho các chiến dịch gọi vốn cộng đồng (crowdfunding) tại Thổ Nhĩ Kỳ. Sử dụng tập dữ liệu Turkish Crowdfunding (`turkishCF.csv`), bài toán hướng tới việc dự đoán trạng thái thành công (`success_status`) của các chiến dịch khác nhau (nghĩa là chiến dịch có `successful` hay không) dựa trên các đặc trưng dạng số, phân loại và văn bản như số tiền mục tiêu, số lượng thẻ (tags), số người ủng hộ, danh mục và mô tả chi tiết của dự án.

## Công cụ & Thư viện sử dụng

Dự án sử dụng các thư viện khoa học dữ liệu và học máy chính sau đây:
- **CatBoost**: Thuật toán gradient boosting được tối ưu hóa cho dữ liệu dạng phân loại (categorical).
- **Optuna**: Nền tảng tinh chỉnh siêu tham số (hyperparameter tuning) nâng cao.
- **SentenceTransformers**: Chuyên dùng để mã hóa (encode) các đặc trưng văn bản (như `project_description`) thành các vector nhúng (embeddings) có ý nghĩa.
- **SHAP**: Cung cấp công cụ diễn giải cho cấu trúc mô hình và mức độ quan trọng của các đặc trưng.
- **Data stack cơ bản**: `Pandas`, `NumPy`, `Matplotlib`, `Seaborn` và `Scikit-learn` để thao tác dữ liệu, chia tập dữ liệu và trực quan hóa.

## Cấu trúc thư mục

Cấu trúc cây thư mục được tổ chức module hóa, phân tách rõ ràng giữa mã nguồn và dữ liệu:

```text
├── data/
│   └── turkishCF.csv         # Tập dữ liệu gốc chứa dữ liệu gọi vốn của Thổ Nhĩ Kỳ
│
├── notebooks/
│   └── CODE.ipynb            # Jupyter Notebook chính bao gồm các phần EDA, tiền xử lý và huấn luyện mô hình
│
└── README.md                 # Tài liệu hướng dẫn của dự án (chính là tệp này)
```

## Cài đặt & Hướng dẫn chạy

1. Đảm bảo bạn đã cài đặt Python 3.8+ trên máy.
2. Cài đặt các thư viện phụ thuộc trực tiếp hoặc thông qua môi trường ảo (virtual environment):
   ```bash
   pip install pandas numpy scikit-learn catboost optuna sentence-transformers shap matplotlib seaborn
   ```
3. Mở tệp `notebooks/CODE.ipynb` trong môi trường Jupyter ưa thích của bạn (chẳng hạn như ứng dụng VS Code, JupyterLab, hoặc Jupyter Notebook).
4. **Lưu ý về đường dẫn tập dữ liệu (dataset pathing)**: Ban đầu, notebook dùng đường dẫn liên kết với Google Colab Drive. Hãy đảm bảo bạn đã chỉnh sửa dòng lệnh `pd.read_csv(...)` trong notebook thành đường dẫn tương đối để tương thích sử dụng file ở máy dưới local: `../data/turkishCF.csv`.
5. Chạy các ô mã (cells) một cách tuần tự để tái tạo lại kết quả của các bước làm sạch dữ liệu, mã hóa ngôn ngữ tự nhiên (NLP padding / embedding), huấn luyện mô hình và đánh giá chỉ số hiệu suất.
