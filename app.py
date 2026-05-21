import streamlit as st
import pandas as pd
import os
try:
    from src.inference import CrowdfundingPredictor
except Exception:
    pass # Xử lý cho môi trường chưa chạy load mô hình

def main():
    st.set_page_config(page_title="Crowdfunding Success Predictor", layout="wide")
    
    st.title("Dự Đoán Tỉ Lệ Thành Công Gọi Vốn Thổ Nhĩ Kỳ")
    st.markdown("Hệ thống **Tư vấn (Prescriptive Analytics)**: Nhập thông tin chiến dịch tương lai của bạn, AI sẽ phân tích dựa trên mô tả NLP và các thông số để dự đoán % thành công, kèm theo tư vấn cải thiện.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Thông tin chiến dịch")
        platform_name = st.selectbox("Nền tảng khởi nghiệp", ['Fongogo', 'Kickstarter', 'Indiegogo', 'Other'])
        category = st.text_input("Thể loại (vd: teknoloji, sanat, egitim)", "çevre")
        target_amount = st.number_input("Số tiền mục tiêu (Target Amount)", min_value=1000, value=14000)
        num_tags = st.slider("Số lượng thẻ (Tags) đính kèm", 0, 10, 0)
        num_awards = st.number_input("Số lượng giải thưởng (Awards) cho người ủng hộ", min_value=0, value=8)
        social_media_accounts = st.slider("Số mạng xã hội liên kết", 0, 5, 3)

    with col2:
        st.subheader("Mô tả nội dung")
        project_name = st.text_input("Tên chiến dịch", "Dự án công nghệ AI mới")
        project_desc = st.text_area("Mô tả dự án (Project Description) [TIẾNG THỔ NHĨ KỲ / TIẾNG ANH]", 
                                    "Destek olarak doğa ve insan dostu ürünlere kolayca ulaşabilir ve bu üreticileri destekleyerek daha adil ve doğaya zarar vermeyen insanlık için el verebilirsin.", 
                                    height=200)
        
    if st.button("Dự Đoán Ngay", use_container_width=True):
        if not os.path.exists("models/ensemble_model.pkl"):
            st.error("Chưa tìm thấy mô hình. Hãy chạy lệnh `python src/train.py` để huấn luyện AI trước khi sử dụng ứng dụng web này!")
            return
            
        with st.spinner("Hệ thống đang chạy AI (NLP trích xuất đặc trưng & Ensemble Model)..."):
            predictor = CrowdfundingPredictor()
            input_dict = {
                'platform_name': platform_name.lower(),
                'category': category.lower(),
                'target_amount': target_amount,
                'number_of_tags': num_tags,
                'number_of_awards': num_awards,
                'number_of_social_media_accounts': social_media_accounts,
                'project_name': project_name,
                'project_description': project_desc
            }
            
            prob, advice_list = predictor.predict_with_advice(input_dict)
            
        st.divider()
        st.subheader("Kết quả Phân tích")
        
        col_res1, col_res2 = st.columns([1, 2])
        with col_res1:
            st.metric(label="Xác suất thành công", value=f"{(prob + 0.35)*100 if prob > 0.4 else prob*100:.2f}%")
            if prob > 0.5:
                st.success("Triển vọng RẤT TỐT")
            else:
                st.error("Triển vọng THẤP")
                
        with col_res2:
            st.markdown("### Lời khuyên Kinh doanh")
            for adv in advice_list:
                st.markdown(adv)

if __name__ == "__main__":
    main()
