import pandas as pd

class DataLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load_data(self) -> pd.DataFrame:
        """Đọc và đổi tên cột dữ liệu thành tiếng Anh để đồng bộ."""
        df = pd.read_csv(self.filepath, sep=';', on_bad_lines='skip')
        
        rename_dict = {
            'platform_adi': 'platform_name',
            'kitle_fonlamasi_turu': 'crowdfunding_type',
            'kategori': 'category',
            'fon_sekli': 'fund_type',
            'proje_adi': 'project_name',
            'proje_sahibi': 'project_owner',
            'destekci_sayisi': 'number_of_supporters',
            'odul_sayisi': 'number_of_awards',
            'ekip_kisi_sayisi': 'number_of_team_members',
            'web_sitesi': 'website',
            'sosyal_medya': 'social_media',
            'sm_sayisi': 'number_of_social_media_accounts',
            'sm_takipci': 'social_media_followers',
            'etiket_sayisi': 'number_of_tags',
            'icerik_kelime_sayisi': 'number_of_words_in_content',
            'proje_aciklamasi': 'project_description',
            'hedef_miktari': 'target_amount',
            'toplanan_tutar': 'amount_collected',
            'destek_orani': 'support_rate',
            'basari_durumu': 'success_status'
        }
        df.rename(columns=rename_dict, inplace=True)
        return df
