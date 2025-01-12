# data/data_cleaner.py
import pandas as pd

class DataCleaner:
    def __init__(self):
        self.columns_to_clean = ['open', 'high', 'low', 'close', 'volume']

    def remove_duplicates(self, data):
        """중복 데이터 제거"""
        initial_length = len(data)
        data = data.drop_duplicates()
        print(f"[Info] 중복 제거 완료: {initial_length - len(data)}건 제거됨")
        return data

    def fill_missing_values(self, data):
        """결측치 보정 (앞 데이터로 채움)"""
        initial_na_count = data.isna().sum().sum()
        data = data.fillna(method="ffill").fillna(method="bfill")
        print(f"[Info] 결측치 보정 완료: {initial_na_count}건 보정됨")
        return data

    def remove_outliers(self, data, column):
        """IQR 기반 이상치 제거"""
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        original_length = len(data)
        data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
        print(f"[Info] 이상치 제거 완료 ({column}): {original_length - len(data)}건 제거됨")
        return data

    def clean_data(self, data):
        """데이터 정제 메인 함수"""
        print("[Info] 데이터 정제 시작")
        data = self.remove_duplicates(data)
        data = self.fill_missing_values(data)
        for column in self.columns_to_clean:
            data = self.remove_outliers(data, column)
        print("[Info] 데이터 정제 완료")
        return data

if __name__ == "__main__":
    import pyupbit
    data = pyupbit.get_ohlcv("KRW-BTC", interval="minute1", count=200)
    cleaner = DataCleaner()
    cleaned_data = cleaner.clean_data(data)
    print("정제 완료 데이터 샘플:")
    print(cleaned_data.tail())
