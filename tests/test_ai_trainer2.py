import unittest
from ai_models.ai_trainer import train_ai_model
from data.data_collector import collect_data
from data.data_cleaner import clean_data
from indicators.bollinger import calculate_bollinger_bands
from indicators.macd import calculate_macd
from indicators.rsi import calculate_rsi
from orders.order_executor import execute_order
from strategies.ai_strategy import generate_trade_signal

class TestTradingProgram(unittest.TestCase):

    def test_data_collection(self):
        data = collect_data()
        self.assertIsNotNone(data, "데이터 수집 실패")

    def test_data_cleaning(self):
        raw_data = collect_data()
        cleaned_data = clean_data(raw_data)
        self.assertTrue(len(cleaned_data) > 0, "데이터 정리 실패")

    def test_bollinger_calculation(self):
        data = collect_data()
        cleaned_data = clean_data(data)
        bollinger = calculate_bollinger_bands(cleaned_data)
        self.assertIn("upper_band", bollinger, "볼린저밴드 계산 실패")

if __name__ == "__main__":
    unittest.main()
