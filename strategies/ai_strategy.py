# strategies/ai_strategy.py
import pandas as pd
from ai_models.lstm_model import LSTMModel
from ai_models.random_forest import RandomForestModel
from indicators.rsi import RSIIndicator
from indicators.macd import MACDIndicator
from indicators.bollinger import BollingerBandsIndicator
from indicators.stochastic import StochasticOscillator

class AIStrategy:
    def __init__(self):
        """
        AI 기반 매매 전략 (LSTM + Random Forest + RSI + MACD + Bollinger Bands + Stochastic)
        """
        self.lstm_model = LSTMModel()
        self.rf_model = RandomForestModel()
        self.rsi_indicator = RSIIndicator()
        self.macd_indicator = MACDIndicator()
        self.bb_indicator = BollingerBandsIndicator()
        self.stoch_indicator = StochasticOscillator()

    def apply_ai_strategy(self, data):
        """
        AI 및 기술적 지표를 활용한 복합 전략 적용
        :param data: OHLCV 데이터프레임
        :return: 매수/매도/보류 신호
        """
        # 지표 계산
        data = self.rsi_indicator.calculate_rsi(data)
        data = self.macd_indicator.calculate_macd(data)
        data = self.bb_indicator.calculate_bollinger_bands(data)
        data = self.stoch_indicator.calculate_stochastic(data)

        # AI 예측
        lstm_prediction = self.lstm_model.predict(data)
        rf_prediction = self.rf_model.predict(data)

        # 기술적 지표 신호 수집
        rsi_signal = self.rsi_indicator.get_rsi_signal(data)
        macd_signal = self.macd_indicator.get_macd_signal(data)
        bb_signal = self.bb_indicator.get_bollinger_signal(data)
        stoch_signal = self.stoch_indicator.get_stochastic_signal(data)

        # AI와 지표 결합 신호
        signals = [rsi_signal, macd_signal, bb_signal, stoch_signal]

        if lstm_prediction == "Buy":
            signals.append("Buy")
        elif lstm_prediction == "Sell":
            signals.append("Sell")

        if rf_prediction == "Buy":
            signals.append("Buy")
        elif rf_prediction == "Sell":
            signals.append("Sell")

        # 다수결 방식으로 최종 신호 결정
        buy_count = signals.count("Buy")
        sell_count = signals.count("Sell")

        if buy_count > sell_count:
            return "Buy"
        elif sell_count > buy_count:
            return "Sell"
        else:
            return "Hold"

if __name__ == "__main__":
    import pyupbit
    data = pyupbit.get_ohlcv("KRW-BTC", interval="minute1", count=200)
    
    ai_strategy = AIStrategy()
    signal = ai_strategy.apply_ai_strategy(data)
    print(f"AI 전략 신호: {signal}")