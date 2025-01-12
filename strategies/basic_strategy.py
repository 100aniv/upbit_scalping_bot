# strategies/basic_strategy.py
from indicators.rsi import RSIIndicator
from indicators.macd import MACDIndicator
from indicators.bollinger import BollingerBandsIndicator
from indicators.stochastic import StochasticOscillator

class BasicStrategy:
    def __init__(self):
        """
        기본 매매 전략 (RSI + MACD + Bollinger Bands + Stochastic)
        """
        self.rsi_indicator = RSIIndicator()
        self.macd_indicator = MACDIndicator()
        self.bb_indicator = BollingerBandsIndicator()
        self.stoch_indicator = StochasticOscillator()

    def apply_strategy(self, data):
        """
        다중 지표를 결합하여 거래 신호 생성
        :param data: OHLCV 데이터프레임
        :return: 매수/매도/보류 신호
        """
        # 각 지표 계산
        data = self.rsi_indicator.calculate_rsi(data)
        data = self.macd_indicator.calculate_macd(data)
        data = self.bb_indicator.calculate_bollinger_bands(data)
        data = self.stoch_indicator.calculate_stochastic(data)

        # 각 지표 신호 수집
        rsi_signal = self.rsi_indicator.get_rsi_signal(data)
        macd_signal = self.macd_indicator.get_macd_signal(data)
        bb_signal = self.bb_indicator.get_bollinger_signal(data)
        stoch_signal = self.stoch_indicator.get_stochastic_signal(data)

        # Majority Voting (다수결)
        signals = [rsi_signal, macd_signal, bb_signal, stoch_signal]
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
    
    strategy = BasicStrategy()
    signal = strategy.apply_strategy(data)
    print(f"기본 전략 신호: {signal}")
