# strategies/backtesting.py
import pandas as pd
from strategies.basic_strategy import BasicStrategy
from strategies.ai_strategy import AIStrategy

class BacktestingEngine:
    def __init__(self, strategy, initial_balance=1000000):
        """
        백테스팅 엔진 생성자
        :param strategy: 적용할 매매 전략
        :param initial_balance: 초기 자본
        """
        self.strategy = strategy
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0
        self.trade_history = []

    def run_backtest(self, data):
        """
        백테스팅 실행
        :param data: OHLCV 데이터프레임
        """
        for index, row in data.iterrows():
            signal = self.strategy.apply_strategy(data.iloc[:index + 1])

            if signal == "Buy" and self.balance > 0:
                self.position = self.balance / row['close']
                self.balance = 0
                self.trade_history.append((row['close'], "Buy"))

            elif signal == "Sell" and self.position > 0:
                self.balance = self.position * row['close']
                self.position = 0
                self.trade_history.append((row['close'], "Sell"))

        final_value = self.balance + (self.position * data.iloc[-1]['close'])
        return final_value, self.trade_history

if __name__ == "__main__":
    import pyupbit
    data = pyupbit.get_ohlcv("KRW-BTC", interval="minute1", count=200)
    basic_strategy = BasicStrategy()
    backtester = BacktestingEngine(strategy=basic_strategy)
    final_value, trades = backtester.run_backtest(data)
    print(f"최종 잔액: {final_value}")
    print(f"거래 내역: {trades}")