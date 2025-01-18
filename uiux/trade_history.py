# trade_history.py

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem
from PyQt5.QtCore import Qt
import requests

class TradeHistory(QWidget):
    def __init__(self, api_url, api_key):
        super().__init__()
        self.api_url = api_url
        self.api_key = api_key

        self.initUI()

    def initUI(self):
        self.setWindowTitle("Trade History & Backtesting Panel")
        self.setGeometry(100, 100, 800, 600)

        # Main Layout
        layout = QVBoxLayout(self)

        # Section: Trade History
        self.trade_history_label = QLabel("거래 내역", self)
        layout.addWidget(self.trade_history_label)

        self.trade_history_table = QTableWidget(self)
        self.trade_history_table.setColumnCount(4)
        self.trade_history_table.setHorizontalHeaderLabels(["일시", "종목", "거래 유형", "수익률"])
        layout.addWidget(self.trade_history_table)

        # Section: Backtesting Results
        self.backtesting_label = QLabel("백테스팅 결과", self)
        layout.addWidget(self.backtesting_label)

        self.backtesting_table = QTableWidget(self)
        self.backtesting_table.setColumnCount(4)
        self.backtesting_table.setHorizontalHeaderLabels(["기간", "총 수익률", "승률", "최대 손실폭"])
        layout.addWidget(self.backtesting_table)

        self.refresh_data()

    def refresh_data(self):
        """Fetches and updates the trade history and backtesting results."""
        trade_data = self.fetch_trade_history()
        self.update_trade_history_table(trade_data)

        backtesting_data = self.fetch_backtesting_results()
        self.update_backtesting_table(backtesting_data)

    def fetch_trade_history(self):
        """Fetch trade history data from the API."""
        try:
            response = requests.get(f"{self.api_url}/trade-history", headers={"Authorization": f"Bearer {self.api_key}"})
            return response.json()
        except Exception as e:
            print(f"Error fetching trade history: {e}")
            return []

    def fetch_backtesting_results(self):
        """Fetch backtesting results data from the API."""
        try:
            response = requests.get(f"{self.api_url}/backtesting-results", headers={"Authorization": f"Bearer {self.api_key}"})
            return response.json()
        except Exception as e:
            print(f"Error fetching backtesting results: {e}")
            return []

    def update_trade_history_table(self, data):
        """Update the trade history table with fetched data."""
        self.trade_history_table.setRowCount(len(data))
        for row, trade in enumerate(data):
            self.trade_history_table.setItem(row, 0, QTableWidgetItem(trade.get("date", "N/A")))
            self.trade_history_table.setItem(row, 1, QTableWidgetItem(trade.get("symbol", "N/A")))
            self.trade_history_table.setItem(row, 2, QTableWidgetItem(trade.get("type", "N/A")))
            self.trade_history_table.setItem(row, 3, QTableWidgetItem(trade.get("return", "N/A")))

    def update_backtesting_table(self, data):
        """Update the backtesting results table with fetched data."""
        self.backtesting_table.setRowCount(len(data))
        for row, result in enumerate(data):
            self.backtesting_table.setItem(row, 0, QTableWidgetItem(result.get("period", "N/A")))
            self.backtesting_table.setItem(row, 1, QTableWidgetItem(result.get("total_return", "N/A")))
            self.backtesting_table.setItem(row, 2, QTableWidgetItem(result.get("win_rate", "N/A")))
            self.backtesting_table.setItem(row, 3, QTableWidgetItem(result.get("max_drawdown", "N/A")))
