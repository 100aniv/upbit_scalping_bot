from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QCheckBox, QTableWidget, QTableWidgetItem
import sys

class MainDashboard(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        layout = QVBoxLayout()

        # 계좌 잔고 섹션
        self.account_balance = QLabel("계좌 잔고: KRW 5,000,000 | BTC 0.25 | ETH 1.2")
        self.profit_rate = QLabel("수익률: +8.5% (일간) | +22.4% (월간)")

        # 실시간 시세 섹션
        self.chart_info = QLabel("[ 실시간 시세 차트 (캔들 + 볼린저밴드) ]")
        self.market_price = QLabel("BTC/KRW | 현재가: 72,000,000 KRW")
        self.macd_status = QLabel("MACD: 골든크로스 | RSI: 67 (과매수)")
        self.ai_confidence = QLabel("AI Confidence: 87%")

        # 포트폴리오 현황
        self.portfolio_info = QLabel("[ 포트폴리오 현황 ]")
        self.btc_info = QLabel("- BTC: 0.25 (72,000,000 KRW)")
        self.eth_info = QLabel("- ETH: 1.2 (5,000,000 KRW)")
        self.risk_score = QLabel("리스크 점수: 중간 (45%)")

        # 최근 체결 내역 (테이블 추가)
        self.recent_trades = QLabel("[ 최근 체결 내역 ]")
        self.trade_table = QTableWidget(2, 3)
        self.trade_table.setHorizontalHeaderLabels(["자산", "체결 수량", "가격"])
        self.trade_table.setItem(0, 0, QTableWidgetItem("BTC"))
        self.trade_table.setItem(0, 1, QTableWidgetItem("0.05 BTC"))
        self.trade_table.setItem(0, 2, QTableWidgetItem("71,500,000 KRW"))
        self.trade_table.setItem(1, 0, QTableWidgetItem("ETH"))
        self.trade_table.setItem(1, 1, QTableWidgetItem("1.0 ETH"))
        self.trade_table.setItem(1, 2, QTableWidgetItem("5,100,000 KRW"))

        # AI 매매 신호 패널
        self.trade_signal_info = QLabel("[ AI 매매 신호 패널 ]")
        self.buy_signal_strength = QLabel("매수 신호 강도: ★★★★☆")
        self.sell_signal_strength = QLabel("매도 신호 강도: ★★☆☆☆")
        self.expected_return = QLabel("예상 수익률: 5% | AI 신뢰도: 87%")

        # 매매 실행 버튼 및 체크박스
        self.auto_trade_checkbox = QCheckBox("매매 자동 실행 [On/Off]")
        self.execute_trade_button = QPushButton("매매 실행")
        self.backtest_button = QPushButton("백테스팅 시작")
        self.strategy_button = QPushButton("전략 변경")

        # 레이아웃에 추가
        layout.addWidget(self.account_balance)
        layout.addWidget(self.profit_rate)
        layout.addWidget(self.chart_info)
        layout.addWidget(self.market_price)
        layout.addWidget(self.macd_status)
        layout.addWidget(self.ai_confidence)
        layout.addWidget(self.portfolio_info)
        layout.addWidget(self.btc_info)
        layout.addWidget(self.eth_info)
        layout.addWidget(self.risk_score)
        layout.addWidget(self.recent_trades)
        layout.addWidget(self.trade_table)
        layout.addWidget(self.trade_signal_info)
        layout.addWidget(self.buy_signal_strength)
        layout.addWidget(self.sell_signal_strength)
        layout.addWidget(self.expected_return)
        layout.addWidget(self.auto_trade_checkbox)
        layout.addWidget(self.execute_trade_button)
        layout.addWidget(self.backtest_button)
        layout.addWidget(self.strategy_button)

        self.setLayout(layout)
        self.setWindowTitle("메인 대시보드")
        self.resize(400, 600)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_dashboard = MainDashboard()
    main_dashboard.show()
    sys.exit(app.exec_())
