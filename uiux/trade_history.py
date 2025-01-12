import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QTableWidget, QTableWidgetItem, QPushButton, QChartView
from PyQt5.QtChart import QChart, QLineSeries, QValueAxis

class TradeHistoryPanel(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("거래 이력 및 백테스팅 패널")
        self.setGeometry(100, 100, 800, 600)
        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # 거래 내역 테이블
        trade_history_label = QLabel("[거래 내역]")
        trade_table = QTableWidget(2, 4)
        trade_table.setHorizontalHeaderLabels(["자산", "거래 유형", "금액 (KRW)", "수익률"])
        trade_table.setItem(0, 0, QTableWidgetItem("BTC"))
        trade_table.setItem(0, 1, QTableWidgetItem("매수"))
        trade_table.setItem(0, 2, QTableWidgetItem("72,000,000"))
        trade_table.setItem(0, 3, QTableWidgetItem("+3% 수익"))  # 강조 적용
        trade_table.setItem(1, 0, QTableWidgetItem("ETH"))
        trade_table.setItem(1, 1, QTableWidgetItem("매도"))
        trade_table.setItem(1, 2, QTableWidgetItem("5,100,000"))
        trade_table.setItem(1, 3, QTableWidgetItem("-1% 손실"))  # 강조 적용

        # 백테스팅 결과
        backtest_label = QLabel("[백테스팅 결과]")
        backtest_info = QLabel("기간: 2023.01.01 ~ 2024.01.01\n총 수익률: +120%\n승률: 75%\n최대 손실폭: -12%")

        # 전략 비교 섹션
        strategy_label = QLabel("[전략 비교]")
        strategy_comparison = QLabel(
            "전략 A: MACD + RSI (70% 수익)\n"
            "전략 B: MACD + LSTM (120% 수익)\n"
            "전략 C: Bollinger + EMA (50% 수익)"
        )

        # 버튼 섹션 추가
        ab_test_button = QPushButton("A/B 전략 테스트 실행")
        strategy_optimize_button = QPushButton("전략 최적화")  # 추가된 버튼

        # 수익률 그래프
        chart = QChart()
        series = QLineSeries()
        series.append(0, 50)
        series.append(1, 70)
        series.append(2, 120)
        chart.addSeries(series)
        axisX = QValueAxis()
        axisY = QValueAxis()
        chart.setAxisX(axisX, series)
        chart.setAxisY(axisY, series)
        chart_view = QChartView(chart)
        chart.setTitle("수익률 그래프")

        # 레이아웃 추가
        main_layout.addWidget(trade_history_label)
        main_layout.addWidget(trade_table)
        main_layout.addWidget(backtest_label)
        main_layout.addWidget(backtest_info)
        main_layout.addWidget(strategy_label)
        main_layout.addWidget(strategy_comparison)
        main_layout.addWidget(ab_test_button)
        main_layout.addWidget(strategy_optimize_button)  # 추가된 버튼
        main_layout.addWidget(chart_view)

        # 메인 위젯에 레이아웃 적용
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TradeHistoryPanel()
    window.show()
    sys.exit(app.exec_())
