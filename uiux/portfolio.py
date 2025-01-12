import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QTableWidget, QTableWidgetItem, QProgressBar, QChartView
from PyQt5.QtChart import QChart, QPieSeries
from PyQt5.QtCore import Qt

class PortfolioPanel(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("포트폴리오 관리 패널")
        self.setGeometry(100, 100, 600, 600)
        self.initUI()

    def initUI(self):
        # 메인 레이아웃 설정
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # 포트폴리오 정보 레이블
        title_label = QLabel("[포트폴리오 현황]")
        title_label.setAlignment(Qt.AlignCenter)

        # 자산 테이블
        self.table = QTableWidget(3, 3)
        self.table.setHorizontalHeaderLabels(["자산", "수량", "평가액 (KRW)"])
        self.table.setItem(0, 0, QTableWidgetItem("BTC"))
        self.table.setItem(0, 1, QTableWidgetItem("0.3"))
        self.table.setItem(0, 2, QTableWidgetItem("72,000,000"))
        self.table.setItem(1, 0, QTableWidgetItem("ETH"))
        self.table.setItem(1, 1, QTableWidgetItem("1.5"))
        self.table.setItem(1, 2, QTableWidgetItem("6,000,000"))
        self.table.setItem(2, 0, QTableWidgetItem("XRP"))
        self.table.setItem(2, 1, QTableWidgetItem("500"))
        self.table.setItem(2, 2, QTableWidgetItem("2,000,000"))

        # 리스크 점수 바
        risk_label = QLabel("AI 리스크 점수: 45% (중간)")
        self.risk_bar = QProgressBar()
        self.risk_bar.setValue(45)

        # 포트폴리오 파이차트
        self.chart = QChart()
        self.pie_series = QPieSeries()
        self.pie_series.append("BTC", 60)
        self.pie_series.append("ETH", 30)
        self.pie_series.append("XRP", 10)
        self.chart.addSeries(self.pie_series)
        self.chart.setTitle("포트폴리오 비중")
        self.chart_view = QChartView(self.chart)

        # 레이아웃에 위젯 추가
        main_layout.addWidget(title_label)
        main_layout.addWidget(self.table)
        main_layout.addWidget(risk_label)
        main_layout.addWidget(self.risk_bar)
        main_layout.addWidget(self.chart_view)

        # 메인 위젯에 레이아웃 설정
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PortfolioPanel()
    window.show()
    sys.exit(app.exec_())
