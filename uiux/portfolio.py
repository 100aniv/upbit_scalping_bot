import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton
import requests

class PortfolioManagementPanel(QWidget):
    def __init__(self, api_key, api_url):
        super().__init__()
        self.api_key = api_key
        self.api_url = api_url

        # UI 구성
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Portfolio Management Panel")

        # 라벨 구성
        self.label_portfolio = QLabel("포트폴리오 현황 불러오는 중...")
        self.label_total_value = QLabel("총 평가 금액: 불러오는 중...")
        self.label_total_percentage = QLabel("총 자산 비중: 불러오는 중...")

        # 버튼 추가
        self.btn_report_download = QPushButton("데이터 리포트 다운로드")
        self.btn_report_download.clicked.connect(self.downloadReport)

        self.btn_risk_optimize = QPushButton("리스크 최적화 재계산")
        self.btn_risk_optimize.clicked.connect(self.optimizeRisk)

        # 레이아웃 구성
        layout = QVBoxLayout()
        layout.addWidget(self.label_portfolio)
        layout.addWidget(self.label_total_value)
        layout.addWidget(self.label_total_percentage)
        layout.addWidget(self.btn_report_download)
        layout.addWidget(self.btn_risk_optimize)

        self.setLayout(layout)

        # 초기 데이터 로딩
        self.fetchData()

    def fetchData(self):
        try:
            response = requests.get(f"{self.api_url}/portfolio", headers={"Authorization": f"Bearer {self.api_key}"})
            data = response.json()
            self.label_portfolio.setText(f"포트폴리오: {data['portfolio']}")
            self.label_total_value.setText(f"총 평가 금액: {data['total_value']} KRW")
            self.label_total_percentage.setText(f"총 자산 비중: {data['total_percentage']}%")
        except Exception as e:
            self.label_portfolio.setText("데이터 로딩 실패")

    def optimizeRisk(self):
        try:
            response = requests.post(f"{self.api_url}/optimize_portfolio_risk", headers={"Authorization": f"Bearer {self.api_key}"})
            if response.status_code == 200:
                self.fetchData()
        except Exception as e:
            print(f"리스크 최적화 오류: {e}")

    def downloadReport(self):
        try:
            response = requests.get(f"{self.api_url}/download_report", headers={"Authorization": f"Bearer {self.api_key}"})
            if response.status_code == 200:
                with open("portfolio_report.pdf", "wb") as file:
                    file.write(response.content)
                print("데이터 리포트 다운로드 완료")
            else:
                print("리포트 다운로드 실패")
        except Exception as e:
            print(f"리포트 다운로드 오류: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    panel = PortfolioManagementPanel(api_key="your_api_key", api_url="https://your-api-url.com")
    panel.show()
    sys.exit(app.exec_())
