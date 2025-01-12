import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton
from PyQt5.QtCore import QTimer
import requests

class MainDashboard(QWidget):
    def __init__(self, api_key, api_url):
        super().__init__()

        # API 키와 URL
        self.api_key = api_key
        self.api_url = api_url

        # UI 초기화
        self.initUI()

        # 데이터 갱신 타이머 설정
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateData)
        self.timer.start(2000)  # 2초마다 갱신

    def initUI(self):
        # UI 요소 정의
        self.account_label = QLabel("계좌 정보: 불러오는 중...")
        self.profit_label = QLabel("수익률: 불러오는 중...")
        self.market_label = QLabel("실시간 시세: 불러오는 중...")
        self.indicators_label = QLabel("지표: 불러오는 중...")
        self.portfolio_label = QLabel("포트폴리오: 불러오는 중...")
        self.ai_signal_label = QLabel("AI 신호: 불러오는 중...")
        self.execute_button = QPushButton("매매 실행")
        self.execute_button.clicked.connect(self.executeTrade)

        # 레이아웃 구성
        layout = QVBoxLayout()
        layout.addWidget(self.account_label)
        layout.addWidget(self.profit_label)
        layout.addWidget(self.market_label)
        layout.addWidget(self.indicators_label)
        layout.addWidget(self.portfolio_label)
        layout.addWidget(self.ai_signal_label)
        layout.addWidget(self.execute_button)
        self.setLayout(layout)

        # 창 설정
        self.setWindowTitle("코인 매매 프로그램")
        self.setGeometry(100, 100, 800, 600)

    def fetch_account_data(self):
        """API를 통해 계좌 데이터를 가져옴"""
        try:
            response = requests.get(f"{self.api_url}/account", headers={"Authorization": f"Bearer {self.api_key}"})
            data = response.json()
            return data
        except Exception as e:
            print(f"계좌 데이터 가져오기 오류: {e}")
            return {"krw": 0, "coins": {}, "daily_profit": "N/A", "monthly_profit": "N/A"}

    def fetch_portfolio_data(self):
        """API를 통해 포트폴리오 데이터를 가져옴"""
        try:
            response = requests.get(f"{self.api_url}/portfolio", headers={"Authorization": f"Bearer {self.api_key}"})
            data = response.json()
            return data
        except Exception as e:
            print(f"포트폴리오 데이터 가져오기 오류: {e}")
            return {}

    def fetch_market_data(self):
        """API를 통해 실시간 시세 데이터를 가져옴"""
        try:
            response = requests.get(f"{self.api_url}/market", headers={"Authorization": f"Bearer {self.api_key}"})
            data = response.json()
            return data
        except Exception as e:
            print(f"시장 데이터 가져오기 오류: {e}")
            return {}

    def fetch_ai_signals(self):
        """AI 모듈에서 신호 데이터를 가져옴"""
        try:
            response = requests.get(f"{self.api_url}/ai_signals", headers={"Authorization": f"Bearer {self.api_key}"})
            data = response.json()
            return data
        except Exception as e:
            print(f"AI 신호 데이터 가져오기 오류: {e}")
            return {}

    def updateData(self):
        """실시간 데이터 갱신"""
        # 계좌 데이터
        account_data = self.fetch_account_data()
        self.account_label.setText(f"계좌 잔고: {account_data['krw']} KRW | "
                                   f"{', '.join([f'{coin}: {amt}' for coin, amt in account_data['coins'].items()])}")
        self.profit_label.setText(f"수익률: {account_data['daily_profit']}% (일간) | "
                                  f"{account_data['monthly_profit']}% (월간)")

        # 포트폴리오 데이터
        portfolio_data = self.fetch_portfolio_data()
        portfolio_text = "\n".join([f"- {coin}: {amt} ({value} KRW)" for coin, (amt, value) in portfolio_data.items()])
        self.portfolio_label.setText(f"[포트폴리오]\n{portfolio_text}")

        # 실시간 시세 데이터
        market_data = self.fetch_market_data()
        self.market_label.setText(f"실시간 시세: {', '.join([f'{coin}: {price} KRW' for coin, price in market_data.items()])}")

        # AI 신호 데이터
        ai_signal = self.fetch_ai_signals()
        self.ai_signal_label.setText(f"[AI 매매 신호]\n매수 신호 강도: {ai_signal['buy_strength']} | "
                                     f"매도 신호 강도: {ai_signal['sell_strength']}\n"
                                     f"전략: {ai_signal['strategy']} | 예상 수익률: {ai_signal['expected_profit']}%")

    def executeTrade(self):
        """매매 실행 버튼 동작"""
        try:
            response = requests.post(f"{self.api_url}/execute_trade", headers={"Authorization": f"Bearer {self.api_key}"})
            if response.status_code == 200:
                print("매매 실행 성공")
            else:
                print(f"매매 실행 실패: {response.text}")
        except Exception as e:
            print(f"매매 실행 요청 오류: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    dashboard = MainDashboard(api_key="your_api_key", api_url="https://your-api-url.com")
    dashboard.show()
    sys.exit(app.exec_())
