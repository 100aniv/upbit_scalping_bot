import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QHBoxLayout
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor
import requests
import websocket
import json

class MainDashboard(QWidget):
    def __init__(self, api_key, api_url, websocket_url):
        super().__init__()

        # API 및 WebSocket 설정
        self.api_key = api_key
        self.api_url = api_url
        self.websocket_url = websocket_url

        # UI 초기화
        self.initUI()

        # 데이터 갱신 타이머 설정
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateData)
        self.timer.start(2000)  # 2초마다 갱신

        # WebSocket 스레드 시작
        self.websocket_thread = WebSocketThread(self.websocket_url, self.api_key)
        self.websocket_thread.update_signal.connect(self.update_from_websocket)
        self.websocket_thread.start()

    def initUI(self):
        # UI 요소 정의
        title_font = QFont("Arial", 16, QFont.Bold)

        self.account_label = QLabel("계좌 정보: 불러오는 중...")
        self.account_label.setFont(title_font)

        self.profit_label = QLabel("수익률: 불러오는 중...")
        self.market_label = QLabel("실시간 시세: 불러오는 중...")
        self.indicators_label = QLabel("지표: 불러오는 중...")
        self.portfolio_label = QLabel("포트폴리오: 불러오는 중...")
        self.ai_signal_label = QLabel("AI 신호: 불러오는 중...")
        self.trail_stop_label = QLabel("트레일링 스탑: 불러오는 중...")

        self.execute_button = QPushButton("매매 실행")
        self.execute_button.clicked.connect(self.executeTrade)

        self.strategy_button = QPushButton("전략 변경")
        self.strategy_button.clicked.connect(self.changeStrategy)

        # 레이아웃 구성
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.account_label)
        main_layout.addWidget(self.profit_label)
        main_layout.addWidget(self.market_label)
        main_layout.addWidget(self.indicators_label)
        main_layout.addWidget(self.portfolio_label)
        main_layout.addWidget(self.ai_signal_label)
        main_layout.addWidget(self.trail_stop_label)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.execute_button)
        button_layout.addWidget(self.strategy_button)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

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

        # AI 신호 데이터
        ai_signal = self.fetch_ai_signals()
        self.ai_signal_label.setText(f"[AI 매매 신호]\n매수 신호 강도: {ai_signal['buy_strength']} | "
                                     f"매도 신호 강도: {ai_signal['sell_strength']}\n"
                                     f"전략: {ai_signal['strategy']} | 예상 수익률: {ai_signal['expected_profit']}%")

        # 트레일링 스탑
        self.trail_stop_label.setText(f"트레일링 스탑 활성화: {ai_signal['trailing_stop']}")

    def update_from_websocket(self, message):
        """WebSocket 메시지를 받아 GUI 업데이트"""
        try:
            data = json.loads(message)
            self.market_label.setText(f"실시간 시세: {', '.join([f'{coin}: {price} KRW' for coin, price in data.items()])}")
        except Exception as e:
            print(f"WebSocket 데이터 처리 오류: {e}")

    def executeTrade(self):
        """매매 실행 버튼 동작"""
        print("매매 실행 요청 전송...")

    def changeStrategy(self):
        """전략 변경 버튼 동작"""
        print("전략 변경 요청 전송...")

class WebSocketThread(QThread):
    update_signal = pyqtSignal(str)

    def __init__(self, websocket_url, api_key):
        super().__init__()
        self.websocket_url = websocket_url
        self.api_key = api_key

    def run(self):
        """WebSocket 연결 시작"""
        def on_message(ws, message):
            self.update_signal.emit(message)

        ws = websocket.WebSocketApp(self.websocket_url, on_message=on_message)
        ws.run_forever()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    dashboard = MainDashboard(api_key="your_api_key", api_url="https://your-api-url.com", websocket_url="wss://your-websocket-url.com")
    dashboard.show()
    sys.exit(app.exec_())
