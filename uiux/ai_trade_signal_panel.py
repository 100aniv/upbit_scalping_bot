import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QCheckBox
from PyQt5.QtCore import QTimer
import requests

class AITradeSignalPanel(QWidget):
    def __init__(self, api_key, api_url):
        super().__init__()

        self.api_key = api_key
        self.api_url = api_url

        # UI 구성
        self.initUI()

    def initUI(self):
        self.setWindowTitle("AI Trade Signal Panel")

        self.label_strategy = QLabel("전략: 불러오는 중...")
        self.label_ai_confidence = QLabel("AI 신뢰도: 불러오는 중...")
        self.label_risk_score = QLabel("리스크 점수: 불러오는 중...")
        self.label_take_profit = QLabel("이익 실현: 불러오는 중...")
        self.label_stop_loss = QLabel("손절매: 불러오는 중...")
        self.label_trailing_stop = QLabel("트레일링 스탑: 불러오는 중...")

        # 트레일링 스탑 활성화 체크박스
        self.trailing_stop_checkbox = QCheckBox("트레일링 스탑 활성화")
        self.trailing_stop_checkbox.stateChanged.connect(self.toggle_trailing_stop)

        # 버튼 추가
        self.btn_auto_trade = QPushButton("매매 자동 실행: OFF")
        self.btn_auto_trade.setCheckable(True)
        self.btn_auto_trade.clicked.connect(self.toggleAutoTrade)

        self.btn_optimize_risk = QPushButton("리스크 최적화 재계산")
        self.btn_optimize_risk.clicked.connect(self.optimize_risk)

        # 레이아웃 구성
        layout = QVBoxLayout()
        layout.addWidget(self.label_strategy)
        layout.addWidget(self.label_ai_confidence)
        layout.addWidget(self.label_risk_score)
        layout.addWidget(self.label_take_profit)
        layout.addWidget(self.label_stop_loss)
        layout.addWidget(self.label_trailing_stop)
        layout.addWidget(self.trailing_stop_checkbox)

        button_layout = QVBoxLayout()
        button_layout.addWidget(self.btn_auto_trade)
        button_layout.addWidget(self.btn_optimize_risk)

        layout.addLayout(button_layout)
        self.setLayout(layout)

        # API 데이터 가져오기
        self.fetchData()

    def fetchData(self):
        try:
            response = requests.get(f"{self.api_url}/ai_trade_signals", headers={"Authorization": f"Bearer {self.api_key}"})
            data = response.json()
            self.label_strategy.setText(f"전략: {data['strategy']}")
            self.label_ai_confidence.setText(f"AI 신뢰도: {data['ai_confidence']}%")
            self.label_risk_score.setText(f"리스크 점수: {data['risk_score']}%")
            self.label_take_profit.setText(f"이익 실현: {data['take_profit']}%")
            self.label_stop_loss.setText(f"손절매: {data['stop_loss']}%")
            trailing_stop_status = "활성화" if data['trailing_stop'] else "비활성화"
            self.label_trailing_stop.setText(f"트레일링 스탑: {trailing_stop_status}")
            self.trailing_stop_checkbox.setChecked(data['trailing_stop'])
        except Exception as e:
            self.label_strategy.setText("데이터 로딩 실패")

    def optimize_risk(self):
        try:
            response = requests.post(f"{self.api_url}/optimize_risk", headers={"Authorization": f"Bearer {self.api_key}"})
            if response.status_code == 200:
                self.fetchData()
        except Exception as e:
            print(f"리스크 최적화 오류: {e}")

    def toggle_trailing_stop(self, state):
        try:
            is_active = bool(state)
            response = requests.post(f"{self.api_url}/toggle_trailing_stop", json={"trailing_stop": is_active}, headers={"Authorization": f"Bearer {self.api_key}"})
            if response.status_code == 200:
                self.fetchData()
        except Exception as e:
            print(f"트레일링 스탑 전환 오류: {e}")

    def toggleAutoTrade(self):
        current_state = self.btn_auto_trade.isChecked()
        self.btn_auto_trade.setText("매매 자동 실행: ON" if current_state else "매매 자동 실행: OFF")
        try:
            response = requests.post(f"{self.api_url}/toggle_auto_trade", json={"auto_trade": current_state}, headers={"Authorization": f"Bearer {self.api_key}"})
            if response.status_code == 200:
                print("매매 자동 실행 상태 변경 완료")
        except Exception as e:
            print(f"매매 자동 실행 전환 오류: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    panel = AITradeSignalPanel(api_key="your_api_key", api_url="https://your-api-url.com")
    panel.show()
    sys.exit(app.exec_())
