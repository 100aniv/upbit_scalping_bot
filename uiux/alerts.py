import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QCheckBox, QPushButton, QMessageBox

class RealTimeAlertPanel(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("실시간 경고 시스템")
        self.setGeometry(100, 100, 600, 400)
        self.initUI()

    def initUI(self):
        # 메인 레이아웃 설정
        main_widget = QWidget()
        layout = QVBoxLayout()

        # 경고 패널 레이블
        self.alert_label = QLabel("[실시간 경고 시스템]")
        layout.addWidget(self.alert_label)

        # 경고 상태 레이블
        self.btc_alert = QLabel("🔴 BTC 가격 급락 경고: RSI 25 (과매도)")
        self.eth_alert = QLabel("🔴 ETH 수익률 하락: -5% (손절매 발동)")
        self.ai_confidence = QLabel("✅ AI 신뢰도 상승: 92% (매수 신호 강함)")

        # 알림 옵션 체크박스
        self.email_alert = QCheckBox("📧 이메일 경고")
        self.push_alert = QCheckBox("📲 푸시 알림")
        self.sms_alert = QCheckBox("📩 SMS 경고")

        # 버튼 설정
        self.alert_button = QPushButton("경고 알림 테스트")
        self.alert_button.clicked.connect(self.trigger_alert)

        # 레이아웃 추가
        layout.addWidget(self.btc_alert)
        layout.addWidget(self.eth_alert)
        layout.addWidget(self.ai_confidence)
        layout.addWidget(self.email_alert)
        layout.addWidget(self.push_alert)
        layout.addWidget(self.sms_alert)
        layout.addWidget(self.alert_button)

        # 메인 위젯에 레이아웃 적용
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

    # 경고 알림 테스트 함수
    def trigger_alert(self):
        message = "📢 경고 알림 테스트 메시지!"
        if self.email_alert.isChecked():
            message += "\n- 이메일 알림 발송됨"
        if self.push_alert.isChecked():
            message += "\n- 푸시 알림 발송됨"
        if self.sms_alert.isChecked():
            message += "\n- SMS 알림 발송됨"

        QMessageBox.information(self, "경고 알림", message)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RealTimeAlertPanel()
    window.show()
    sys.exit(app.exec_())
