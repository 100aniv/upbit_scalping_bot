# alerts.py

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QListWidget
import requests

class AlertsPanel(QWidget):
    def __init__(self, api_url, api_key):
        super().__init__()
        self.api_url = api_url
        self.api_key = api_key

        self.initUI()

    def initUI(self):
        self.setWindowTitle("Alerts & Notifications")
        self.setGeometry(100, 100, 800, 600)

        # Main Layout
        layout = QVBoxLayout(self)

        # Section: Alerts and Notifications
        self.alerts_label = QLabel("실시간 경고 시스템", self)
        layout.addWidget(self.alerts_label)

        self.alerts_list = QListWidget(self)
        layout.addWidget(self.alerts_list)

        # Section: Notification Settings
        self.settings_label = QLabel("알림 설정", self)
        layout.addWidget(self.settings_label)

        self.notification_settings = QLabel(
            "📧 이메일 경고 | 📱 푸시 알림 | 📩 SMS 경고", self
        )
        layout.addWidget(self.notification_settings)

        self.refresh_alerts()

    def refresh_alerts(self):
        """Fetches and updates the alerts."""
        alerts = self.fetch_alerts()
        self.update_alerts_list(alerts)

    def fetch_alerts(self):
        """Fetch alerts data from the API."""
        try:
            response = requests.get(
                f"{self.api_url}/alerts", headers={"Authorization": f"Bearer {self.api_key}"}
            )
            return response.json()
        except Exception as e:
            print(f"Error fetching alerts: {e}")
            return []

    def update_alerts_list(self, alerts):
        """Update the alerts list with fetched data."""
        self.alerts_list.clear()
        for alert in alerts:
            icon = "🔴" if alert["type"] == "warning" else "🟢"
            self.alerts_list.addItem(f"{icon} {alert['message']}")
