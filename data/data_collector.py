# data/data_collector.py
import pyupbit
import websocket
import json
import time
import pandas as pd

class DataCollector:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = []
        self.ws = None

    def get_historical_data(self, count=200):
        """과거 OHLCV 데이터 수집"""
        try:
            data = pyupbit.get_ohlcv(self.ticker, interval="minute1", count=count)
            if data is None:
                raise ValueError("데이터를 가져오는 데 실패했습니다.")
            return data
        except Exception as e:
            print(f"[Error] {e}")
            return None

    def on_message(self, ws, message):
        """WebSocket으로 실시간 데이터 수신"""
        try:
            data = json.loads(message)
            self.data.append(data)
            print(f"[실시간 데이터]: {data}")
        except Exception as e:
            print(f"[Error] 데이터 수신 오류: {e}")

    def on_error(self, ws, error):
        print(f"[Error] WebSocket 오류 발생: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print("[Info] WebSocket 연결이 종료되었습니다. 재연결 중...")
        time.sleep(5)
        self.start_realtime_stream()

    def on_open(self, ws):
        """WebSocket 구독 메시지 전송"""
        payload = [
            {"ticket": "test"},
            {"type": "ticker", "codes": [self.ticker]},
            {"format": "DEFAULT"}
        ]
        ws.send(json.dumps(payload))
        print(f"[Info] {self.ticker} 데이터 수집 시작")

    def start_realtime_stream(self):
        """실시간 데이터 수집 WebSocket 연결 시작"""
        self.ws = websocket.WebSocketApp(
            "wss://api.upbit.com/websocket/v1",
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        self.ws.on_open = self.on_open
        self.ws.run_forever()

    def stop_stream(self):
        """WebSocket 연결 종료"""
        if self.ws:
            self.ws.close()

if __name__ == "__main__":
    collector = DataCollector("KRW-BTC")
    historical_data = collector.get_historical_data()
    print("과거 데이터 수집 완료:")
    print(historical_data.tail())

    # 실시간 데이터 수집 실행
    try:
        print("실시간 데이터 수집을 시작합니다.")
        collector.start_realtime_stream()
    except KeyboardInterrupt:
        collector.stop_stream()
        print("데이터 수집 종료")
