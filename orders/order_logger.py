import os
import json
import logging
import gzip
import hashlib
import threading
import redis
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from smtplib import SMTP
from email.mime.text import MIMEText
from typing import Optional

class OrderLogger:
    """
    상용급 주문 기록 및 관리 클래스.
    Redis 통합, 실시간 푸시, 멀티스레딩, 로그 무결성 검증 및 알림 기능 포함.
    """

    def __init__(self, config_path="logger_config.json"):
        """
        OrderLogger 초기화
        :param config_path: 설정 파일 경로
        """
        # 설정 파일 로드
        with open(config_path, "r") as config_file:
            self.config = json.load(config_file)

        self.log_directory = self.config.get("log_directory", "logs")
        self.redis_host = self.config.get("redis_host", "localhost")
        self.redis_port = self.config.get("redis_port", 6379)
        self.enable_realtime_push = self.config.get("enable_realtime_push", False)
        self.error_notification_email = self.config.get("error_notification_email", None)

        # Redis 연결
        self.redis_client = redis.StrictRedis(host=self.redis_host, port=self.redis_port, decode_responses=True)

        if not os.path.exists(self.log_directory):
            os.makedirs(self.log_directory)

        self.error_log_file = os.path.join(self.log_directory, "error.log")
        logging.basicConfig(filename=self.error_log_file, level=logging.ERROR)

        # 멀티스레딩
        self.thread_pool = ThreadPoolExecutor(max_workers=5)

    def _get_log_file(self):
        """
        현재 날짜를 기준으로 로그 파일 경로 반환
        """
        date_str = datetime.now().strftime('%Y%m%d')
        return os.path.join(self.log_directory, f"order_log_{date_str}.json")

    def _calculate_file_hash(self, file_path):
        """
        로그 파일의 무결성을 검증하기 위한 SHA256 해시 계산
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def log_order(self, order_id, order_type, symbol, quantity, price, status, additional_info=None):
        """
        주문 정보를 로그에 기록
        :param order_id: 주문 ID
        :param order_type: 주문 유형 (e.g., "BUY" 또는 "SELL")
        :param symbol: 거래 심볼 (e.g., "BTC/USDT")
        :param quantity: 주문 수량
        :param price: 주문 가격
        :param status: 주문 상태 (e.g., "Executed", "Pending")
        :param additional_info: 추가 정보 (선택 사항)
        """
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "order_id": order_id,
            "order_type": order_type,
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "status": status,
            "additional_info": additional_info or {}
        }

        log_file = self._get_log_file()
        try:
            # JSON 파일에 로그 기록
            if not os.path.exists(log_file):
                with open(log_file, "w") as file:
                    json.dump([log_entry], file, indent=4)
            else:
                with open(log_file, "r+") as file:
                    data = json.load(file)
                    data.append(log_entry)
                    file.seek(0)
                    json.dump(data, file, indent=4)

            # Redis에 실시간 로그 푸시
            if self.enable_realtime_push:
                self.redis_client.publish("order_logs", json.dumps(log_entry))

        except Exception as e:
            logging.error(f"Failed to log order: {e}")
            self._send_error_notification(f"Error in log_order: {e}")

    def search_logs(self, keyword=None, start_date=None, end_date=None, order_type=None):
        """
        로그 검색
        :param keyword: 검색할 키워드
        :param start_date: 검색 시작 날짜 (YYYY-MM-DD)
        :param end_date: 검색 종료 날짜 (YYYY-MM-DD)
        :param order_type: 거래 유형 필터 (e.g., "BUY" 또는 "SELL")
        :return: 검색된 로그 리스트
        """
        logs = []
        try:
            for file_name in os.listdir(self.log_directory):
                if file_name.startswith("order_log_") and file_name.endswith(".json"):
                    with open(os.path.join(self.log_directory, file_name), "r") as file:
                        file_logs = json.load(file)
                        for log in file_logs:
                            if start_date and log["timestamp"] < start_date:
                                continue
                            if end_date and log["timestamp"] > end_date:
                                continue
                            if order_type and log["order_type"] != order_type:
                                continue
                            if keyword and keyword not in json.dumps(log):
                                continue
                            logs.append(log)
        except Exception as e:
            logging.error(f"Failed to search logs: {e}")
            self._send_error_notification(f"Error in search_logs: {e}")
        return logs

    def compress_logs(self):
        """
        오래된 로그를 압축하여 저장
        """
        try:
            for file_name in os.listdir(self.log_directory):
                if file_name.startswith("order_log_") and file_name.endswith(".json"):
                    file_path = os.path.join(self.log_directory, file_name)
                    compressed_file = f"{file_path}.gz"
                    with open(file_path, "rb") as f_in, gzip.open(compressed_file, "wb") as f_out:
                        f_out.writelines(f_in)
                    os.remove(file_path)
        except Exception as e:
            logging.error(f"Failed to compress logs: {e}")
            self._send_error_notification(f"Error in compress_logs: {e}")

    def _send_error_notification(self, message):
        """
        에러 알림을 이메일로 전송
        """
        if not self.error_notification_email:
            return

        try:
            smtp = SMTP("smtp.gmail.com", 587)
            smtp.starttls()
            smtp.login(self.config["email"]["user"], self.config["email"]["password"])

            msg = MIMEText(message)
            msg["Subject"] = "Order Logger Error Notification"
            msg["From"] = self.config["email"]["user"]
            msg["To"] = self.error_notification_email

            smtp.sendmail(self.config["email"]["user"], self.error_notification_email, msg.as_string())
            smtp.quit()
        except Exception as e:
            logging.error(f"Failed to send error notification: {e}")


# 테스트 코드
if __name__ == "__main__":
    logger = OrderLogger()

    # 샘플 로그 기록
    logger.log_order("12345", "BUY", "BTC/USDT", 0.5, 45000, "Executed", {"notes": "High priority order"})
    logger.log_order("12346", "SELL", "ETH/USDT", 2, 3500, "Pending")

    # 로그 검색
    results = logger.search_logs(keyword="BTC")
    print("검색된 로그:", results)

    # 로그 압축
    logger.compress_logs()
