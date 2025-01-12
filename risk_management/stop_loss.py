# risk_management/stop_loss.py

class StopLoss:
    def __init__(self, stop_loss_percentage=5):
        """
        손절매 클래스
        :param stop_loss_percentage: 손절매 비율 (기본값 5%)
        """
        self.stop_loss_percentage = stop_loss_percentage

    def check_stop_loss(self, entry_price, current_price):
        """
        손절매 발생 여부 확인
        :param entry_price: 진입 가격
        :param current_price: 현재 가격
        :return: 손절매 발생 여부 (True/False)
        """
        loss_threshold = entry_price * (1 - self.stop_loss_percentage / 100)
        return current_price <= loss_threshold

    def calculate_new_balance(self, entry_price, current_price, position_size):
        """
        손절매 발생 시 잔액 업데이트
        :param entry_price: 진입 가격
        :param current_price: 현재 가격
        :param position_size: 보유 포지션 크기
        :return: 손절매 발생 시 업데이트된 잔액
        """
        if self.check_stop_loss(entry_price, current_price):
            return current_price * position_size
        return entry_price * position_size

if __name__ == "__main__":
    # 예제 사용
    stop_loss = StopLoss(stop_loss_percentage=5)
    entry_price = 100000
    current_price = 95000
    position_size = 1

    if stop_loss.check_stop_loss(entry_price, current_price):
        new_balance = stop_loss.calculate_new_balance(entry_price, current_price, position_size)
        print(f"손절매 발생: 현재 가격 {current_price}원, 잔액 {new_balance}원")
    else:
        print(f"포지션 유지 중: 현재 가격 {current_price}원")
