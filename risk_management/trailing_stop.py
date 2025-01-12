# risk_management/trailing_stop.py

class TrailingStop:
    def __init__(self, trailing_stop_percentage=5):
        """
        트레일링 스탑 클래스
        :param trailing_stop_percentage: 트레일링 스탑 비율 (기본값 5%)
        """
        self.trailing_stop_percentage = trailing_stop_percentage
        self.highest_price = None

    def update_highest_price(self, current_price):
        """
        현재 가격이 기존 최고가보다 높을 경우 최고가 갱신
        :param current_price: 현재 가격
        """
        if self.highest_price is None or current_price > self.highest_price:
            self.highest_price = current_price

    def check_trailing_stop(self, current_price):
        """
        트레일링 스탑 발생 여부 확인
        :param current_price: 현재 가격
        :return: 트레일링 스탑 발생 여부 (True/False)
        """
        if self.highest_price is None:
            return False
        stop_threshold = self.highest_price * (1 - self.trailing_stop_percentage / 100)
        return current_price <= stop_threshold

    def calculate_new_balance(self, current_price, position_size):
        """
        트레일링 스탑 발생 시 잔액 업데이트
        :param current_price: 현재 가격
        :param position_size: 보유 포지션 크기
        :return: 트레일링 스탑 발생 시 업데이트된 잔액
        """
        if self.check_trailing_stop(current_price):
            return current_price * position_size
        return self.highest_price * position_size

if __name__ == "__main__":
    # 예제 사용
    trailing_stop = TrailingStop(trailing_stop_percentage=5)
    prices = [100000, 102000, 104000, 103000, 101000, 98000]

    for price in prices:
        trailing_stop.update_highest_price(price)
        if trailing_stop.check_trailing_stop(price):
            new_balance = trailing_stop.calculate_new_balance(price, 1)
            print(f"트레일링 스탑 발생: 현재 가격 {price}원, 잔액 {new_balance}원")
        else:
            print(f"현재 가격 {price}원: 포지션 유지")
