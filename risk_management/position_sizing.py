# risk_management/position_sizing.py

class PositionSizing:
    def __init__(self, balance, risk_percentage=1):
        """
        포지션 크기 관리 클래스
        :param balance: 총 계좌 잔액
        :param risk_percentage: 거래당 위험 비율 (기본값 1%)
        """
        if balance <= 0:
            raise ValueError("잔액은 0보다 커야 합니다.")
        if not (0 < risk_percentage <= 100):
            raise ValueError("위험 비율은 0과 100 사이여야 합니다.")
        
        self.balance = balance
        self.risk_percentage = risk_percentage

    def calculate_position_size(self, stop_loss_distance):
        """
        포지션 크기 계산
        :param stop_loss_distance: 손절매 기준 거리
        :return: 포지션 크기
        """
        if stop_loss_distance <= 0:
            raise ValueError("손절매 기준 거리는 0보다 커야 합니다.")
        
        risk_amount = self.balance * (self.risk_percentage / 100)
        position_size = risk_amount / stop_loss_distance
        return position_size

    def update_balance(self, profit_loss_amount):
        """
        잔액 업데이트
        :param profit_loss_amount: 수익 또는 손실 금액
        """
        self.balance += profit_loss_amount
        if self.balance <= 0:
            print("경고: 계좌 잔액이 0 이하입니다!")
        return self.balance

if __name__ == "__main__":
    # 예제 사용
    position_sizing = PositionSizing(balance=1000000, risk_percentage=1)
    stop_loss_distance = 5000
    try:
        position_size = position_sizing.calculate_position_size(stop_loss_distance)
        print(f"계산된 포지션 크기: {position_size:.2f}")
    except ValueError as e:
        print(f"오류 발생: {e}")

    # 잔액 업데이트 예제
    updated_balance = position_sizing.update_balance(-100000)
    print(f"잔액 업데이트 후: {updated_balance:.2f}")
