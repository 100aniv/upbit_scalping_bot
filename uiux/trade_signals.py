# trade_signals.py
# AI 기반 매매 신호 패널 (RSI + MACD + LSTM 예측)
# 주석: 각 코드 라인의 설명을 자세하게 추가하였습니다.

# 필요한 라이브러리 임포트
import numpy as np
import matplotlib.pyplot as plt

# 매매 전략과 데이터 시뮬레이션을 위한 라이브러리
from datetime import datetime
import random

# AI 신호 패널 설정
ai_confidence = 87  # AI 신뢰도 (%)
risk_score = 35  # 리스크 점수 (%)
expected_return = 5  # 예상 수익률 (%)

# RSI, MACD, LSTM 신호를 생성하는 시뮬레이션 함수 정의
def calculate_rsi(prices, period=14):
    """RSI (상대강도지수) 계산 함수"""
    deltas = np.diff(prices)
    gains = deltas[deltas > 0].sum()
    losses = -deltas[deltas < 0].sum()
    rs = gains / losses if losses != 0 else 1
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices):
    """MACD (이동평균 수렴 확산 지수) 계산"""
    short_ema = np.mean(prices[-12:])  # 12일 단기 이동평균
    long_ema = np.mean(prices[-26:])   # 26일 장기 이동평균
    macd = short_ema - long_ema
    signal_line = np.mean(prices[-9:])  # 9일 시그널 라인
    return macd, signal_line

def lstm_prediction(prices):
    """LSTM 예측 (모델 시뮬레이션)"""
    # 실제 AI 모델 대신 임의의 수치로 대체 (87% 신뢰도 반영)
    return random.uniform(0.85, 0.95)

# 매매 신호를 계산하는 함수
def generate_trade_signals(prices):
    """RSI, MACD, LSTM 기반의 매매 신호 생성"""
    rsi = calculate_rsi(prices)
    macd, signal_line = calculate_macd(prices)
    lstm_confidence = lstm_prediction(prices)

    # 매매 신호 결정 로직
    buy_signal = rsi < 30 and macd > signal_line and lstm_confidence > 0.87
    sell_signal = rsi > 70 and macd < signal_line

    # 결과 출력
    print(f"\n[ AI 매매 신호 패널 ]")
    print(f"AI 신뢰도: {ai_confidence}%")
    print(f"리스크 점수: {risk_score}%")
    print(f"RSI: {rsi:.2f} | MACD: {macd:.2f} | Signal Line: {signal_line:.2f}")
    print(f"LSTM 신뢰도: {lstm_confidence*100:.1f}%")
    print(f"매수 신호: {'✅ 발생' if buy_signal else '❌ 없음'}")
    print(f"매도 신호: {'✅ 발생' if sell_signal else '❌ 없음'}")

    return buy_signal, sell_signal

# 예제 데이터 (비트코인 가격 데이터 시뮬레이션)
prices = [random.uniform(70000, 75000) for _ in range(100)]

# 매매 신호 생성 및 실행
buy_signal, sell_signal = generate_trade_signals(prices)

# 데이터 시각화 (캔들차트 및 RSI, MACD)
plt.figure(figsize=(12, 6))
plt.plot(prices, label='BTC Price', color='blue')
plt.title('BTC 가격 변화 및 AI 매매 신호')
plt.xlabel('시간')
plt.ylabel('가격 (KRW)')
plt.legend()
plt.show()

# 매매 실행 시뮬레이션 (자동매매 옵션)
def execute_trade(buy_signal, sell_signal):
    """자동 매매 실행 함수"""
    if buy_signal:
        print("✅ 매수 실행: 0.05 BTC 구매 완료")
    elif sell_signal:
        print("✅ 매도 실행: 0.05 BTC 판매 완료")
    else:
        print("📉 신호 없음: 대기 중")

# 자동매매 실행 여부 (True일 경우 매매 자동 실행)
auto_trade = True

if auto_trade:
    execute_trade(buy_signal, sell_signal)
else:
    print("🔔 자동매매 OFF: 수동으로 거래를 진행해주세요.")

