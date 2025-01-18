import pandas as pd # type: ignore
import numpy as np # type: ignore
from numba import njit, prange, cuda # type: ignore
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import RobustScaler, StandardScaler # type: ignore
from websocket import create_connection # type: ignore
from scipy.signal import savgol_filter # type: ignore
from pywt import wavedec, waverec # type: ignore
import logging
import os
import time
from dotenv import load_dotenv # type: ignore
from textblob import TextBlob # type: ignore
from data_collector import fetch_upbit_ticker, fetch_coinness_news

# 환경 변수 로드 및 로그 설정
load_dotenv()
logging.basicConfig(filename="indicators.log", level=logging.INFO)


################################### 기술적 지표 클래스 ###################################
class TechnicalIndicators:
    """
    고급 기술적 지표 모듈 (27종 이상)
    - 기존: RSI, MACD, VWAP, Bollinger Bands, ATR, EMA, Fibonacci Retracement, Ichimoku Cloud, Donchian Channel
    - 추가: Keltner Channel, Supertrend, Parabolic SAR, CCI, ROC, Williams %R, Stochastic Oscillator, ADX, MFI, Elder-Ray Index
    - 스케일링 및 데이터 전처리를 위한 RobustScaler, Z-Score Scaling 추가
    - NumPy 기반 벡터 연산 최적화 및 Numba JIT 적용
    - AI 학습에 최적화
    """

    @staticmethod
    @njit
    def calculate_rsi(data, period=14):
        """
        RSI (Relative Strength Index) 계산 함수
        RSI는 최근의 가격 변동 강도를 측정하는 지표입니다.
        - 70 이상: 과매수 상태 (매도 신호)
        - 30 이하: 과매도 상태 (매수 신호)

        :param data: 종가 데이터를 포함한 NumPy 배열
        :param period: RSI 계산에 사용할 기간 (기본값 14)
        :return: RSI 값을 추가한 NumPy 배열
        """
        delta = np.diff(data)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(data)
        avg_loss = np.zeros_like(data)

        for i in range(period, len(data)):
            if i == period:
                avg_gain[i] = np.mean(gain[:period])
                avg_loss[i] = np.mean(loss[:period])
            else:
                avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i - 1]) / period
                avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i - 1]) / period

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        result = np.empty_like(data)
        result[:period] = np.nan
        result[period:] = rsi[period - 1:]
        return result
    
    @staticmethod
    @njit
    def calculate_macd(data, short_period=12, long_period=26, signal_period=9):
        """
        MACD (Moving Average Convergence Divergence) 계산 함수
        MACD는 두 개의 이동 평균선의 차이를 통해 추세 강도를 측정합니다.
        - MACD 라인이 신호선 위를 교차할 때: 매수 신호
        - MACD 라인이 신호선 아래로 교차할 때: 매도 신호

        :param data: 종가 데이터를 포함한 NumPy 배열
        :param short_period: 단기 EMA 기간 (기본값 12)
        :param long_period: 장기 EMA 기간 (기본값 26)
        :param signal_period: 신호선 EMA 기간 (기본값 9)
        :return: MACD, 신호선이 추가된 NumPy 배열
        """
        short_ema = np.zeros_like(data)
        long_ema = np.zeros_like(data)
        macd = np.zeros_like(data)
        signal_line = np.zeros_like(data)

        alpha_short = 2 / (short_period + 1)
        alpha_long = 2 / (long_period + 1)
        alpha_signal = 2 / (signal_period + 1)

        for i in range(1, len(data)):
            short_ema[i] = alpha_short * data[i] + (1 - alpha_short) * short_ema[i - 1]
            long_ema[i] = alpha_long * data[i] + (1 - alpha_long) * long_ema[i - 1]
            macd[i] = short_ema[i] - long_ema[i]
            signal_line[i] = alpha_signal * macd[i] + (1 - alpha_signal) * signal_line[i - 1]

        return macd, signal_line

    @staticmethod
    @njit
    def ema_numpy(data, period):
        """
        NumPy 기반 EMA 계산 함수
        :param data: 입력 데이터 배열
        :param period: EMA 기간
        :return: EMA 배열
        """
        alpha = 2 / (period + 1)
        ema = np.empty_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema

    def calculate_vwap(self, data):
        """
        VWAP (Volume Weighted Average Price) 계산 함수
        - 거래량 가중 평균 가격
        - 사용 목적: 기관 투자자들이 많이 사용하는 지표
        """
        cumulative_volume = np.cumsum(data['volume'].values)
        cumulative_price_volume = np.cumsum(data['close'].values * data['volume'].values)
        data['vwap'] = cumulative_price_volume / (cumulative_volume + 1e-10)
        return data

    @staticmethod
    @njit
    def calculate_bollinger_bands(data, period=20):
        """
        Bollinger Bands (볼린저 밴드) 계산 함수
        볼린저 밴드는 가격 변동성을 나타내며, 상단/하단 밴드를 기준으로 과매수 및 과매도 상태를 판단합니다.
        - 상단 밴드: SMA + (2 * 표준편차)
        - 하단 밴드: SMA - (2 * 표준편차)

        :param data: 종가 데이터를 포함한 NumPy 배열
        :param period: 볼린저 밴드 계산에 사용할 기간 (기본값 20)
        :return: (중간 밴드, 상단 밴드, 하단 밴드)를 포함한 NumPy 배열
        """
        rolling_mean = np.convolve(data, np.ones(period) / period, mode='valid')
        rolling_std = np.array([np.std(data[i - period:i]) for i in range(period, len(data) + 1)])
        middle_band = np.concatenate((np.full(period - 1, np.nan), rolling_mean))
        upper_band = middle_band + 2 * rolling_std
        lower_band = middle_band - 2 * rolling_std
        return middle_band, upper_band, lower_band
    
    def calculate_atr(self, data, period=14):
        """
        ATR (Average True Range) 계산 함수
        - 변동성을 측정하는 지표
        - 수익 목표 및 손절매 설정에 활용
        """
        high_low = data['high'].values - data['low'].values
        high_close = np.abs(data['high'].values - data['close'].shift().values)
        low_close = np.abs(data['low'].values - data['close'].shift().values)
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = np.convolve(tr, np.ones(period) / period, mode='valid')
        data['atr'] = np.concatenate((np.full(period - 1, np.nan), atr))
        return data


    def calculate_ema(self, data, period=20):
        """
        EMA (Exponential Moving Average) 계산 함수
        EMA는 최근 데이터에 더 높은 가중치를 두는 이동 평균입니다.
        - EMA는 시장의 최근 동향을 빠르게 반영하는 지표입니다.
        - 주가가 EMA를 상향 돌파할 경우 매수 신호
        - 주가가 EMA를 하향 돌파할 경우 매도 신호
        """
        data['ema'] = data['close'].ewm(span=period, adjust=False).mean()
        return data

    def calculate_fibonacci_retracement(self, data):
        """
        Fibonacci Retracement (피보나치 되돌림) 계산 함수(벡터화 적용)
        피보나치 되돌림은 지지선과 저항선을 예측하는 데 사용됩니다.
        - 23.6%, 38.2%, 50%, 61.8% 수준을 계산
        - 주가가 피보나치 수준에서 반등하면 매수 신호, 돌파하면 추가 하락 가능성
        """
        high = data['high'].max()
        low = data['low'].min()
        diff = high - low
        levels = [0.236, 0.382, 0.5, 0.618]
        for level in levels:
            data[f'fibonacci_{int(level*100)}'] = high - diff * level
        return data

    def calculate_ichimoku_cloud(self, data):
        """
        Ichimoku Cloud (일목균형표) 계산 함수
        - Tenkan-sen (전환선): 최근 9일간의 고가와 저가의 평균
        - Kijun-sen (기준선): 최근 26일간의 고가와 저가의 평균
        - Senkou Span A: 전환선과 기준선의 평균을 26일 미래로 이동
        - Senkou Span B: 최근 52일간의 고가와 저가의 평균을 26일 미래로 이동
        - Chikou Span (후행 스팬): 현재 종가를 26일 뒤로 이동
        """
        high_9 = data['high'].rolling(window=9).max()
        low_9 = data['low'].rolling(window=9).min()
        data['tenkan_sen'] = (high_9 + low_9) / 2

        high_26 = data['high'].rolling(window=26).max()
        low_26 = data['low'].rolling(window=26).min()
        data['kijun_sen'] = (high_26 + low_26) / 2

        data['senkou_span_a'] = ((data['tenkan_sen'] + data['kijun_sen']) / 2).shift(26)
        high_52 = data['high'].rolling(window=52).max()
        low_52 = data['low'].rolling(window=52).min()
        data['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)

        data['chikou_span'] = data['close'].shift(-26)
        return data

    def calculate_donchian_channel(self, data, period=20):
        """
        Donchian Channel (돈치안 채널) 계산 함수
        - Donchian High: 최근 기간 동안의 최고가
        - Donchian Low: 최근 기간 동안의 최저가
        - 사용 목적: 돌파 전략 및 추세 확인
        """
        data['donchian_high'] = data['high'].rolling(window=period).max()
        data['donchian_low'] = data['low'].rolling(window=period).min()
        return data

# 추가 구현될 지표
#1.	트렌드 기반 지표 (3종 추가):
    def calculate_keltner_channel(self, data, period=20, multiplier=1.5):
        """
        Keltner Channel 계산 함수
        - 가격의 평균과 ATR을 결합하여 추세 및 변동성을 분석

        :param data: 고가, 저가, 종가를 포함한 데이터프레임
        :param period: EMA 계산에 사용할 기간 (기본값 20)
        :param atr_period: ATR 계산에 사용할 기간 (기본값 14)
        :return: Keltner Channel 상단 및 하단 밴드를 추가한 데이터프레임
        """
        atr = self.calculate_atr(data, period)
        middle_band = data['close'].rolling(window=period).mean()
        data['kc_upper'] = middle_band + multiplier * atr
        data['kc_lower'] = middle_band - multiplier * atr
        return data

    def calculate_supertrend(self, data, period=10, multiplier=3):
        """
        Supertrend 계산 함수
        - ATR을 이용하여 현재 추세를 분석하는 강력한 지표

        :param data: 고가, 저가, 종가를 포함한 데이터프레임
        :param atr_period: ATR 계산에 사용할 기간 (기본값 10)
        :param multiplier: ATR 곱연산 계수 (기본값 3)
        :return: Supertrend 상단 및 하단 값을 추가한 데이터프레임
        """
        atr = self.calculate_atr(data, period)
        hl2 = (data['high'] + data['low']) / 2
        upperband = hl2 + (multiplier * atr)
        lowerband = hl2 - (multiplier * atr)
        data['supertrend'] = upperband.where(data['close'] < upperband, lowerband)
        data['supertrend_direction'] = np.where(data['close'] > data['supertrend'], 1, -1)
        return data

    def calculate_parabolic_sar(self, data, start_af=0.02, max_af=0.2):
        """
        Parabolic SAR 계산 함수
        - 시간에 따른 추세의 방향과 포지션 전환 신호를 제공

        :param data: 고가, 저가, 종가를 포함한 데이터프레임
        :param step: 가속 계수 (기본값 0.02)
        :param max_step: 가속 계수의 최대값 (기본값 0.2)
        :return: Parabolic SAR 값을 추가한 데이터프레임
        """
        sar = np.zeros_like(data['close'])
        af = start_af
        ep = data['high'][0]
        sar[0] = data['low'][0]

        for i in range(1, len(data)):
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            if data['close'][i] > sar[i]:
                af = min(af + start_af, max_af)
                ep = max(ep, data['high'][i])
            elif data['close'][i] < sar[i]:
                af = min(af + start_af, max_af)
                ep = min(ep, data['low'][i])

        data['psar'] = sar
        return data

    #2.	모멘텀 기반 지표 (5종 추가):
    def calculate_cci(data, period=20):
        """
        CCI (Commodity Channel Index) 계산 함수
        - 가격의 이동평균과의 편차를 측정하여 과매수/과매도를 판단

        :param data: 고가, 저가, 종가를 포함한 데이터프레임
        :param period: CCI 계산에 사용할 기간 (기본값 20)
        :return: CCI 값을 추가한 데이터프레임
        """
        tp = (data['high'] + data['low'] + data['close']) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.fabs(x - x.mean()).mean())
        data['cci'] = (tp - sma) / (0.015 * mad)
        return data

    def calculate_roc(data, period=14):
        """
        ROC (Rate of Change) 계산 함수
        - 가격의 상대적 변화율을 측정하여 추세 반전 또는 지속성을 평가

        :param data: 종가를 포함한 데이터프레임
        :param period: ROC 계산에 사용할 기간 (기본값 14)
        :return: ROC 값을 추가한 데이터프레임
        """
        data['roc'] = (data['close'] / data['close'].shift(period) - 1) * 100
        return data

    def calculate_williams_r(data, period=14):
        """
        Williams %R 계산 함수
        - 최고가 대비 현재 가격의 위치를 계산하여 과매수/과매도를 판단

        :param data: 고가, 저가, 종가를 포함한 데이터프레임
        :param period: Williams %R 계산에 사용할 기간 (기본값 14)
        :return: Williams %R 값을 추가한 데이터프레임
        """
        highest_high = data['high'].rolling(window=period).max()
        lowest_low = data['low'].rolling(window=period).min()
        data['williams_r'] = ((highest_high - data['close']) / (highest_high - lowest_low)) * -100
        return data

    def calculate_stochastic_oscillator(data, period=14, smooth_k=3, smooth_d=3):
        """
        Stochastic Oscillator 계산 함수
        - 현재 가격과 최고/최저가의 관계를 분석하여 시장의 강도를 파악

        :param data: 고가, 저가, 종가를 포함한 데이터프레임
        :param period: Stochastic Oscillator의 기본 계산 기간 (기본값 14)
        :param smooth_k: %K 선을 부드럽게 하는 기간 (기본값 3)
        :param smooth_d: %D 선을 부드럽게 하는 기간 (기본값 3)
        :return: %K와 %D 값을 추가한 데이터프레임
        """
        lowest_low = data['low'].rolling(window=period).min()
        highest_high = data['high'].rolling(window=period).max()
        data['%K'] = 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))
        data['%D'] = data['%K'].rolling(window=smooth_d).mean()
        return data

    def calculate_adx(data, period=14):
        """
        ADX (Average Directional Index) 계산 함수
        - 추세의 강도를 측정하며 상승/하락 방향과는 무관

        :param data: 고가, 저가, 종가를 포함한 데이터프레임
        :param period: ADX 계산에 사용할 기간 (기본값 14)
        :return: ADX 값을 추가한 데이터프레임
        """
        plus_dm = np.where(data['high'].diff() > data['low'].diff(), data['high'].diff(), 0)
        minus_dm = np.where(data['low'].diff() > data['high'].diff(), data['low'].diff(), 0)
        tr = calculate_atr(data, period)['atr'] # type: ignore
        plus_di = (plus_dm.rolling(window=period).sum() / tr) * 100
        minus_di = (minus_dm.rolling(window=period).sum() / tr) * 100
        dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        data['adx'] = dx.rolling(window=period).mean()
        return data

    #3.	거래량 기반 지표 (3종 추가):
    def calculate_mfi(data, period=14):
        """
        MFI (Money Flow Index) 계산 함수
        - 가격과 거래량을 결합하여 과매수/과매도 상태를 분석

        :param data: 고가, 저가, 종가, 거래량을 포함한 데이터프레임
        :param period: MFI 계산에 사용할 기간 (기본값 14)
        :return: MFI 값을 추가한 데이터프레임
        """
        tp = (data['high'] + data['low'] + data['close']) / 3
        mf = tp * data['volume']
        positive_mf = np.where(tp > tp.shift(1), mf, 0)
        negative_mf = np.where(tp < tp.shift(1), mf, 0)
        mfr = positive_mf.sum() / (negative_mf.sum() + 1e-10)
        data['mfi'] = 100 - (100 / (1 + mfr))
        return data

    def calculate_elder_ray(data, period=14):
        """
        Elder-Ray Index 계산 함수
        - 매수와 매도 압력을 측정하여 시장의 힘을 평가

        :param data: 고가, 저가, 종가를 포함한 데이터프레임
        :param period: EMA 계산에 사용할 기간 (기본값 14)
        :return: Bull Power와 Bear Power 값을 추가한 데이터프레임
        """
        ema = data['close'].ewm(span=period, adjust=False).mean()
        data['bull_power'] = data['high'] - ema
        data['bear_power'] = data['low'] - ema
        return data

    def calculate_obv(data):
        """
        OBV (On-Balance Volume) 계산 함수
        - 거래량의 흐름을 분석하여 시장의 추세를 파악

        :param data: 종가와 거래량을 포함한 데이터프레임
        :return: OBV 값을 추가한 데이터프레임
        """
        data['obv'] = np.where(data['close'] > data['close'].shift(1), data['volume'], -data['volume']).cumsum()
        return data

    #4.	시장 복잡성 지표 (2종 추가):
    def calculate_hurst_exponent(data, lag_range=20):
        """
        Hurst Exponent 계산 함수
        - 가격 움직임의 랜덤성과 추세를 분석하여 시장의 복잡성을 평가

        :param data: 종가를 포함한 데이터프레임
        :param lag_range: 계산에 사용할 최대 시차 (기본값 20)
        :return: Hurst 값을 추가한 데이터프레임
        """
        lags = range(2, lag_range)
        tau = [np.std(np.subtract(data['close'].values[lag:], data['close'].values[:-lag])) for lag in lags]
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        hurst = reg[0] * 2
        data['hurst_exponent'] = hurst
        return data

    def calculate_fractal_dimension(data, period=14):
        """
        Fractal Dimension 계산 함수
        - 시장의 패턴 복잡성을 측정하여 변동성과 추세를 분석

        :param data: 종가를 포함한 데이터프레임
        :param period: 계산에 사용할 기간 (기본값 14)
        :return: Fractal Dimension 값을 추가한 데이터프레임
        """
        high_low_diff = data['high'] - data['low']
        fd = high_low_diff.rolling(window=period).apply(lambda x: np.log(np.sum(x) / len(x)) / np.log(len(x)))
        data['fractal_dimension'] = fd
        return data

    #5.	조정된 지표 (5종 추가):
    def calculate_adjusted_rsi(data, sentiment_scores, period=14):
        """
        Adjusted RSI 계산 함수
        - 뉴스 감정 점수를 반영하여 RSI 값을 조정

        :param data: 종가 데이터를 포함한 데이터프레임
        :param sentiment_scores: 감정 점수 배열 또는 시리즈
        :param period: RSI 계산에 사용할 기간 (기본값 14)
        :return: Adjusted RSI 값을 추가한 데이터프레임
        """
        delta = data['close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.convolve(gain, np.ones(period) / period, mode='valid')
        avg_loss = np.convolve(loss, np.ones(period) / period, mode='valid')
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        adjusted_rsi = rsi + (sentiment_scores[:len(rsi)] * 10)
        data['adjusted_rsi'] = np.concatenate((np.full(period - 1, np.nan), adjusted_rsi))
        return data

    def calculate_sentiment_adjusted_macd(data, sentiment_scores, short_period=12, long_period=26, signal_period=9):
        """
        Sentiment-Adjusted MACD 계산 함수
        - 감정 점수를 반영하여 MACD 신호의 신뢰도를 강화

        :param data: 종가 데이터를 포함한 데이터프레임
        :param sentiment_scores: 감정 점수 배열 또는 시리즈
        :param short_period: 단기 EMA의 기간 (기본값 12)
        :param long_period: 장기 EMA의 기간 (기본값 26)
        :param signal_period: 신호선 EMA의 기간 (기본값 9)
        :return: Sentiment-Adjusted MACD 값을 추가한 데이터프레임
        """
        short_ema = data['close'].ewm(span=short_period, adjust=False).mean()
        long_ema = data['close'].ewm(span=long_period, adjust=False).mean()
        macd = short_ema - long_ema
        signal_line = macd.ewm(span=signal_period, adjust=False).mean()
        sentiment_adjusted_macd = macd + (sentiment_scores[:len(macd)] * signal_line)
        data['sentiment_adjusted_macd'] = sentiment_adjusted_macd
        return data

    def calculate_multi_timeframe_ema(data, short_period=12, long_period=26):
        """
        Multi-Timeframe EMA 계산 함수
        - 다양한 시간 프레임의 EMA를 결합하여 추세의 신뢰도를 강화

        :param data: 종가 데이터를 포함한 데이터프레임
        :param short_period: 단기 EMA의 기간 (기본값 12)
        :param long_period: 장기 EMA의 기간 (기본값 26)
        :return: Multi-Timeframe EMA 값을 추가한 데이터프레임
        """
        short_ema = data['close'].ewm(span=short_period, adjust=False).mean()
        long_ema = data['close'].ewm(span=long_period, adjust=False).mean()
        data['multi_timeframe_ema'] = (short_ema + long_ema) / 2
        return data

    def calculate_adjusted_vwap(data, sentiment_scores):
        """
        Adjusted VWAP 계산 함수
        - 뉴스 데이터와 거래량 데이터를 반영한 가중 평균 가격

        :param data: 고가, 저가, 종가, 거래량 데이터를 포함한 데이터프레임
        :param sentiment_scores: 감정 점수 배열 또는 시리즈
        :return: Adjusted VWAP 값을 추가한 데이터프레임
        """
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        cumulative_tp_volume = (typical_price * data['volume']).cumsum()
        cumulative_volume = data['volume'].cumsum()
        vwap = cumulative_tp_volume / cumulative_volume
        adjusted_vwap = vwap + (sentiment_scores[:len(vwap)] * 0.1)
        data['adjusted_vwap'] = adjusted_vwap
        return data

    def calculate_sentiment_weighted_bollinger_bands(data, sentiment_scores, period=20):
        """
        Sentiment-Weighted Bollinger Bands 계산 함수
        - 뉴스 감정 데이터를 변동성 계산에 통합하여 정확도를 높임

        :param data: 종가 데이터를 포함한 데이터프레임
        :param sentiment_scores: 감정 점수 배열 또는 시리즈
        :param period: 볼린저 밴드 계산에 사용할 기간 (기본값 20)
        :return: 상단 및 하단 밴드 값을 추가한 데이터프레임
        """
        sma = data['close'].rolling(window=period).mean()
        std_dev = data['close'].rolling(window=period).std()
        upper_band = sma + (std_dev * 2) + (sentiment_scores[:len(sma)] * 0.1)
        lower_band = sma - (std_dev * 2) - (sentiment_scores[:len(sma)] * 0.1)
        data['sentiment_bollinger_upper'] = upper_band
        data['sentiment_bollinger_lower'] = lower_band
        return data

########################### 최적화 함수 (별도의 유틸리티 함수) ##############################
def optimize_dataframe(df):
    """
    데이터프레임의 메모리 사용량을 줄이기 위해 데이터 타입 최적화
    :param df: 최적화할 pandas 데이터프레임
    :return: 최적화된 데이터프레임
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2  # 메모리 사용량(MB)
    print(f"최적화 전 메모리 사용량: {start_mem:.2f} MB")
    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:  # 문자열이 아닌 경우에만 처리
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type).startswith('int'):  # 정수형 최적화
                if c_min >= -128 and c_max <= 127:
                    df[col] = df[col].astype('int8')
                elif c_min >= -32768 and c_max <= 32767:
                    df[col] = df[col].astype('int16')
                elif c_min >= -2147483648 and c_max <= 2147483647:
                    df[col] = df[col].astype('int32')
                else:
                    df[col] = df[col].astype('int64')
            elif str(col_type).startswith('float'):  # 실수형 최적화
                if c_min >= -3.4e38 and c_max <= 3.4e38:
                    df[col] = df[col].astype('float32')
                else:
                    df[col] = df[col].astype('float64')
    
    end_mem = df.memory_usage().sum() / 1024 ** 2  # 최적화 후 메모리 사용량(MB)
    print(f"최적화 후 메모리 사용량: {end_mem:.2f} MB")
    print(f"메모리 사용량 감소율: {100 * (start_mem - end_mem) / start_mem:.1f}%")
    
    return df

################################### 데이터 처리 클래스스 ###################################

class DataProcessor:
    """
    데이터 처리 및 개선 사항 통합 클래스
    - 실시간 데이터 처리 및 기술적 지표 계산
    - 다중 시간 프레임 생성
    - 뉴스 감정 점수 분석
    - 멀티스레딩 및 결과 통합
    - 보안 및 에러 처리 포함
    """

    def __init__(self):
        self.ws_url = os.getenv("WEBSOCKET_URL")  # WebSocket URL 환경 변수
        self.api_key = os.getenv("API_KEY")  # API 키 환경 변수


    # 기존 기능: 실시간 데이터 수집
def collect_real_time_websocket(self, max_retries=5, backoff_factor=2):
    """
    WebSocket 연결을 이용한 실시간 데이터 수집 (백오프 전략 포함)
    :param max_retries: 재연결 시도 횟수
    :param backoff_factor: 재시도 간 대기 시간 증가 계수
    """
    retry_count = 0
    while retry_count < max_retries:
        try:
            ws = create_connection(self.ws_url)
            logging.info("WebSocket 연결 성공")
            print("WebSocket 연결 성공. 실시간 데이터 수집 시작...")
            while True:
                data = ws.recv()
                print(f"실시간 데이터: {data}")  # 실제 데이터 처리 로직 추가 가능
        except Exception as e:
            retry_count += 1
            wait_time = backoff_factor ** retry_count  # 재시도 간 대기 시간 증가
            logging.error(f"WebSocket 연결 실패: {e} (재시도 {retry_count}/{max_retries})")
            if retry_count >= max_retries:
                logging.error("WebSocket 재연결 실패. 최대 재시도 횟수 초과.")
                print("WebSocket 재연결 실패. 프로그램을 종료합니다.")
                return
            print(f"{wait_time}초 후 재시도합니다...")
            time.sleep(wait_time)


    # 기존 기능: 데이터 스무딩
    @staticmethod
    def smooth_data(data, method="sma", window=3):
        """
        데이터 스무딩 (SMA 또는 Gaussian)
        :param data: 입력 데이터 시리즈
        :param method: 스무딩 방식 ('sma' 또는 'gaussian')
        :param window: 윈도우 크기
        :return: 스무딩된 데이터 시리즈
        """
        if method == "sma":
            return data.rolling(window=window).mean()
        elif method == "gaussian":
            return data.rolling(window=window, win_type='gaussian').mean(std=window/2)
        else:
            raise ValueError("지원하지 않는 스무딩 방식입니다.")

    # 기존 기능: 다중 시간 프레임 생성
    @staticmethod
    def apply_multiple_timeframes(data, intervals=["1T", "5T", "1H"]):
        """
        다중 시간 프레임 생성
        :param data: 원본 데이터프레임
        :param intervals: 사용할 시간 간격 목록
        :return: 각 시간 프레임 데이터를 포함한 딕셔너리
        """
        resampled_data = {}
        for interval in intervals:
            resampled_data[interval] = data.resample(interval).mean()
        return resampled_data

   # 기존 기능: 뉴스 감정 분석
    @staticmethod
    def analyze_sentiment(news):
        """
        뉴스 텍스트를 분석하여 감정 점수를 반환
        :param news: 뉴스 텍스트 문자열
        :return: 감정 점수 (-1.0 ~ 1.0)
        """
        sentiment = TextBlob(news).sentiment.polarity
        logging.info(f"뉴스 감정 분석: {sentiment}")
        return sentiment

    # 기존 기능: 데이터 스케일링
    @staticmethod
    def scale_data(data, method="robust"):
        """
        데이터 스케일링 (RobustScaler 또는 Z-Score)
        :param data: 입력 데이터프레임
        :param method: 스케일링 방식 ('robust' 또는 'zscore')
        :return: 스케일링된 데이터
        """
        if method == "robust":
            scaler = RobustScaler()
        elif method == "zscore":
            scaler = StandardScaler()
        else:
            raise ValueError("지원하지 않는 스케일링 방식입니다.")
        return scaler.fit_transform(data)

    # 기존 기능: 멀티스레딩 지표 계산
    def calculate_indicators_multithreaded(self, data, indicators):
        """
        멀티스레딩을 이용한 지표 동시 계산
        :param data: 원본 데이터프레임
        :param indicators: 실행할 지표 계산 함수 리스트
        :return: 계산 결과가 포함된 데이터프레임
        """
        try:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(indicator, data) for indicator in indicators]
                for future in futures:
                    result = future.result()  # 각 작업 결과를 기다림
            logging.info("모든 지표가 성공적으로 계산되었습니다.")
        except Exception as e:
            logging.error(f"지표 계산 에러: {e}")
        return data

    # 기존 기능: 결과 통합
    def integrate_results(self, data, multi_timeframe_data, sentiment_score):
        """
        결과 통합 및 반환
        :param data: 기본 데이터프레임
        :param multi_timeframe_data: 다중 시간 프레임 데이터
        :param sentiment_score: 뉴스 감정 점수
        :return: 통합 데이터프레임
        """
        try:
            data['sentiment_score'] = sentiment_score
            for timeframe, df in multi_timeframe_data.items():
                data[f'{timeframe}_close'] = df['close']
            logging.info("결과가 성공적으로 통합되었습니다.")
        except Exception as e:
            logging.error(f"결과 통합 에러: {e}")
        return data

   # 추가 기능: GPU 가속 볼린저 밴드 계산
    def calculate_bollinger_bands_gpu(self, data, period=20):
        """
        GPU 기반 볼린저 밴드 계산.
        GPU가 없는 경우 CPU 기반 계산으로 대체.
        """
        if not cuda.is_available():
            logging.warning("GPU가 감지되지 않았습니다. CPU 기반 계산으로 전환합니다.")
            return self.calculate_bollinger_bands(data, period)  # 기존 CPU 기반 함수 호출

        # GPU 기반 계산 로직
        try:
            @cuda.jit
            def compute_bands(prices, sma_out, upper_out, lower_out, period):
                pos = cuda.grid(1)
                if pos >= period and pos < len(prices):
                    sma = sum(prices[pos - period:pos]) / period
                    std_dev = (sum((prices[pos - period:pos] - sma) ** 2) / period) ** 0.5
                    sma_out[pos] = sma
                    upper_out[pos] = sma + 2 * std_dev
                    lower_out[pos] = sma - 2 * std_dev

            n = len(data['close'])
            sma_out = cuda.device_array(n, dtype=np.float32)
            upper_out = cuda.device_array(n, dtype=np.float32)
            lower_out = cuda.device_array(n, dtype=np.float32)
            threads_per_block = 128
            blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

            compute_bands[blocks_per_grid, threads_per_block](data['close'].values, sma_out, upper_out, lower_out, period)
            data['bb_middle'] = sma_out.copy_to_host()
            data['bollinger_upper'] = upper_out.copy_to_host()
            data['bollinger_lower'] = lower_out.copy_to_host()

            logging.info("GPU 기반 볼린저 밴드 계산 완료")
            return data
        except Exception as e:
            logging.error(f"GPU 계산 실패: {e}. CPU 기반 계산으로 전환합니다.")
            return self.calculate_bollinger_bands(data, period)  # 실패 시 CPU 기반 계산으로 대체

    # 추가 기능: Wavelet Transform 기반 스무딩
    def smooth_wavelet(self, data, column):
        """
        Wavelet Transform 기반 노이즈 제거
        """
        coeffs = wavedec(data[column].values, 'db1', level=2)
        coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]  # 노이즈 제거
        smoothed = waverec(coeffs, 'db1')
        data[f'{column}_smoothed'] = smoothed[:len(data)]
        return data

    # 추가 기능: 적응형 ATR 계산
    def adaptive_atr(self, data, period=14, multiplier=1.5):
        """
        적응형 ATR 계산
        """
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift(1))
        low_close = np.abs(data['low'] - data['close'].shift(1))
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        adaptive_period = np.clip((atr / atr.mean() * period).astype(int), 5, 2 * period)

        adaptive_atr_values = []
        for i in range(len(true_range)):
            if i < adaptive_period[i]:
                adaptive_atr_values.append(np.nan)
            else:
                adaptive_atr_values.append(true_range[i - adaptive_period[i]:i].mean())

        data['adaptive_atr'] = adaptive_atr_values
        return data

    def __init__(self):
        self.ws_url = os.getenv("WEBSOCKET_URL")  # WebSocket URL 환경 변수
        self.api_key = os.getenv("API_KEY")  # API 키 환경 변수
    
    # 추가 기능: 업비트에서 모든 마켓의 실시간 티커 데이터를 가져오는 함수
    def fetch_upbit_ticker_all(self, markets, retries=3):
        """
        업비트에서 모든 마켓의 실시간 티커 데이터를 가져오는 함수
        :param markets: 마켓 리스트
        :param retries: API 호출 실패 시 재시도 횟수
        :return: 모든 마켓의 실시간 데이터 리스트
        """
        url = f"https://api.upbit.com/v1/ticker?markets={','.join(markets)}"
        for attempt in range(retries):
            try:
                response = requests.get(url)
                response.raise_for_status()
                ticker_data = response.json()
                logging.info(f"업비트 티커 데이터 수신: {len(ticker_data)}개 마켓 (시도 {attempt + 1}/{retries})")
                return ticker_data
            except Exception as e:
                logging.error(f"Upbit API 호출 실패 (시도 {attempt + 1}/{retries}): {e}")
                if attempt + 1 == retries:
                    logging.error("최대 재시도 횟수 초과. 기본값 반환.")
                    return []
        return []

    # 추가 기능 : 업비트에서 지원하는 모든 마켓 리스트를 가져오는 함수
    def fetch_upbit_all_markets(self, retries=3):
        """
        업비트에서 지원하는 모든 마켓 리스트를 가져오는 함수
        :param retries: API 호출 실패 시 재시도 횟수
        """
        url = "https://api.upbit.com/v1/market/all"
        for attempt in range(retries):
            try:
                response = requests.get(url)
                response.raise_for_status()
                markets = response.json()
                logging.info(f"업비트 마켓 리스트 수신: {len(markets)}개 마켓 (시도 {attempt + 1}/{retries})")
                return [market['market'] for market in markets if market['market'].startswith('KRW')]
            except Exception as e:
                logging.error(f"업비트 마켓 리스트 호출 실패 (시도 {attempt + 1}/{retries}): {e}")
                if attempt + 1 == retries:
                    logging.error("최대 재시도 횟수 초과. 기본값 반환.")
                    return []
        return []

    # 추가 기능 : 업비트에서 모든 마켓의 실시간 티커 데이터를 배치 단위로 가져오는 함수
    def fetch_upbit_ticker_all_batched(self, markets, batch_size=50, retries=3, delay=1):
        """
        업비트에서 모든 마켓의 실시간 티커 데이터를 배치 단위로 가져오는 함수
        :param markets: 마켓 리스트
        :param batch_size: 한 번에 요청할 마켓 수
        :param retries: API 호출 실패 시 재시도 횟수
        :param delay: 각 배치 호출 간의 대기 시간 (초)
        :return: 모든 마켓의 실시간 데이터 리스트
        """
        batched_data = []
        for i in range(0, len(markets), batch_size):
            batch = markets[i:i + batch_size]
            for attempt in range(retries):
                try:
                    batch_data = self.fetch_upbit_ticker_all(batch)
                    batched_data.extend(batch_data)
                    logging.info(f"배치 {i // batch_size + 1}: {len(batch)}개 마켓 데이터 수신")
                    break
                except Exception as e:
                    logging.error(f"배치 {i // batch_size + 1} 호출 실패 (시도 {attempt + 1}/{retries}): {e}")
                    if attempt + 1 == retries:
                        logging.error("최대 재시도 횟수 초과. 데이터 수집 중단.")
            time.sleep(delay)  # 배치 호출 간 대기
        return batched_data


    # 추가 기능 : 업비트 실시간 데이터를 DataFrame 형태로 변환
    def process_upbit_data_to_dataframe(self):
        """
        업비트 실시간 데이터를 DataFrame 형태로 변환
        """
        markets = self.fetch_upbit_all_markets()
        if not markets:
            logging.error("마켓 리스트를 가져오지 못했습니다.")
            return pd.DataFrame()  # 빈 데이터프레임 반환

        ticker_data = self.fetch_upbit_ticker_all_batched(markets, batch_size=50)
        if not ticker_data:
            logging.error("티커 데이터를 가져오지 못했습니다.")
            return pd.DataFrame()  # 빈 데이터프레임 반환

        # 데이터를 데이터프레임으로 변환
        ticker_df = pd.DataFrame(ticker_data)
        logging.info(f"티커 데이터를 데이터프레임으로 변환 완료: {ticker_df.shape[0]}개의 행")
        return ticker_df

        
    # 추가 기능 : 코인니스, upbit 실시간 데이터 처리 함수
    def process_coinness_news(self, retries=3):
        """
        코인니스 뉴스 데이터를 가져와서 처리하는 함수
        API 호출 실패 시 기본값 처리 및 재시도 로직 포함.
        """
        headers = {"Authorization": f"Bearer {os.getenv('COINNESS_API_KEY')}"}
        url = "https://api.coinness.com/news/latest"
        for attempt in range(retries):
            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                logging.info("코인니스 API 호출 성공")
                return response.json()
            except Exception as e:
                logging.error(f"Coinness API 호출 실패 (시도 {attempt + 1}/{retries}): {e}")
        logging.error("최대 재시도 횟수 초과. 기본값 반환.")
        return {"sentiment_score": 0}

    # 추가 기능 : 실시간 데이터 통합 및 데이터프레임에 통합합
    def integrate_real_time_data(self, data, upbit_data, sentiment_score):
        """
        실시간 데이터(업비트 및 코인니스 감정 점수)를 통합
        실시간 데이터(업비트 및 코인니스)를 데이터프레임에 통합
        """
        try:
            # 업비트 데이터 통합
            data['upbit_price'] = upbit_data.get('trade_price', np.nan)
            data['price_change'] = upbit_data.get('signed_change_rate', np.nan)

            # 코인니스 감정 점수 통합
            data['sentiment_score'] = sentiment_score

            # 통합 검증 및 로그
            if data['upbit_price'].isna().all():
                logging.warning("모든 업비트 가격 데이터가 NaN입니다.")
            else:
                logging.info("실시간 데이터 통합 완료.")
        except Exception as e:
            logging.error(f"실시간 데이터 통합 실패: {e}")
        return data
    
    # 추가 기능 : 데이터프레임의 필수 열 검증 및 누락된 열 초기화 코드
def validate_dataframe_columns(data, required_columns):
    """
    데이터프레임의 필수 열 검증 및 누락된 열 초기화
    :param data: 데이터프레임
    :param required_columns: 필수 열 리스트
    :return: 검증 및 수정된 데이터프레임
    """
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logging.warning(f"누락된 열 발견: {missing_columns}")
        print(f"누락된 열이 발견되었습니다: {missing_columns}. 기본값으로 추가합니다.")
        for col in missing_columns:
            data[col] = np.nan  # 기본값은 NaN으로 설정
    else:
        logging.info("모든 필수 열이 데이터프레임에 포함되어 있습니다.")
        print("모든 필수 열이 정상적으로 포함되어 있습니다.")
    return data

    # 추가 기능 : 특정 열이 없더라도 시각화를 중단하지 않고 실행하도록 설계
    def visualize_data(data):
        """
        데이터 시각화 함수
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        if 'close' in data.columns:
            plt.plot(data.index, data['close'], label='Close Price')
        if 'bb_middle' in data.columns:
            plt.plot(data.index, data['bb_middle'], label='Bollinger Middle')
            plt.plot(data.index, data['bollinger_upper'], label='Bollinger Upper')
            plt.plot(data.index, data['bollinger_lower'], label='Bollinger Lower')
        plt.legend()
        plt.grid()
        plt.title("Technical Indicators Visualization")
        plt.show()



########################################################################################################
############################################ Main 함수 ##################################################
if __name__ == "__main__":
    import time
    import requests # type: ignore

    # 샘플 데이터 생성
    data = pd.DataFrame({
        'close': np.random.rand(100) * 100,
        'high': np.random.rand(100) * 100,
        'low': np.random.rand(100) * 100,
        'volume': np.random.rand(100) * 1000
    }, index=pd.date_range(start="2023-01-01", periods=100, freq="T"))

# 필수 열 검증 및 초기화
required_columns = [
    'close', 'high', 'low', 'volume',  # 기본 열
    'sentiment_score'  # 감정 점수를 사용하는 지표를 위한 추가 열
]
data = validate_dataframe_columns(data, required_columns)
# 기술적 지표 클래스 인스턴스 생성
indicators = TechnicalIndicators()
# 성능적 지표 클래스 인스턴스 생성
processor = DataProcessor()

################################ 실시간 데이터 수집 #############################
# 업비트 데이터 가져오기
print("=== 업비트 데이터 수집 및 처리 ===")
upbit_data = processor.process_upbit_data(market="KRW-BTC")

# 코인니스 뉴스 데이터 가져오기  
print("\n=== 코인니스 뉴스 데이터 수집 및 처리 ===")
coinness_news = processor.process_coinness_news()

###################################################성능 지표 테스트##################################################
# 27개 모든 지표 계산 함수 호출
# 기존 지표
try:
    data = indicators.calculate_rsi(data)  # RSI
    data = indicators.calculate_macd(data)  # MACD 
    data = indicators.calculate_ichimoku_cloud(data)  # Ichimoku Cloud
    data = indicators.calculate_donchian_channel(data)  # Donchian Channel
    data = indicators.calculate_vwap(data)  # VWAP
except Exception as e:
    print(f"지표 계산 중 오류 발생: {str(e)}")
    raise
    data = indicators.calculate_atr(data)  # ATR
    data = indicators.calculate_bollinger_bands(data)  # Bollinger Bands
    data = indicators.calculate_fibonacci_retracement(data)  # Fibonacci Retracement
    data = indicators.calculate_ema(data)  # EMA

    # 추가 지표
    data = indicators.calculate_keltner_channel(data)  # Keltner Channel
    data = indicators.calculate_supertrend(data)  # Supertrend
    data = indicators.calculate_parabolic_sar(data)  # Parabolic SAR
    data = indicators.calculate_cci(data)  # CCI
    data = indicators.calculate_roc(data)  # ROC
    data = indicators.calculate_williams_r(data)  # Williams %R
    data = indicators.calculate_stochastic_oscillator(data)  # Stochastic Oscillator
    data = indicators.calculate_adx(data)  # ADX
    data = indicators.calculate_mfi(data)  # MFI
    data = indicators.calculate_elder_ray(data)  # Elder-Ray Index
    data = indicators.calculate_obv(data)  # OBV
    data = indicators.calculate_hurst_exponent(data)  # Hurst Exponent
    data = indicators.calculate_fractal_dimension(data)  # Fractal Dimension
    data = indicators.calculate_adjusted_rsi(data, sentiment_scores=np.random.rand(100))  # Adjusted RSI
    data = indicators.calculate_sentiment_adjusted_macd(data, sentiment_scores=np.random.rand(100))  # Sentiment-Adjusted MACD
    data = indicators.calculate_multi_timeframe_ema(data)  # Multi-Timeframe EMA
    data = indicators.calculate_adjusted_vwap(data, sentiment_scores=np.random.rand(100))  # Adjusted VWAP
    data = indicators.calculate_sentiment_weighted_bollinger_bands(data, sentiment_scores=np.random.rand(100))  # Sentiment-Weighted Bollinger Bands

    # 필수 열 검증 및 초기화
    required_columns = [
        'close', 'high', 'low', 'volume',  # 기본 열
        'sentiment_score'  # 감정 점수를 사용하는 지표를 위한 추가 열
    ]
    data = validate_dataframe_columns(data, required_columns)

    # 업비트 데이터 수집 및 지표 계산
    markets = processor.fetch_upbit_all_markets()
    ticker_data = processor.fetch_upbit_ticker_all_batched(markets)
    data = pd.DataFrame(ticker_data)


###################################################기술 지표 테스트##################################################

    # 1. WebSocket 연결 테스트 (실시간 데이터 수집)
# 1. WebSocket 연결 테스트 (실시간 데이터 수집)
    print("=== WebSocket 실시간 데이터 테스트 ===")
    try:
        # 기존 호출 방식에 백오프 전략 포함
        processor.collect_real_time_websocket(max_retries=5, backoff_factor=2)  # 최대 5회 재연결, 백오프 포함
    except KeyboardInterrupt:
        print("WebSocket 테스트 종료")

    # 2. 데이터 스무딩 적용
    print("=== 데이터 스무딩 시작 ===")
    start = time.time()
    data['smoothed_close'] = processor.smooth_data(data['close'], method="sma", window=5)
    print(f"데이터 스무딩 완료: {time.time() - start:.4f}초")
    print(data[['close', 'smoothed_close']].head())

    # 3. 다중 시간 프레임 생성
    print("\n=== 다중 시간 프레임 생성 시작 ===")
    start = time.time()
    multi_timeframe_data = processor.apply_multiple_timeframes(data, intervals=["1T", "5T", "1H"])
    print(f"다중 시간 프레임 생성 완료: {time.time() - start:.4f}초")

    # 4. 뉴스 감정 점수 분석
    print("\n=== 뉴스 감정 점수 분석 시작 ===")
    sample_news = "Bitcoin prices are expected to surge this month due to institutional interest."
    sentiment_score = processor.analyze_sentiment(sample_news)
    print(f"뉴스 감정 점수: {sentiment_score}")

    # 5. 멀티스레딩 지표 계산
    print("\n=== 멀티스레딩 지표 계산 시작 ===")
    indicators = [
        lambda df: df.assign(rsi=np.random.rand(len(df))),  # Dummy RSI 계산
        lambda df: df.assign(macd=np.random.rand(len(df)))  # Dummy MACD 계산
    ]
    start = time.time()
    data = processor.calculate_indicators_multithreaded(data, indicators)
    print(f"멀티스레딩 지표 계산 완료: {time.time() - start:.4f}초")

    # 6. 고급화 함수 호출
    print("\n=== GPU 가속 볼린저 밴드 계산 시작 ===")
    start = time.time()
    data = processor.calculate_bollinger_bands_gpu(data)
    print(f"GPU 가속 볼린저 밴드 계산 완료: {time.time() - start:.4f}초")

    print("\n=== Wavelet Transform 기반 스무딩 시작 ===")
    start = time.time()
    data = processor.smooth_wavelet(data, 'close')
    print(f"Wavelet Transform 스무딩 완료: {time.time() - start:.4f}초")

    print("\n=== 적응형 ATR 계산 시작 ===")
    start = time.time()
    data = processor.adaptive_atr(data, period=14, multiplier=1.5)
    print(f"적응형 ATR 계산 완료: {time.time() - start:.4f}초")

    # 7. 결과 통합 및 반환
    print("\n=== 결과 통합 시작 ===")
    start = time.time()
    final_data = processor.integrate_results(data, multi_timeframe_data, sentiment_score)
    print(f"결과 통합 완료: {time.time() - start:.4f}초")

    # 8-1. 업비트 모든 마켓 리스트 가져오기기
    print("=== 업비트 마켓 리스트 수집 ===")
    markets = processor.fetch_upbit_all_markets()
    print(f"수집된 마켓 수: {len(markets)}")

    # 8-2. 업비트 모든 마켓의 실시간 티커 데이터 가져오기
    print("\n=== 업비트 실시간 티커 데이터 수집 ===")
    ticker_data = processor.fetch_upbit_ticker_all(markets)
    if ticker_data:
        print(f"수신된 데이터 수: {len(ticker_data)}")
        # 예시로 첫 5개 데이터 출력
        for ticker in ticker_data[:5]:
            print(ticker)
    else:
        print("실시간 데이터 수집 실패 또는 데이터 없음.")

    # 8-3. 업비트 실시간 데이터를 DataFrame 형태로 변환
    print("=== 업비트 마켓 리스트 및 티커 데이터 수집 ===")
    ticker_df = processor.process_upbit_data_to_dataframe()
    if not ticker_df.empty:
        print("수집된 티커 데이터:")
        print(ticker_df.head())
    else:
        print("업비트 데이터를 가져오지 못했습니다.")

    # 9. 코인니스 뉴스 데이터 가져오기
    print("\n=== 코인니스 뉴스 데이터 수집 및 처리 ===")
    coinness_news = processor.process_coinness_news()
    sentiment_score = coinness_news.get('sentiment_score', 0)  # 감정 점수 추출

    # 10. 실시간 데이터 통합
    print("\n=== 실시간 데이터 통합 ===")
    data = processor.integrate_real_time_data(data, upbit_data, sentiment_score)

    # 10-1. 필수 열 검증 및 추가
    required_columns = ['close', 'high', 'low', 'volume', 'open', 'upbit_price', 'price_change', 'sentiment_score']
    data = validate_dataframe_columns(data, required_columns)

    # 11. 결과 저장 및 시각화
    print("\n=== 결과 저장 및 시각화 ===")
    data.to_csv("final_results.csv", index=True)
    print("결과가 'final_results.csv' 파일로 저장되었습니다.")

    # 12. 결과 저장 및 시각화    
    print("\n=== 추가 지표 테스트 ===")
    data = processor.calculate_keltner_channel(data)
    print(data[['kc_upper', 'kc_lower']].head())

    data = processor.calculate_supertrend(data)
    print(data[['supertrend', 'supertrend_direction']].head())

    data = processor.calculate_parabolic_sar(data)
    print(data[['psar']].head())

    # 13. 성능 측정 테스트
    # 성능 비교 테스트
    print("\n=== GPU와 CPU 기반 성능 비교 ===")
    try:
        # GPU 기반 계산 시간 측정
        start_gpu = time.time()
        data = processor.calculate_bollinger_bands_gpu(data)
        gpu_time = time.time() - start_gpu
        
        # CPU 기반 계산 시간 측정
        start_cpu = time.time()
        data = processor.calculate_bollinger_bands(data)
        cpu_time = time.time() - start_cpu

        # 성능 비교 결과 로그 기록
        logging.info(f"GPU 가속 볼린저 밴드 계산 시간: {gpu_time:.4f}초, CPU 계산 시간: {cpu_time:.4f}초")
        print(f"GPU 시간: {gpu_time:.4f}초, CPU 시간: {cpu_time:.4f}초")
        print(f"성능 개선 비율: {cpu_time / gpu_time:.2f}배")
    except Exception as e:
        logging.error(f"성능 비교 중 오류 발생: {e}")


    # 14. 시각화 확장
    print("\n=== 결과 시각화 ===")
    fig, axs = plt.subplots(3, 1, figsize=(12, 15)) # type: ignore

    # Close 가격 및 Bollinger Bands
    axs[0].plot(data.index, data['close'], label='Close Price', color='blue')
    axs[0].plot(data.index, data['bb_middle'], label='Bollinger Middle', color='orange')
    axs[0].plot(data.index, data['bollinger_upper'], label='Bollinger Upper', color='green')
    axs[0].plot(data.index, data['bollinger_lower'], label='Bollinger Lower', color='red')
    axs[0].set_title("Bollinger Bands")
    axs[0].legend()
    axs[0].grid()

    # Supertrend
    if 'supertrend' in data.columns:
        axs[1].plot(data.index, data['close'], label='Close Price', color='blue')
        axs[1].plot(data.index, data['supertrend'], label='Supertrend', color='purple')
        axs[1].set_title("Supertrend")
        axs[1].legend()
        axs[1].grid()

    # Parabolic SAR
    if 'psar' in data.columns:
        axs[2].plot(data.index, data['close'], label='Close Price', color='blue')
        axs[2].scatter(data.index, data['psar'], label='Parabolic SAR', color='black', s=10)
        axs[2].set_title("Parabolic SAR")
        axs[2].legend()
        axs[2].grid()

    plt.tight_layout() # type: ignore
    plt.show()

    # 15. 결과 데이터프레임 검증
    print("\n=== 결과 데이터프레임 검증 ===")
    validate_dataframe_columns(data)

    # 16. 데이터터티 검증 및 시각화
    print("=== 업비트 데이터 수집 및 검증 ===")
    markets = processor.fetch_upbit_all_markets()
    ticker_data = processor.fetch_upbit_ticker_all_batched(markets)
    df = pd.DataFrame(ticker_data)

    # 16. 최적화 함수 호출
    # 실시간 데이터 수집
    print("=== 업비트 데이터 수집 ===")
    ticker_df = processor.process_upbit_data_to_dataframe()

    # 지표 계산
    if not ticker_df.empty:
        ticker_df = processor.calculate_bollinger_bands(ticker_df)
        ticker_df = processor.calculate_keltner_channel(ticker_df)

        # 데이터프레임 최적화
        ticker_df = optimize_dataframe(ticker_df)

        # 검증 및 시각화
        validate_dataframe_columns(ticker_df)
        visualize_data(ticker_df)
    else:
        print("업비트 데이터를 가져오지 못했습니다.")

    
    # 데이터 검증
    print("\n=== 데이터프레임 검증 ===")
    validate_dataframe_columns(df)

    # 데이터 시각화
    print("\n=== 데이터 시각화 ===")
    visualize_data(df)

########################################################################################################################

    # 결과 출력 / 기술적 지표
    print(data.head())
    print(f"데이터프레임에 포함된 열: {data.columns.tolist()}")

    # 결과 출력 / 성능 지표
    print("\n=== 최종 데이터 출력 ===")
    print(df.head())
    print(f"데이터프레임에 포함된 열: {df.columns.tolist()}")