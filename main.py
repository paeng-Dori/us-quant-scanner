import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import requests
import os
import time
from datetime import datetime, timedelta
import warnings

# 불필요한 경고문구 숨김 처리
warnings.filterwarnings('ignore')

# ==========================================
# 1. 시스템 및 리스크 설정 (Configuration)
# ==========================================
BOT_TOKEN = os.environ.get('TG_TOKEN')
CHAT_ID = os.environ.get('TG_CHAT_ID')
RISK_AMOUNT = 200        # 1회 타점당 고정 리스크 ($200)
EARNINGS_WINDOW = 7      # 실적 발표 전후 피할 기간 (일)

# 2026년 주요 매크로 일정 (FOMC, 주요 CPI 발표일 등)
# 시장 변동성이 극대화되는 날짜 목록 (정기적인 업데이트 필요)
MACRO_EVENT_DATES = [
    "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17", # 2026 FOMC 회의
    "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16", 
    "2026-02-13", "2026-03-12", "2026-04-10"                # 주요 CPI
]

# ==========================================
# 2. 알림 모듈 (Telegram Notification)
# ==========================================
def send_telegram(message):
    """단일 텔레그램 메시지 발송"""
    if not BOT_TOKEN or not CHAT_ID: 
        print("⚠️ 텔레그램 환경변수(TG_TOKEN, TG_CHAT_ID)가 설정되지 않았습니다.\n", message)
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    try: 
        requests.post(url, data=data, timeout=10)
    except Exception as e: 
        print(f"텔레그램 발송 실패: {e}")

def send_telegram_chunks(msg_list, header, footer):
    """메시지 분할 발송 (도배 방지 및 가독성 확보)"""
    if not msg_list:
        send_telegram(header + "❌ <b>오늘은 조건에 맞는 1급 주도주가 없습니다.</b>\n" + footer)
        return
    chunk_size = 3 
    for i in range(0, len(msg_list), chunk_size):
        chunk = msg_list[i:i + chunk_size]
        body = "\n".join(chunk)
        title = f"{header} (파트 {i//chunk_size + 1})\n\n"
        send_telegram(title + body + (footer if i + chunk_size >= len(msg_list) else ""))
        time.sleep(1) # API Rate Limit 보호

# ==========================================
# 3. 핵심 필터 및 연산 엔진 (Core Engine)
# ==========================================
def is_macro_event_day():
    """매크로 이벤트(오늘/내일) 필터링"""
    today_str = datetime.now().strftime('%Y-%m-%d')
    tomorrow_str = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    if today_str in MACRO_EVENT_DATES: return True, "오늘(매크로 지표 발표 당일)"
    if tomorrow_str in MACRO_EVENT_DATES: return True, "내일(매크로 지표 발표 예정)"
    return False, None

def is_earnings_near(ticker_symbol):
    """실적 발표일 근접 여부 필터링"""
    try:
        ticker_obj = yf.Ticker(ticker_symbol)
        calendar = ticker_obj.calendar
        if calendar is not None and not calendar.empty:
            e_date = calendar.loc['Earnings Date'].values[0] if 'Earnings Date' in calendar.index else calendar.iloc[0, 0]
            if isinstance(e_date, (datetime, pd.Timestamp)):
                days_to_earnings = (e_date.date() - datetime.now().date()).days
                if 0 <= days_to_earnings <= EARNINGS_WINDOW:
                    return True, e_date.date()
    except Exception: 
        pass
    return False, None

# ==========================================
# 4. 메인 분석 프로세스 (Main Process)
# ==========================================
def analyze():
    start_date = (pd.Timestamp.now() - pd.DateOffset(years=3)).strftime('%Y-%m-%d')
    print(f"🚀 PRO-MASTER V4.1 스캔 시작: {datetime.now()}")

    # [STEP 1] 매크로 리스크 필터
    is_macro, reason = is_macro_event_day()
    if is_macro:
        send_telegram(f"🛑 <b>매크로 리스크 감지</b>\n{reason}은(는) 시장 변동성이 극도로 높습니다. 자산 보호를 위해 오늘 스캔을 건너뜁니다.")
        return

    # [STEP 2] 시장 추세 필터 (SPY 중장기 추세 & VIX 변동성)
    print("시장 상태(SPY/VIX) 검증 중...")
    try: 
        m_data = yf.download(["SPY", "^VIX"], start=start_date, progress=False)['Close']
    except Exception as e: 
        print(f"지수 데이터 로드 실패: {e}")
        return
        
    spy, vix = m_data['SPY'].dropna(), m_data['^VIX'].dropna()
    spy_ma200, spy_ma50 = ta.sma(spy, 200), ta.sma(spy, 50)
    spy_curr, vix_curr = float(spy.iloc[-1]), float(vix.iloc[-1])
    
    if not (spy_curr > float(spy_ma200.iloc[-1]) and spy_curr > float(spy_ma50.iloc[-1]) and vix_curr < 25):
        send_telegram(f"⚠️ <b>시장 필터 작동</b>\nSPY 역배열 또는 VIX({vix_curr:.2f}) 불안정으로 매수 스캔을 중단합니다.")
        return

    # [STEP 3] 유니버스 구성 (S&P 500 + NASDAQ 100)
    print("유니버스 구성 및 데이터 다운로드 중...")
    try:
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
        ndx100 = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')[4]['Ticker'].tolist()
        tickers = list(set(sp500 + ndx100))
    except Exception: 
        print("위키피디아 파싱 실패, Fallback CSV 사용")
        tickers = pd.read_csv("https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv")['Symbol'].tolist()
    
    tickers = [t.replace('.', '-') for t in tickers]
    raw_data = yf.download(tickers, start=start_date, group_by='ticker', threads=True, progress=False)
    
    # [STEP 4] RS (Relative Strength) 스코어링 및 상위 주도주 추출
    print("전체 유니버스 상대강도(RS) 랭킹 산출 중...")
    rs_scores = {}
    for ticker in tickers:
        try:
            df = raw_data[ticker].dropna() if isinstance(raw_data.columns, pd.MultiIndex) else raw_data.dropna()
            if len(df) < 260: continue
            
            cp, avg_v = float(df['Close'].iloc[-1]), float(df['Volume'].rolling(20).mean().iloc[-1])
            if cp < 10 or (cp * avg_v < 20000000): continue # 동전주 및 거래대금 미달 제외
            
            # 정배열 기초 필터
            if cp > float(ta.sma(df['Close'], 200).iloc[-1]) and cp > float(ta.sma(df['Close'], 50).iloc[-1]):
                periods, weights, score = [63, 126, 189, 252], [0.4, 0.2, 0.2, 0.2], 0
                for p, w in zip(periods, weights):
                    score += ((float(df['Close'].iloc[-1]) / float(df['Close'].iloc[-p])) / 
                              (float(spy.iloc[-1]) / float(spy.iloc[-p]))) * w
                if score > 0: rs_scores[ticker] = score
        except Exception: 
            continue

    if not rs_scores: return
    rs_ranks = pd.Series(rs_scores).rank(pct=True) * 100
    leading_stocks = rs_ranks[rs_ranks >= 80].index.tolist() # 상위 20% 주도주

    # [STEP 5] 정밀 타점 스캔 및 실적 발표 필터
    print(f"상위 주도주 {len(leading_stocks)}개 정밀 차트 스캔 중...")
    msg_list = []
    
    for ticker in leading_stocks:
        try:
            df = raw_data[ticker].dropna()
            
            # 기술적 지표 생성
            df['MA20'], df['MA50'] = ta.sma(df['Close'], 20), ta.sma(df['Close'], 50)
            df['BB_MID'] = ta.bbands(df['Close'], 20, 2.0)['BBM_20_2.0']
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], 14)
            df['avg_v20'] = ta.sma(df['Volume'], 20)
            df['prev_v'] = df['Volume'].shift(1)
            
            # 조건 벡터 연산
            df['is_vol_ok'] = ((df['Volume'] > df['prev_v']) | ((df['prev_v'] > df['avg_v20'] * 1.5) & (df['Volume'] > df['avg_v20']))) & (df['Volume'] < df['avg_v20'] * 10.0)
            df['c_range'] = df['High'] - df['Low']
            df['rev_pos'] = np.where(df['c_range'] > 0, (df['Close'] - df['Low']) / df['c_range'], 0)
            df['is_green'] = df['Close'] > df['Open']
            
            # Sync_Signal: 정배열 + BB 중심선 하단 + 양봉 캔들상단 마감 + 거래량 동반
            if (df['MA20'].iloc[-1] > df['MA50'].iloc[-1]) and (df['Close'].iloc[-1] <= df['BB_MID'].iloc[-1]) and \
               df['is_green'].iloc[-1] and (df['rev_pos'].iloc[-1] >= 0.6) and df['is_vol_ok'].iloc[-1]:
                
                # 실적발표 임박 종목 제외
                near_earnings, e_date = is_earnings_near(ticker)
                if near_earnings: 
                    print(f"⏭️ {ticker} 스킵 (실적 발표 임박: {e_date})")
                    continue
                
                cp = float(df['Close'].iloc[-1])
                atr = float(df['ATR'].iloc[-1])
                
                # 포지션 사이징 및 타겟 산출 (유니버설 하드코딩 기준)
                stop_dist = 2.0 * atr
                limit_stop_l = cp - stop_dist
                qty = int(RISK_AMOUNT // stop_dist) if stop_dist > 0 else 0
                
                target_p = float(df['High'].iloc[-21:-1].max()) # 최근 20일 고점
                if target_p <= cp: target_p = cp + (3.0 * atr)  # 이미 신고가 부근이면 보정

                msg_list.append(
                    f"🚀 <b>[실전 주문] {ticker}</b> (RS Rank: {rs_ranks[ticker]:.1f})\n"
                    f"━━━━━━━━━━━━━━━━━━\n"
                    f"🎯 <b>조건부 돌파매수 : ${cp:.2f} 돌파 시</b>\n"
                    f"🛑 <b>초기 스탑로스 : ${limit_stop_l:.2f}</b>\n"
                    f"📦 <b>매수 수량 : {qty}주</b> (리스크 ${RISK_AMOUNT})\n"
                    f"━━━━━━━━━━━━━━━━━━\n"
                    f"💰 <b>1차 익절(50%) : ${target_p:.2f}</b>\n"
                    f"📈 <b>추세 청산(50%) : 종가 SMA 20 이탈 시</b>\n\n"
                )
        except Exception: 
            continue

    # [STEP 6] 최종 리포트 발송
    header = f"<b>📅 {datetime.now().date()} PRO 퀀트 리포트 (V4.1)</b>\n\n"
    send_telegram_chunks(msg_list, header, f"\n<b>[결과]</b> 승률 높은 타점 {len(msg_list)}개 포착")

if __name__ == "__main__":
    analyze()
