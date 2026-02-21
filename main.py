import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import requests
import os
import time
import tempfile
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
MACRO_EVENT_DATES = [
    "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17", 
    "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16", 
    "2026-02-13", "2026-03-12", "2026-04-10"                
]

# ==========================================
# 2. 알림 모듈 (Telegram Notification)
# ==========================================
def send_telegram(message):
    """단일 텔레그램 메시지 발송"""
    if not BOT_TOKEN or not CHAT_ID: 
        print("⚠️ 텔레그램 환경변수(TG_TOKEN, TG_CHAT_ID) 미설정\n", message)
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    try: requests.post(url, data=data, timeout=10)
    except Exception as e: print(f"텔레그램 발송 실패: {e}")

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
    except: pass
    return False, None

def get_optimal_metrics(df):
    """3년 치 과거 데이터를 바탕으로 종목별 최적 ATR 배수 및 갭(Gap) 한도 추출"""
    mae_list, historical_gaps, reversal_strengths = [], [], []
    signals = df[df['Sync_Signal']].index
    
    for idx in signals[:-1]:
        loc = df.index.get_loc(idx)
        if loc + 11 >= len(df): continue 
        
        close_p = float(df.iloc[loc]['Close'])
        atr_p = float(df.iloc[loc]['ATR'])
        low_p = float(df.iloc[loc]['Low'])
        next_open_p = float(df.iloc[loc+1]['Open'])
        
        # 갭(Gap) 상승률 기록
        historical_gaps.append(((next_open_p - close_p) / close_p) * 100)

        f_low = float(df.iloc[loc+1 : loc+11]['Low'].min())
        if (close_p - f_low) > 0 and atr_p > 0: mae_list.append((close_p - f_low) / atr_p)
        if atr_p > 0: reversal_strengths.append((close_p - low_p) / atr_p)
    
    if len(mae_list) < 5: return 2.0, 2.0, 0.5, True 
        
    opt_mult = max(np.percentile(mae_list, 90), 2.0)
    
    # [종목별 최적화 갭] 과거 갭의 80백분위수 산출 (최소 0.5% ~ 최대 4.0%로 안전 가드)
    max_gap_threshold = np.clip(np.percentile(historical_gaps, 80), 0.5, 4.0)
    
    min_reversal_factor = np.percentile(reversal_strengths, 25) 
    
    return opt_mult, max_gap_threshold, min_reversal_factor, (opt_mult <= 2.0)

def calc_rs_score(df, spy_df):
    """가중 누적 수익률을 활용한 상대강도(RS) 점수 산출"""
    try:
        periods, weights, score = [63, 126, 189, 252], [0.4, 0.2, 0.2, 0.2], 0
        for p, w in zip(periods, weights):
            if len(df) > p and len(spy_df) > p:
                score += ((float(df['Close'].iloc[-1]) / float(df['Close'].iloc[-p])) / 
                          (float(spy_df['Close'].iloc[-1]) / float(spy_df['Close'].iloc[-p]))) * w
        return score
    except: return 0

# --- [4. 유니버스 데이터 수집 함수 (안전망 확보)] ---
def fetch_wiki_tickers_safe(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code != 200: return []
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(res.text)
            tmp_path = f.name
        tables = pd.read_html(tmp_path)
        os.remove(tmp_path)
        for df in tables:
            if 'Symbol' in df.columns: return df['Symbol'].tolist()
            if 'Ticker' in df.columns: return df['Ticker'].tolist()
    except: pass
    return []

def fetch_fallback_tickers():
    tickers = []
    try:
        sp500_df = pd.read_csv("https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv")
        if 'Symbol' in sp500_df.columns: tickers.extend(sp500_df['Symbol'].tolist())
    except: pass
    return tickers

# ==========================================
# 5. 메인 분석 프로세스 (Main Process)
# ==========================================
def analyze():
    start_date = (pd.Timestamp.now() - pd.DateOffset(years=3)).strftime('%Y-%m-%d')
    print(f"🚀 PRO-MASTER V5 스캔 시작: {datetime.now()}")

    # [STEP 1] 매크로 리스크 필터
    is_macro, reason = is_macro_event_day()
    if is_macro:
        send_telegram(f"🛑 <b>매크로 리스크 감지</b>\n{reason}은(는) 시장 변동성이 극도로 높습니다. 안전을 위해 오늘 스캔을 건너뜁니다.")
        return

    # [STEP 2] 시장 추세 필터 (SPY 중장기 추세 & VIX 변동성)
    print("시장 상태(SPY/VIX) 검증 중...")
    try: 
        m_data = yf.download(["SPY", "^VIX"], start=start_date, progress=False)['Close']
    except Exception as e: 
        print(f"지수 데이터 로드 실패: {e}")
        return
        
    if m_data.empty or 'SPY' not in m_data or '^VIX' not in m_data: return
    spy, vix = m_data['SPY'].dropna(), m_data['^VIX'].dropna()
    
    if len(spy) < 200 or len(vix) < 1: return
    
    spy_ma200, spy_ma50 = ta.sma(spy, 200), ta.sma(spy, 50)
    spy_curr, vix_curr = float(spy.iloc[-1]), float(vix.iloc[-1])
    
    if not (spy_curr > float(spy_ma200.iloc[-1]) and spy_curr > float(spy_ma50.iloc[-1]) and vix_curr < 25):
        send_telegram(f"⚠️ <b>시장 필터 작동</b>\nSPY 중장기 역배열 또는 VIX({vix_curr:.2f}) 불안정으로 매수 스캔을 중단합니다.")
        return

    # [STEP 3] 유니버스 구성 (S&P 500 + NASDAQ 100)
    print("유니버스 구성 및 데이터 다운로드 중...")
    tickers = []
    for _ in range(3):
        sp500 = fetch_wiki_tickers_safe('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        ndx100 = fetch_wiki_tickers_safe('https://en.wikipedia.org/wiki/Nasdaq-100')
        tickers = list(set(sp500 + ndx100))
        if len(tickers) > 400: break
        time.sleep(3)
        
    if len(tickers) < 400: tickers = list(set(fetch_fallback_tickers()))
    tickers = [t.replace('.', '-') for t in tickers]

    if len(tickers) < 100:
        send_telegram("⚠️ <b>데이터 수집 최종 실패</b>\n티커 명단 확보에 실패했습니다.")
        return

    raw_data = yf.download(tickers, start=start_date, group_by='ticker', threads=True, progress=False)
    
    # [STEP 4] RS (Relative Strength) 스코어링 및 상위 주도주 추출
    print("전체 유니버스 상대강도(RS) 랭킹 산출 중...")
    rs_scores = {}
    for ticker in tickers:
        try:
            df = raw_data[ticker].dropna() if isinstance(raw_data.columns, pd.MultiIndex) else raw_data.dropna()
            if len(df) < 260: continue
            
            cp, avg_v = float(df['Close'].iloc[-1]), float(df['Volume'].rolling(20).mean().iloc[-1])
            if cp < 10 or (cp * avg_v < 20000000): continue 
            
            # 정배열 기초 필터
            if cp > float(ta.sma(df['Close'], 200).iloc[-1]) and cp > float(ta.sma(df['Close'], 50).iloc[-1]):
                score = calc_rs_score(df, spy)
                if score > 0: rs_scores[ticker] = score
        except: continue

    if not rs_scores: return
    rs_ranks = pd.Series(rs_scores).rank(pct=True) * 100
    leading_stocks = rs_ranks[rs_ranks >= 80].index.tolist() 

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
            
            # 조건 벡터 연산 (기관 매집 10배 허용)
            cond_increase = df['Volume'] > df['prev_v']
            cond_exception = (df['prev_v'] > df['avg_v20'] * 1.5) & (df['Volume'] > df['avg_v20'])
            df['is_vol_ok'] = (cond_increase | cond_exception) & (df['Volume'] < df['avg_v20'] * 10.0)
            
            df['c_range'] = df['High'] - df['Low']
            df['rev_pos'] = np.where(df['c_range'] > 0, (df['Close'] - df['Low']) / df['c_range'], 0)
            df['is_green'] = df['Close'] > df['Open']
            
            # Sync_Signal: 정배열 + BB 중심선 하단 + 양봉 + 밑꼬리(캔들상단) 마감 + 거래량 동반
            df['Sync_Signal'] = (df['MA20'] > df['MA50']) & (df['Close'] <= df['BB_MID']) & \
                                df['is_green'] & (df['rev_pos'] >= 0.6) & df['is_vol_ok']
                                
            if df['Sync_Signal'].iloc[-1]:
                
                # 실적발표 임박 종목 제외
                near_earnings, e_date = is_earnings_near(ticker)
                if near_earnings: continue
                
                # [DNA 추출] 3년 데이터로 종목별 최적 파라미터 도출
                opt_mult, max_gap_limit, min_rev, is_def = get_optimal_metrics(df)
                
                cp = float(df['Close'].iloc[-1])
                atr = float(df['ATR'].iloc[-1])
                curr_rev = float(df['rev_pos'].iloc[-1])
                
                if curr_rev < min_rev: continue # 반등강도 미달 패스
                
                # 포지션 사이징
                stop_dist = opt_mult * atr
                limit_stop_l = cp - stop_dist
                qty = int(RISK_AMOUNT // stop_dist) if stop_dist > 0 else 0
                
                # [안전장치 1] 진입 제한 상한가 (과거 갭 데이터 반영)
                max_entry_price = cp * (1 + max_gap_limit / 100)
                
                # [안전장치 2] 1차 익절 타겟 & 최소 손익비 방어 (어제까지의 20일 전고점)
                target_p = float(df['High'].iloc[-21:-1].max())
                if target_p < cp + (stop_dist * 1.5): 
                    target_p = cp + (stop_dist * 2.0)

                atr_label = "하한선 방어" if is_def else "동적 계산"

                msg_list.append(
                    f"🚀 <b>[실전 주문] {ticker}</b> (RS Rank: {rs_ranks[ticker]:.1f})\n"
                    f"━━━━━━━━━━━━━━━━━━\n"
                    f"<b>[진입 플랜]</b>\n"
                    f"🎯 <b>조건부 돌파매수 : ${cp:.2f} 돌파 시</b>\n"
                    f"   <i>(※ 진입제한 상한가: ${max_entry_price:.2f} / 과거 갭 데이터 기준)</i>\n"
                    f"🛑 <b>초기 스탑로스 : ${limit_stop_l:.2f}</b>\n"
                    f"📦 <b>매수 수량 : {qty}주</b> (리스크 ${RISK_AMOUNT})\n"
                    f"🛡️ 방어 기준 : ATR {opt_mult:.2f}배 ({atr_label})\n"
                    f"━━━━━━━━━━━━━━━━━━\n"
                    f"<b>[매도 작전 명령서]</b>\n"
                    f"💰 <b>1차 익절(50%) : ${target_p:.2f}</b>\n"
                    f"📈 <b>추세 청산(50%) : 종가 SMA 20 이탈 시</b>\n"
                    f"💡 <i>(Tip: 1차 익절 도달 시 남은 수량 손절가를 진입가로 변경)</i>\n\n"
                )
        except Exception: 
            continue

    # [STEP 6] 최종 리포트 발송
    header = f"<b>📅 {datetime.now().date()} PRO 퀀트 리포트 (V5)</b>\n\n"
    send_telegram_chunks(msg_list, header, f"\n<b>[결과]</b> 승률 높은 최적화 타점 {len(msg_list)}개 포착")

if __name__ == "__main__":
    analyze()
