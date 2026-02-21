import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import requests
import os
import time
import tempfile
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# --- [1. 자산 및 리스크 설정] ---
BOT_TOKEN = os.environ.get('TG_TOKEN')
CHAT_ID = os.environ.get('TG_CHAT_ID')
RISK_AMOUNT = 200 # 1회 타점당 고정 리스크 ($200)

def send_telegram(message):
    if not BOT_TOKEN or not CHAT_ID: 
        print("⚠️ 텔레그램 설정 누락:\n", message)
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    try: requests.post(url, data=data, timeout=10)
    except: pass

def send_telegram_chunks(msg_list, header, footer):
    if not msg_list:
        send_telegram(header + "❌ <b>오늘은 조건에 맞는 1급 주도주가 없습니다.</b>\n" + footer)
        return
    chunk_size = 3 
    for i in range(0, len(msg_list), chunk_size):
        chunk = msg_list[i:i + chunk_size]
        body = "\n".join(chunk)
        title = f"{header} (파트 {i//chunk_size + 1})\n\n"
        send_telegram(title + body + (footer if i + chunk_size >= len(msg_list) else ""))
        time.sleep(1)

# --- [2. 핵심 퀀트 엔진: 매수 방어선 계산] ---
def get_optimal_buy_metrics(df):
    """과거 시그널 기반 매수 방어선(ATR 배수) 및 갭 한도 도출"""
    mae_list, historical_gaps, reversal_strengths = [], [], []
    signals = df[df['Sync_Signal']].index
    
    for idx in signals[:-1]:
        loc = df.index.get_loc(idx)
        if loc + 11 >= len(df): continue 
        
        close_p, atr_p, low_p = float(df.iloc[loc]['Close']), float(df.iloc[loc]['ATR']), float(df.iloc[loc]['Low'])
        next_open_p = float(df.iloc[loc+1]['Open'])
        
        historical_gaps.append(((next_open_p - close_p) / close_p) * 100)
        f_low = float(df.iloc[loc+1 : loc+11]['Low'].min())
        f_max = float(df.iloc[loc+1 : loc+11]['High'].max())
        
        if (close_p - f_low) > 0 and atr_p > 0: mae_list.append((close_p - f_low) / atr_p)
        if f_max > close_p and atr_p > 0: reversal_strengths.append((close_p - low_p) / atr_p)
    
    if len(mae_list) < 5: return 2.0, 2.0, 0.5, True 
        
    raw_opt_mult = np.percentile(mae_list, 90)
    is_defense = raw_opt_mult <= 2.0
    opt_mult = max(raw_opt_mult, 2.0) 
    max_gap_threshold = max(np.percentile(historical_gaps, 80), 0.5)
    min_reversal_factor = np.percentile(reversal_strengths, 25) 
    
    return opt_mult, max_gap_threshold, min_reversal_factor, is_defense

def calc_rs_score(df, spy_df):
    try:
        periods, weights, score = [63, 126, 189, 252], [0.4, 0.2, 0.2, 0.2], 0
        for p, w in zip(periods, weights):
            if len(df) > p and len(spy_df) > p:
                score += ((float(df['Close'].iloc[-1]) / float(df['Close'].iloc[-p])) / 
                          (float(spy_df['Close'].iloc[-1]) / float(spy_df['Close'].iloc[-p]))) * w
        return score
    except: return 0

# --- [3. 메인 분석 로직] ---
def analyze():
    start_date = (pd.Timestamp.now() - pd.DateOffset(years=3)).strftime('%Y-%m-%d')
    print(f"🚀 스캔 시작: {datetime.now()} (기준일: {start_date})")
    
    # [수정 2] 시장 필터 민감도 완화 (MA5 -> MA50으로 변경하여 넉넉한 눌림목 허용)
    print("시장 상태(SPY/VIX) 확인 중...")
    try: m_data = yf.download(["SPY", "^VIX"], start=start_date, progress=False)['Close']
    except: return
    spy, vix = m_data['SPY'].dropna(), m_data['^VIX'].dropna()
    if len(spy) < 200 or len(vix) < 1: return
    
    spy_ma200, spy_ma50 = ta.sma(spy, 200), ta.sma(spy, 50)
    spy_curr, vix_curr = float(spy.iloc[-1]), float(vix.iloc[-1])
    
    if not (spy_curr > float(spy_ma200.iloc[-1]) and spy_curr > float(spy_ma50.iloc[-1]) and vix_curr < 25):
        send_telegram(f"⚠️ <b>시장 필터 작동</b>\nSPY 중장기 역배열 또는 VIX({vix_curr:.2f}) 불안정으로 매수 스캔 중단.")
        return

    print("유니버스 구성 중...")
    tickers = []
    try:
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
        ndx100 = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')[4]['Ticker'].tolist()
        tickers = list(set(sp500 + ndx100))
    except: 
        tickers = pd.read_csv("https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv")['Symbol'].tolist()
    
    tickers = [t.replace('.', '-') for t in tickers]
    raw_data = yf.download(tickers, start=start_date, group_by='ticker', threads=True, progress=False)
    
    rs_scores = {}
    for ticker in tickers:
        try:
            df = raw_data[ticker].dropna() if isinstance(raw_data.columns, pd.MultiIndex) else raw_data.dropna()
            if len(df) < 260: continue
            cp, avg_v20 = float(df['Close'].iloc[-1]), float(df['Volume'].rolling(20).mean().iloc[-1])
            if cp < 10 or (cp * avg_v20 < 20000000): continue
            if cp > float(ta.sma(df['Close'], 200).iloc[-1]) and cp > float(ta.sma(df['Close'], 50).iloc[-1]):
                score = calc_rs_score(df, spy)
                if score > 0: rs_scores[ticker] = score
        except: continue

    if not rs_scores: return
    rs_ranks = pd.Series(rs_scores).rank(pct=True) * 100
    leading_stocks = rs_ranks[rs_ranks >= 80].index.tolist()

    msg_list = []
    final_pass_count = 0
    
    for ticker in leading_stocks:
        try:
            df = raw_data[ticker].dropna()
            df['MA20'], df['MA50'] = ta.sma(df['Close'], 20), ta.sma(df['Close'], 50)
            df['BB_MID'] = ta.bbands(df['Close'], 20, 2.0)['BBM_20_2.0']
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], 14)
            df['avg_v20'] = ta.sma(df['Volume'], 20)
            df['prev_v'] = df['Volume'].shift(1)
            
            cp = float(df['Close'].iloc[-1])
            
            # [수정 3] 거래량 상한선 필터 완화 (3.0배 -> 10.0배로 열어두어 기관 Capitulation 포착)
            df['is_vol_ok'] = ((df['Volume'] > df['prev_v']) | ((df['prev_v'] > df['avg_v20'] * 1.5) & (df['Volume'] > df['avg_v20']))) & (df['Volume'] < df['avg_v20'] * 10.0)
            
            c_range = float(df['High'].iloc[-1]) - float(df['Low'].iloc[-1])
            rev_pos = (cp - float(df['Low'].iloc[-1])) / c_range if c_range > 0 else 0
            
            df['is_green'] = df['Close'] > df['Open']
            df['Sync_Signal'] = (df['MA20'] > df['MA50']) & (df['Close'] <= df['BB_MID']) & df['is_green'] & (df['rev_pos'] >= 0.6) & df['is_vol_ok']
            
            if df['Sync_Signal'].iloc[-1]:
                opt_mult, max_gap, min_rev, is_def = get_optimal_buy_metrics(df)
                curr_rev = (cp - float(df['Low'].iloc[-1])) / float(df['ATR'].iloc[-1])
                
                if curr_rev < min_rev: continue 
                
                final_pass_count += 1
                
                stop_dist = opt_mult * float(df['ATR'].iloc[-1])
                limit_stop_l = cp - stop_dist
                qty = int(RISK_AMOUNT // stop_dist) if stop_dist > 0 else 0
                max_entry_price = cp * (1 + max_gap / 100)
                
                # [수정 1] 과최적화 제거 및 유니버설 매도 전략 하드코딩 (20일 전고점 타겟)
                target_price = float(df['High'].iloc[-20:].max())
                if target_price <= cp: target_price = cp + (stop_dist * 2.0) # 전고점이 너무 낮으면 보정

                # [수정 4] 떨어지는 칼날 방어용 멘트 (Stop Buy / 조건부 매수)
                msg_list.append(
                    f"🚀 <b>[실전 주문] {ticker}</b> (RS Rank: {rs_ranks[ticker]:.1f})\n"
                    f"━━━━━━━━━━━━━━━━━━\n"
                    f"<b>[진입 플랜 - 역배열 갭하락 회피]</b>\n"
                    f"🎯 <b>조건부 돌파매수 : ${cp:.2f} 돌파 시 체결</b>\n"
                    f"   <i>(※ 단, 시가가 ${max_entry_price:.2f} 초과 시 매수 취소)</i>\n"
                    f"🛑 <b>초기 스탑로스 : ${limit_stop_l:.2f}</b>\n"
                    f"📦 <b>매수 수량 : {qty}주</b> (리스크 $200 고정)\n"
                    f"━━━━━━━━━━━━━━━━━━\n"
                    f"<b>[매도 작전 명령서 - 유니버설 하이브리드]</b>\n"
                    f"💰 <b>1차 익절(50%) : ${target_price:.2f}</b> (최근 20일 전고점)\n"
                    f"📈 <b>추세 청산(50%) : 종가 SMA 20 이탈 시 매도</b>\n"
                    f"💡 <i>(Tip: 1차 익절 도달 시, 남은 물량 손절가를 본전으로 올리세요)</i>\n\n"
                )
        except: continue

    header = f"<b>📅 {datetime.now().date()} 퀀트 보고서 (PRO-MASTER V2)</b>\n\n"
    footer = f"\n<b>[결과]</b> 타점 {len(msg_list)}개 포착"
    send_telegram_chunks(msg_list, header, footer)

if __name__ == "__main__":
    print("🚀 PRO-MASTER V2 퀀트 스캐너 가동...")
    analyze()
