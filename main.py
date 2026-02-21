import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import requests
import os
import time
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# --- [1. 자산 및 리스크 설정] ---
BOT_TOKEN = os.environ.get('TG_TOKEN')
CHAT_ID = os.environ.get('TG_CHAT_ID')

RISK_AMOUNT = 200         # 1회 타점당 고정 리스크 ($200)
MAX_PER_SECTOR = 2        # ⚠️ 동일 섹터 최대 진입 허용 개수 (분산 투자용)

def send_telegram(message):
    if not BOT_TOKEN or not CHAT_ID: 
        print(message)
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    try: requests.post(url, data=data, timeout=10)
    except: pass

def send_telegram_chunks(msg_list, header, footer):
    if not msg_list:
        send_telegram(header + "❌ <b>오늘은 조건에 맞는 1급 주도주가 없습니다.</b>\n" + footer)
        return
    for i in range(0, len(msg_list), 5):
        send_telegram(header + "\n".join(msg_list[i:i+5]) + footer)
        time.sleep(1)

# --- [2. 핵심 퀀트 엔진] ---
def get_optimal_metrics(df):
    mae_list, historical_gaps, reversal_strengths = [], [], []
    signals = df[df['Sync_Signal']].index
    for idx in signals[:-1]:
        loc = df.index.get_loc(idx)
        if loc + 11 >= len(df): continue
        cp, atr, low = float(df.iloc[loc]['Close']), float(df.iloc[loc]['ATR']), float(df.iloc[loc]['Low'])
        next_open = float(df.iloc[loc+1]['Open'])
        historical_gaps.append((next_open - cp) / cp * 100)
        f_low, f_high = float(df.iloc[loc+1:loc+11]['Low'].min()), float(df.iloc[loc+1:loc+11]['High'].max())
        if atr > 0:
            if (cp - f_low) > 0: mae_list.append((cp - f_low) / atr)
            if f_high > cp: reversal_strengths.append((cp - low) / atr)
    if len(mae_list) < 10: return 2.0, 2.0, 0.5
    return (max(np.percentile(mae_list, 90), 2.0), np.percentile(historical_gaps, 80), np.percentile(reversal_strengths, 25))

def calc_rs_score(df, spy):
    score = 0
    periods, weights = [63, 126, 189, 252], [0.4, 0.2, 0.2, 0.2]
    for p, w in zip(periods, weights):
        if len(df) > p and len(spy) > p:
            score += (float(df['Close'].iloc[-1]) / float(df['Close'].iloc[-p])) / \
                     (float(spy.iloc[-1]) / float(spy.iloc[-p])) * w
    return score

def is_earnings_near(ticker_symbol):
    try:
        cal = yf.Ticker(ticker_symbol).calendar
        if cal is not None and not cal.empty:
            ed = cal.loc['Earnings Date'].values[0] if 'Earnings Date' in cal.index else cal.iloc[0, 0]
            if 0 <= (ed.date() - datetime.now().date()).days <= 3: return True
    except: pass
    return False

# --- [3. 메인 분석 로직] ---
def analyze():
    start_date = (pd.Timestamp.now() - pd.DateOffset(years=3)).strftime('%Y-%m-%d')
    print(f"🚀 스캔 시작: {datetime.now()}")

    mkt = yf.download(["SPY", "^VIX"], start=start_date, progress=False)['Close']
    spy, vix = mkt['SPY'].dropna(), mkt['^VIX'].dropna()
    if spy.iloc[-1] <= ta.sma(spy, 200).iloc[-1] or vix.iloc[-1] > 25:
        send_telegram("⚠️ <b>시장 필터 작동 (매수 중단)</b>")
        return

    try:
        sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]['Symbol'].tolist()
        ndx100 = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")[4]['Ticker'].tolist()
        tickers = list(set(sp500 + ndx100))
    except:
        tickers = pd.read_csv("https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv")['Symbol'].tolist()
    
    tickers = [t.replace('.', '-') for t in tickers]
    raw = yf.download(tickers, start=start_date, group_by='ticker', threads=True, progress=False)

    rs_scores = {}
    for t in tickers:
        try:
            df = raw[t].dropna() if isinstance(raw.columns, pd.MultiIndex) else raw.dropna()
            if len(df) < 260: continue
            df['MA50'], df['MA200'] = ta.sma(df['Close'], 50), ta.sma(df['Close'], 200)
            if df['Close'].iloc[-1] > df['MA50'].iloc[-1] > df['MA200'].iloc[-1] and df['MA50'].diff(5).iloc[-1] > 0:
                score = calc_rs_score(df, spy)
                if score > 0: rs_scores[t] = score
        except: continue

    # RS 점수 기준 내림차순 정렬 (가장 쎈 놈부터 순차적으로 검사하기 위함)
    leaders_sorted = pd.Series(rs_scores).sort_values(ascending=False)
    leaders = leaders_sorted.head(int(len(leaders_sorted) * 0.2)).index # 상위 20%

    msg_list = []
    sector_counts = {} # 🛡️ 섹터별 카운팅을 저장할 딕셔너리

    for t in leaders:
        try:
            df = raw[t].dropna()
            df['MA20'], df['MA50'] = ta.sma(df['Close'], 20), ta.sma(df['Close'], 50)
            df['BB_MID'] = ta.bbands(df['Close'], 20)['BBM_20_2.0']
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], 14)
            df['avg_v20'], df['prev_v'] = ta.sma(df['Volume'], 20), df['Volume'].shift(1)

            cp, atr = float(df['Close'].iloc[-1]), float(df['ATR'].iloc[-1])
            recent_high = float(df['High'].rolling(20).max().iloc[-1])
            
            pullback_pct = (recent_high - cp) / recent_high
            pullback_dist = recent_high - cp

            if pullback_pct < 0.03 or pullback_pct > 0.12: continue
            if pullback_dist < atr * 1.0 or pullback_dist > atr * 6.0: continue

            is_vol_ok = ((df['Volume'] > df['prev_v']) | ((df['prev_v'] > df['avg_v20'] * 1.5) & (df['Volume'] > df['avg_v20']))) & (df['Volume'] < df['avg_v20'] * 3)
            rev_pos = (cp - df['Low'].iloc[-1]) / (df['High'].iloc[-1] - df['Low'].iloc[-1]) if (df['High'].iloc[-1] - df['Low'].iloc[-1]) > 0 else 0
            
            df['Sync_Signal'] = (df['MA20'] > df['MA50']) & (df['Close'] <= df['BB_MID']) & (df['Close'] > df['Open']) & (rev_pos >= 0.6) & is_vol_ok
            
            # 모든 차트 셋업이 완료되었을 때만 섹터 및 실적 검사 진행 (API 속도 최적화)
            if df['Sync_Signal'].iloc[-1] and not is_earnings_near(t):
                
                # --- [섹터 집중 방지 로직] ---
                try:
                    sector = yf.Ticker(t).info.get('sector', 'Unknown')
                except:
                    sector = 'Unknown'

                if sector != 'Unknown' and sector_counts.get(sector, 0) >= MAX_PER_SECTOR:
                    print(f"⏭️ {t} 스킵 (섹터 집중 방지: {sector} 이미 {MAX_PER_SECTOR}개 확보)")
                    continue
                # -----------------------------

                opt_mult, max_gap, min_rev = get_optimal_metrics(df)
                if (cp - df['Low'].iloc[-1]) / atr >= min_rev:
                    
                    # 최종 합격 시 섹터 카운트 1 증가
                    sector_counts[sector] = sector_counts.get(sector, 0) + 1
                    
                    stop = cp - (opt_mult * atr)
                    msg_list.append(
                        f"🚀 <b>[실전 주문] {t}</b> <code>[{sector}]</code>\n"
                        f"🎯 지정가 매수: ${cp*(1+max_gap/100):.2f}\n"
                        f"🛑 스탑로스: ${stop:.2f}\n"
                        f"📦 수량: {int(RISK_AMOUNT // (cp - stop))}주\n\n"
                    )
        except Exception as e: 
            continue

    send_telegram_chunks(msg_list, f"<b>📅 {datetime.now().date()} 퀀트 보고서 (PRO-MASTER)</b>\n\n", f"\n<b>[진단 완료] 최종 타점: {len(msg_list)}개</b>")

if __name__ == "__main__":
    analyze()
