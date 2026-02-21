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

def send_telegram(message):
    if not BOT_TOKEN or not CHAT_ID: 
        print("⚠️ 텔레그램 설정 누락:\n", message)
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, data=data, timeout=10)
    except Exception as e:
        print(f"텔레그램 발송 실패: {e}")

def send_telegram_chunks(msg_list, header, footer):
    if not msg_list:
        send_telegram(header + "❌ <b>오늘은 조건에 맞는 1급 주도주가 없습니다.</b>\n" + footer)
        return
    chunk_size = 5
    for i in range(0, len(msg_list), chunk_size):
        chunk = msg_list[i:i + chunk_size]
        body = "\n".join(chunk)
        title = f"{header} (파트 {i//chunk_size + 1})\n\n"
        send_telegram(title + body + (footer if i + chunk_size >= len(msg_list) else ""))
        time.sleep(1)

# --- [2. 핵심 퀀트 엔진] ---
def get_optimal_metrics(df):
    mae_list, historical_gaps, reversal_strengths = [], [], []

    signals = df[df['Sync_Signal']].index
    
    for idx in signals[:-1]: # 오늘 발생한 신호는 미래 결과가 없으므로 제외
        loc = df.index.get_loc(idx)
        if loc + 11 >= len(df): continue

        cp = float(df.iloc[loc]['Close'])
        atr = float(df.iloc[loc]['ATR'])
        low = float(df.iloc[loc]['Low'])
        next_open = float(df.iloc[loc+1]['Open'])

        historical_gaps.append((next_open - cp) / cp * 100)

        f_low = float(df.iloc[loc+1:loc+11]['Low'].min())
        f_high = float(df.iloc[loc+1:loc+11]['High'].max())

        if atr > 0:
            if (cp - f_low) > 0: mae_list.append((cp - f_low) / atr)
            if f_high > cp: reversal_strengths.append((cp - low) / atr)

    if len(mae_list) < 10:
        return 2.0, 2.0, 0.5

    return (
        max(np.percentile(mae_list, 90), 2.0),
        np.percentile(historical_gaps, 80),
        np.percentile(reversal_strengths, 25)
    )

def calc_rs_score(df, spy):
    """기준일(Index)을 맞춰서 RS 점수 계산 오류 방지"""
    score = 0
    periods = [63, 126, 189, 252]
    weights = [0.4, 0.2, 0.2, 0.2]
    for p, w in zip(periods, weights):
        if len(df) > p and len(spy) > p:
            stock_ret = float(df['Close'].iloc[-1]) / float(df['Close'].iloc[-p])
            spy_ret = float(spy.iloc[-1]) / float(spy.iloc[-p])
            score += (stock_ret / spy_ret) * w
    return score

def is_earnings_near(ticker_symbol):
    """yfinance 에러를 방지하는 안전한 실적발표일 체크 (3일 이내)"""
    try:
        cal = yf.Ticker(ticker_symbol).calendar
        if cal is not None and not cal.empty:
            ed = cal.loc['Earnings Date'].values[0] if 'Earnings Date' in cal.index else cal.iloc[0, 0]
            if isinstance(ed, (datetime, pd.Timestamp)):
                if 0 <= (ed.date() - datetime.now().date()).days <= 3:
                    return True
    except:
        pass
    return False

# --- [3. 유니버스 수집 (S&P 500 + NASDAQ 100 복구)] ---
def get_universe_tickers():
    tickers = []
    try:
        sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]['Symbol'].tolist()
        ndx100 = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")[4]['Ticker'].tolist()
        tickers = list(set(sp500 + ndx100))
    except:
        tickers = pd.read_csv("https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv")['Symbol'].tolist()
    return [t.replace('.', '-') for t in tickers]

# --- [4. 메인 분석 로직] ---
def analyze():
    start_date = (pd.Timestamp.now() - pd.DateOffset(years=3)).strftime('%Y-%m-%d')
    print(f"🚀 스캔 시작: {datetime.now()}")

    mkt = yf.download(["SPY", "^VIX"], start=start_date, progress=False)['Close']
    spy, vix = mkt['SPY'].dropna(), mkt['^VIX'].dropna()

    if spy.iloc[-1] <= ta.sma(spy, 200).iloc[-1] or vix.iloc[-1] > 25:
        send_telegram("⚠️ <b>시장 필터 작동 (매수 중단)</b>\n지수 역배열 또는 VIX 불안정.")
        return

    tickers = get_universe_tickers()
    raw = yf.download(tickers, start=start_date, group_by='ticker', threads=True, progress=False)

    rs_scores = {}
    print("1차 패스: 유니버스 추세 및 RS 랭킹 산출 중...")
    for t in tickers:
        try:
            df = raw[t].dropna() if isinstance(raw.columns, pd.MultiIndex) else raw.dropna()
            if len(df) < 260: continue

            df['MA50'] = ta.sma(df['Close'], 50)
            df['MA200'] = ta.sma(df['Close'], 200)
            df['MA50_slope'] = df['MA50'].diff(5) # 회원님 아이디어: 50일선 기울기

            # 완벽한 정배열 및 50일선 우상향 조건
            if df['Close'].iloc[-1] > df['MA50'].iloc[-1] > df['MA200'].iloc[-1] and df['MA50_slope'].iloc[-1] > 0:
                score = calc_rs_score(df, spy)
                if score > 0: rs_scores[t] = score
        except: continue

    if not rs_scores: return
    leaders = pd.Series(rs_scores).rank(pct=True)
    leading_stocks = leaders[leaders >= 0.8].index.tolist()

    msg_list = []
    print(f"2차 패스: 상위 20% 주도주({len(leading_stocks)}개) 정밀 타점 스캔 중...")
    
    for t in leading_stocks:
        try:
            df = raw[t].dropna()
            df['MA20'] = ta.sma(df['Close'], 20)
            df['MA50'] = ta.sma(df['Close'], 50)
            df['BB_MID'] = ta.bbands(df['Close'], 20)['BBM_20_2.0']
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], 14)
            df['avg_v20'] = ta.sma(df['Volume'], 20)
            df['prev_v'] = df['Volume'].shift(1)

            cp = float(df['Close'].iloc[-1])
            recent_high = float(df['High'].rolling(20).max().iloc[-1])
            pullback = (recent_high - cp) / recent_high

            # [새로운 엣지] 1. 눌림목 깊이 필터 (3~12%)
            if pullback < 0.03 or pullback > 0.12: continue

            # [새로운 엣지] 2. 과열(설거지) 방지 필터 (최근 5일내 12% 이상 폭등 이력 제외)
            if float(df['Close'].pct_change().rolling(5).max().iloc[-1]) > 0.12: continue

            # [복구된 엣지] 3. 오늘 자 매수 방아쇠(Trigger) 검증 로직
            cond_increase = df['Volume'] > df['prev_v']
            cond_exception = (df['prev_v'] > df['avg_v20'] * 1.5) & (df['Volume'] > df['avg_v20'])
            df['is_vol_ok'] = (cond_increase | cond_exception) & (df['Volume'] < df['avg_v20'] * 3)
            
            df['c_range'] = df['High'] - df['Low']
            df['rev_pos'] = np.where(df['c_range'] > 0, (df['Close'] - df['Low']) / df['c_range'], 0)
            df['is_green'] = df['Close'] > df['Open']
            
            # Sync_Signal 백테스트용 컬럼 생성 (과거 지표 도출용)
            df['Sync_Signal'] = (df['MA20'] > df['MA50']) & (df['Close'] <= df['BB_MID']) & df['is_green'] & (df['rev_pos'] >= 0.6) & df['is_vol_ok']
            
            # 오늘 캔들이 완벽한 매수 셋업을 만족했는가?
            if df['Sync_Signal'].iloc[-1]:
                
                # [새로운 엣지] 4. 실적발표 임박 종목 필터링
                if is_earnings_near(t): continue

                opt_mult, max_gap, min_rev = get_optimal_metrics(df)
                atr = float(df['ATR'].iloc[-1])
                
                # 반등 강도(밑꼬리)가 과거 하위 25% 기준치보다 약하면 패스
                curr_rev = (cp - float(df['Low'].iloc[-1])) / atr
                if curr_rev < min_rev: continue

                stop = cp - (opt_mult * atr)
                qty = int(200 // (cp - stop)) if cp > stop else 0

                msg_list.append(
                    f"🚀 <b>[실전 주문] {t}</b>\n"
                    f"🎯 지정가 매수: ${cp*(1+max_gap/100):.2f} (이하)\n"
                    f"🛑 스탑로스: ${stop:.2f}\n"
                    f"📦 수량: {qty}주\n\n"
                )
        except Exception:
            continue

    send_telegram_chunks(
        msg_list,
        f"<b>📅 {datetime.now().date()} 퀀트 보고서 (PRO-MASTER)</b>\n\n",
        f"\n<b>[진단 완료] 최종 타점: {len(msg_list)}개</b>"
    )

if __name__ == "__main__":
    print("🚀 PRO-MASTER 통합 매수 스캐너 가동 중...")
    analyze()
    print("✅ 스캔 완료.")
