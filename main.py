import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import requests
import os
from datetime import datetime

# --- [1. 자산 및 리스크 설정] ---
BOT_TOKEN = os.environ.get('TG_TOKEN')
CHAT_ID = os.environ.get('TG_CHAT_ID')
TOTAL_CAPITAL = 10000
RISK_PER_TRADE = 0.02 # 2% 리스크 ($200)

def send_telegram(message):
    if not BOT_TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    requests.post(url, data=data)

def get_optimal_atr_mult(df):
    mae_list = []
    signals = df[df['Buy_Signal_Historical']].index
    for idx in signals:
        loc = df.index.get_loc(idx)
        if loc + 10 >= len(df): continue
        entry_p = df.iloc[loc]['Close']
        entry_atr = df.iloc[loc]['ATR']
        if entry_atr <= 0: continue
        future_low = df.iloc[loc+1 : loc+11]['Low'].min()
        drawdown = entry_p - future_low
        if drawdown > 0: mae_list.append(drawdown / entry_atr)
    return np.percentile(mae_list, 90) if mae_list else 2.5

def analyze():
    # 지수 종목 리스트 수집
    try:
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
        nasdaq100 = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')[0]['Symbol'].tolist()
        tickers = list(set(sp500 + nasdaq100))
        tickers = [t.replace('.', '-') for t in tickers]
    except:
        tickers = ["NVDA", "AAPL", "MSFT", "TSLA"]

    msg_list = [f"<b>📅 {datetime.now().date()} 퀀트 스캔 보고서</b>\n(기준: S&P500/나스닥100 우량주)\n"]
    found = 0

    for ticker in tickers:
        try:
            df = yf.download(ticker, start="2024-01-01", progress=False)
            if len(df) < 60: continue
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

            # 지표 계산
            curr_price = df['Close'].iloc[-1]
            curr_vol = df['Volume'].iloc[-1]
            avg_vol_20 = df['Volume'].rolling(20).mean().iloc[-1]
            turnover = curr_price * avg_vol_20
            
            df['MA20'] = ta.sma(df['Close'], 20)
            df['MA50'] = ta.sma(df['Close'], 50)
            adx_df = ta.adx(df['High'], df['Low'], df['Close'], 14)
            df['ADX'], df['PDI'], df['MDI'] = adx_df['ADX_14'], adx_df['DMP_14'], adx_df['DMN_14']
            bb = ta.bbands(df['Close'], 20, 2.0)
            df['BB_MID'], df['BB_LOW'] = bb['BBM_20_2.0'], bb['BBL_20_2.0']
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], 14)
            df['RSI'] = ta.rsi(df['Close'], 14)

            # --- [필터링 및 매수 조건] ---
            # 1. 가격 & 유동성 필터 ($10~$300 & $20M↑)
            if not (10 <= curr_price <= 300) or turnover < 20000000: continue
            # 2. 거래량 급감 (0.7배) & RSI (35↑)
            if curr_vol >= (avg_vol_20 * 0.7) or df['RSI'].iloc[-1] <= 35: continue

            # 3. 기술적 조건
            c1 = df['MA20'] > df['MA50']
            c2 = (df['ADX'] >= 20) & (df['ADX'] >= df['ADX'].shift(1)) & (df['PDI'] > df['MDI'])
            c3 = (df['Close'] <= df['BB_MID']) & (df['Close'] > df['BB_LOW'])
            
            df['Buy_Signal_Historical'] = c1 & c2 & c3

            if c1 and c2 and c3:
                found += 1
                opt_mult = get_optimal_atr_mult(df)
                stop_l = curr_price - (opt_mult * df['ATR'].iloc[-1])
                qty = int(200 // (curr_price - stop_l)) if curr_price > stop_l else 0
                
                cnt_24 = df.loc['2024-01-01':'2024-12-31', 'Buy_Signal_Historical'].sum()
                cnt_25 = df.loc['2025-01-01':, 'Buy_Signal_Historical'].sum()

                msg_list.append(
                    f"<b>★ {ticker}</b> (${curr_price:.2f})\n"
                    f"└ 과거기회: 24~25년(총 {int(cnt_24+cnt_25)}회)\n"
                    f"└ 최적손절: ATR x {opt_mult:.2f}배 (<b>${stop_l:.2f}</b>)\n"
                    f"└ <b>추천수량: {qty}주</b>\n"
                )
        except: continue

    if found > 0: send_telegram("\n".join(msg_list))
    else: send_telegram("<b>📅 {datetime.now().date()}</b>\n❄️ 오늘 조건에 맞는 우량 눌림목 종목이 없습니다.")

if __name__ == "__main__": analyze()
