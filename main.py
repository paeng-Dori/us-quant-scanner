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
    try:
        header = {"User-Agent": "Mozilla/5.0"}
        sp500 = pd.read_html(requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', headers=header).text)[0]['Symbol'].tolist()
        nasdaq100 = pd.read_html(requests.get('https://en.wikipedia.org/wiki/Nasdaq-100', headers=header).text)[0]['Symbol'].tolist()
        tickers = list(set(sp500 + nasdaq100))
        tickers = [t.replace('.', '-') for t in tickers]
    except:
        tickers = ["NVDA", "AAPL", "MSFT", "TSLA", "AMD", "GOOGL", "META"]

    msg_list = [f"<b>📅 {datetime.now().date()} 퀀트 스캔 보고서</b>\n"]
    found = 0

    for ticker in tickers:
        try:
            df = yf.download(ticker, start="2024-01-01", progress=False)
            if df.empty or len(df) < 60: continue
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

            curr_price = float(df['Close'].iloc[-1])
            curr_vol = float(df['Volume'].iloc[-1])
            avg_vol_20 = float(df['Volume'].rolling(20).mean().iloc[-1])
            turnover = curr_price * avg_vol_20
            
            df['MA20'] = ta.sma(df['Close'], 20)
            df['MA50'] = ta.sma(df['Close'], 50)
            adx_df = ta.adx(df['High'], df['Low'], df['Close'], 14)
            df['ADX'], df['PDI'], df['MDI'] = adx_df['ADX_14'], adx_df['DMP_14'], adx_df['DMN_14']
            bb = ta.bbands(df['Close'], 20, 2.0)
            df['BB_MID'] = bb['BBM_20_2.0']
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], 14)
            rsi_val = ta.rsi(df['Close'], 14).iloc[-1]

            # 필터링 조건
            if not (10 <= curr_price <= 300) or turnover < 20000000: continue
            cond_vol = curr_vol < (avg_vol_20 * 0.8)
            cond_rsi = rsi_val > 35

            c1 = df['MA20'].iloc[-1] > df['MA50'].iloc[-1]
            c2 = (df['ADX'].iloc[-1] >= 20) and (df['ADX'].iloc[-1] >= df['ADX'].iloc[-2]) and (df['PDI'].iloc[-1] > df['MDI'].iloc[-1])
            c3 = (df['Close'].iloc[-1] <= df['BB_MID'].iloc[-1])
            
            df['Buy_Signal_Historical'] = (df['MA20'] > df['MA50']) & (df['ADX'] >= 20) & (df['PDI'] > df['MDI']) & (df['Close'] <= df['BB_MID'])

            if c1 and c2 and c3 and cond_vol and cond_rsi:
                found += 1
                opt_mult = get_optimal_atr_mult(df)
                stop_l = curr_price - (opt_mult * df['ATR'].iloc[-1])
                qty = int(200 // (curr_price - stop_l)) if curr_price > stop_l else 0
                cnt_total = int(df.loc['2024-01-01':, 'Buy_Signal_Historical'].sum())

                # --- [요청하신 매수 포착 양식 반영] ---
                msg_list.append(
                    f"🚀 <b>[매수 포착] {ticker}</b>\n"
                    f"- 현재가 : ${curr_price:.2f}\n"
                    f"- 과거기회 : 총 {cnt_total}회 (24~25년)\n"
                    f"- 최적 손절가 : <b>${stop_l:.2f}</b> (ATR x {opt_mult:.2f}배)\n"
                    f"- 추천수량 : <b>{qty}주</b>\n"
                )
        except: continue

    if found > 0: 
        send_telegram("\n".join(msg_list))
    else: 
        send_telegram("❌ <b>오늘은 조건에 맞는 눌림목 종목이 없습니다.</b>")

if __name__ == "__main__": analyze()
