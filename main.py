import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import requests
import os
import time
import tempfile
from datetime import datetime

# --- [1. 자산 및 리스크 설정] ---
BOT_TOKEN = os.environ.get('TG_TOKEN')
CHAT_ID = os.environ.get('TG_CHAT_ID')

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

# Pandas Errno 2 버그 원천 차단을 위한 안전 추출 함수
def fetch_wiki_tickers_safe(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36'}
    res = requests.get(url, headers=headers, timeout=15)
    
    # 가상의 물리적 임시 파일 생성
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
        f.write(res.text)
        tmp_path = f.name
        
    try:
        # 파일 경로를 직접 넘겨주어 경로 오인 버그 차단
        tables = pd.read_html(tmp_path)
        for df in tables:
            # 위키피디아 테이블 구조 변동에 대비해 Symbol과 Ticker 모두 검사
            if 'Symbol' in df.columns: return df['Symbol'].tolist()
            if 'Ticker' in df.columns: return df['Ticker'].tolist()
    finally:
        os.remove(tmp_path) # 사용 완료 후 임시 파일 삭제
    return []

def analyze():
    tickers = []
    max_retries = 3
    retry_delay = 10

    # 1. 종목 리스트 수집
    for attempt in range(1, max_retries + 1):
        try:
            print(f"🚀 종목 리스트 수집 시도 ({attempt}/{max_retries})...")
            
            sp500 = fetch_wiki_tickers_safe('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            nasdaq100 = fetch_wiki_tickers_safe('https://en.wikipedia.org/wiki/Nasdaq-100')
            
            tickers = list(set(sp500 + nasdaq100))
            tickers = [t.replace('.', '-') for t in tickers]
            
            if len(tickers) > 400:
                print(f"✅ {len(tickers)}개 종목 명단 확보 성공!")
                break
        except Exception as e:
            print(f"⚠️ {attempt}차 수집 실패 사유: {e}")
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                send_telegram(f"⚠️ <b>데이터 수집 최종 실패</b>\n위키피디아 접속 또는 파싱에 실패했습니다.\n(사유: {str(e)})")
                return

    total_scan = len(tickers)
    step1_pass, step2_pass, final_pass = 0, 0, 0
    msg_list = []
    
    for ticker in tickers:
        try:
            df = yf.download(ticker, start="2024-01-01", progress=False)
            if df.empty or len(df) < 60: continue
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

            curr_price = float(df['Close'].iloc[-1])
            curr_vol = float(df['Volume'].iloc[-1])
            avg_vol_20 = float(df['Volume'].rolling(20).mean().iloc[-1])
            turnover = curr_price * avg_vol_20
            
            if not (10 <= curr_price <= 300) or turnover < 20000000: continue
            step1_pass += 1
            
            df['MA20'], df['MA50'] = ta.sma(df['Close'], 20), ta.sma(df['Close'], 50)
            adx_df = ta.adx(df['High'], df['Low'], df['Close'], 14)
            df['ADX'], df['PDI'], df['MDI'] = adx_df['ADX_14'], adx_df['DMP_14'], adx_df['DMN_14']
            df['BB_MID'] = ta.bbands(df['Close'], 20, 2.0)['BBM_20_2.0']
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], 14)
            rsi_val = ta.rsi(df['Close'], 14).iloc[-1]

            if curr_vol >= (avg_vol_20 * 0.8) or rsi_val <= 35: continue
            step2_pass += 1

            c1 = df['MA20'].iloc[-1] > df['MA50'].iloc[-1]
            c2 = (df['ADX'].iloc[-1] >= 20) and (df['ADX'].iloc[-1] >= df['ADX'].iloc[-2]) and (df['PDI'].iloc[-1] > df['MDI'].iloc[-1])
            c3 = (df['Close'].iloc[-1] <= df['BB_MID'].iloc[-1])
            
            df['Buy_Signal_Historical'] = (df['MA20'] > df['MA50']) & (df['ADX'] >= 20) & (df['PDI'] > df['MDI']) & (df['Close'] <= df['BB_MID'])

            if c1 and c2 and c3:
                final_pass += 1
                opt_mult = get_optimal_atr_mult(df)
                stop_l = curr_price - (opt_mult * df['ATR'].iloc[-1])
                qty = int(200 // (curr_price - stop_l)) if curr_price > stop_l else 0
                cnt_total = int(df.loc['2024-01-01':, 'Buy_Signal_Historical'].sum())

                msg_list.append(
                    f"🚀 <b>[매수 포착] {ticker}</b>\n"
                    f"- 현재가 : ${curr_price:.2f}\n"
                    f"- 과거기회 : 총 {cnt_total}회 (24~25년)\n"
                    f"- 최적 손절가 : <b>${stop_l:.2f}</b> (ATR x {opt_mult:.2f}배)\n"
                    f"- 추천수량 : <b>{qty}주</b>\n"
                )
        except: continue

    header = f"<b>📅 {datetime.now().date()} 퀀트 스캔 보고서</b>\n\n"
    body = "\n".join(msg_list) if final_pass > 0 else "❌ <b>오늘은 조건에 맞는 눌림목 종목이 없습니다.</b>\n"
    footer = (f"\n<b>[진단 결과]</b>\n"
              f"* 총 스캔 종목: {total_scan}개\n"
              f"* 가격/유동성 통과: {step1_pass}개\n"
              f"* RSI/거래량 급감 통과: {step2_pass}개\n"
              f"* 최종 매수 조건 통과: {final_pass}개")
    
    send_telegram(header + body + footer)

if __name__ == "__main__": analyze()
