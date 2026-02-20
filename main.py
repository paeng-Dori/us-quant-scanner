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
    
    # [퀀트 방패] 과거 기회가 10번 미만이면 데이터 부족으로 판단
    if len(mae_list) < 10:
        return None
    return np.percentile(mae_list, 90)

# 1차 메인 수집 루트 (위키피디아)
def fetch_wiki_tickers_safe(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36'}
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

# 2차 우회 수집 루트 (GitHub Public CSV 및 Slickcharts)
def fetch_fallback_tickers():
    tickers = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36'}
    try:
        print("⚠️ 위키피디아 수집 실패. 우회 루트(CSV/대체사이트)로 명단 수집을 시도합니다.")
        # S&P 500
        sp500_csv_url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        sp500_df = pd.read_csv(sp500_csv_url)
        if 'Symbol' in sp500_df.columns: tickers.extend(sp500_df['Symbol'].tolist())
        
        # Nasdaq 100
        res = requests.get('https://www.slickcharts.com/nasdaq100', headers=headers, timeout=10)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(res.text)
            tmp_path = f.name
        tables = pd.read_html(tmp_path)
        os.remove(tmp_path)
        for df in tables:
            if 'Symbol' in df.columns: tickers.extend(df['Symbol'].tolist())
    except Exception as e:
        print(f"우회 수집 실패: {e}")
    return tickers

def analyze():
    tickers = []
    max_retries = 3
    
    # 1. 유니버스 구성 (메인 루트)
    for attempt in range(1, max_retries + 1):
        sp500 = fetch_wiki_tickers_safe('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        nasdaq100 = fetch_wiki_tickers_safe('https://en.wikipedia.org/wiki/Nasdaq-100')
        tickers = list(set(sp500 + nasdaq100))
        if len(tickers) > 400: break
        print(f"⚠️ 위키피디아 {attempt}차 수집 실패...")
        time.sleep(5)
        
    # 메인 루트 3회 실패 시 우회 루트 가동
    if len(tickers) < 400:
        fallback_list = fetch_fallback_tickers()
        tickers = list(set(fallback_list))

    # 커스텀 라이징 스타 강제 추가
    custom_stars = ["RKLB", "LUNR", "PLTR", "MSTR", "IONQ", "SMCI", "SOFI", "ASTS", "U"]
    tickers = list(set(tickers + custom_stars))
    tickers = [t.replace('.', '-') for t in tickers]

    # 최종 명단 검수 (100개 미만이면 심각한 에러로 판단하여 중단)
    if len(tickers) < 100:
        send_telegram("⚠️ <b>데이터 수집 최종 실패</b>\n메인/우회 루트 모두 명단 확보에 실패했습니다.")
        return

    # 기존 포맷 출력을 위한 카운터 변수 복구
    total_scan = len(tickers)
    step1_pass, step2_pass, final_pass = 0, 0, 0
    msg_list = []
    
    start_date = "2023-01-01"

    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, progress=False)
            if df.empty or len(df) < 60: continue
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

            curr_price = float(df['Close'].iloc[-1])
            curr_vol = float(df['Volume'].iloc[-1])
            avg_vol_20 = float(df['Volume'].rolling(20).mean().iloc[-1])
            turnover = curr_price * avg_vol_20

            # 1. 가격 및 거래대금 통과 확인
            if not (10 <= curr_price <= 300) or turnover < 20000000: continue
            step1_pass += 1

            df['MA20'], df['MA50'] = ta.sma(df['Close'], 20), ta.sma(df['Close'], 50)
            adx_df = ta.adx(df['High'], df['Low'], df['Close'], 14)
            df['ADX'], df['PDI'], df['MDI'] = adx_df['ADX_14'], adx_df['DMP_14'], adx_df['DMN_14']
            df['BB_MID'] = ta.bbands(df['Close'], 20, 2.0)['BBM_20_2.0']
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], 14)
            rsi_val = ta.rsi(df['Close'], 14).iloc[-1]

            # 2. RSI 및 거래량 급감 통과 확인
            if curr_vol >= (avg_vol_20 * 0.8) or rsi_val <= 35: continue
            step2_pass += 1

            c1 = df['MA20'].iloc[-1] > df['MA50'].iloc[-1]
            c2 = (df['ADX'].iloc[-1] >= 20) and (df['ADX'].iloc[-1] >= df['ADX'].iloc[-2]) and (df['PDI'].iloc[-1] > df['MDI'].iloc[-1])
            c3 = (df['Close'].iloc[-1] <= df['BB_MID'].iloc[-1])
            
            df['Buy_Signal_Historical'] = (df['MA20'] > df['MA50']) & (df['ADX'] >= 20) & (df['PDI'] > df['MDI']) & (df['Close'] <= df['BB_MID'])

            # 3. 최종 매수 조건 통과 확인
            if c1 and c2 and c3:
                final_pass += 1
                opt_mult = get_optimal_atr_mult(df)
                cnt_total = int(df.loc[start_date:, 'Buy_Signal_Historical'].sum())

                if opt_mult is None:
                    stop_text = "<b>추천 불가</b> (과거 신호 10회 미만)"
                    qty_text = "<b>계산 불가</b> (손절가 미확정)"
                else:
                    stop_l = curr_price - (opt_mult * df['ATR'].iloc[-1])
                    stop_text = f"<b>${stop_l:.2f}</b> (ATR x {opt_mult:.2f}배)"
                    qty = int(200 // (curr_price - stop_l)) if curr_price > stop_l else 0
                    qty_text = f"<b>{qty}주</b>"

                msg_list.append(
                    f"🚀 <b>[매수 포착] {ticker}</b>\n"
                    f"- 현재가 : ${curr_price:.2f}\n"
                    f"- 과거기회 : 총 {cnt_total}회 (23년~현재)\n"
                    f"- 최적 손절가 : {stop_text}\n"
                    f"- 추천수량 : {qty_text}\n"
                )
        except: continue

    header = f"<b>📅 {datetime.now().date()} 퀀트 스캔 보고서</b>\n\n"
    body = "\n".join(msg_list) if final_pass > 0 else "❌ <b>오늘은 조건에 맞는 눌림목 종목이 없습니다.</b>\n"
    
    # 요청하신 기존 4줄 포맷으로 완벽 복구
    footer = (f"\n<b>[진단 결과]</b>\n"
              f"* 총 스캔 종목: {total_scan}개\n"
              f"* 가격/유동성 통과: {step1_pass}개\n"
              f"* RSI/거래량 급감 통과: {step2_pass}개\n"
              f"* 최종 매수 조건 통과: {final_pass}개")
    
    send_telegram(header + body + footer)

if __name__ == "__main__": analyze()
