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

def get_optimal_metrics(df):
    """최적 ATR 배수(손절)와 최대 허용 갭 임계치(진입 제한)를 산출"""
    mae_list = []
    historical_gaps = []
    
    signals = df[df['Buy_Signal_Historical']].index
    
    for idx in signals:
        loc = df.index.get_loc(idx)
        # 1. 갭 데이터 수집 (다음 날 시가 기준)
        if loc + 1 >= len(df): continue
        close_p = df.iloc[loc]['Close']
        next_open_p = df.iloc[loc+1]['Open']
        gap_pct = ((next_open_p - close_p) / close_p) * 100
        historical_gaps.append(gap_pct)

        # 2. MAE 데이터 수집 (최대 역행 폭)
        if loc + 10 >= len(df): continue
        entry_atr = df.iloc[loc]['ATR']
        future_low = df.iloc[loc+1 : loc+11]['Low'].min()
        drawdown = close_p - future_low
        if drawdown > 0 and entry_atr > 0:
            mae_list.append(drawdown / entry_atr)
    
    # [퀀트 방패] 샘플 10개 미만 시 데이터 부족으로 판단
    if len(mae_list) < 10:
        return None, None
        
    # 손절용 ATR 배수(상위 90% 생존) 및 진입 제한용 갭(상위 80% 허용)
    opt_mult = np.percentile(mae_list, 90)
    max_gap_threshold = np.percentile(historical_gaps, 80)
    
    return opt_mult, max_gap_threshold

def calc_rs_score(df, spy_df):
    """가중 누적 수익률을 활용한 개별 종목의 RS 점수(절대값) 계산"""
    try:
        # 기준일(분기 단위: 대략 63, 126, 189, 252 거래일)
        periods = [63, 126, 189, 252]
        weights = [0.4, 0.2, 0.2, 0.2] # 최근 성과에 가장 높은 가중치 40%
        score = 0
        
        for p, w in zip(periods, weights):
            if len(df) > p and len(spy_df) > p:
                stock_ret = df['Close'].iloc[-1] / df['Close'].iloc[-p]
                spy_ret = spy_df['Close'].iloc[-1] / spy_df['Close'].iloc[-p]
                relative_ret = stock_ret / spy_ret 
                score += relative_ret * w
        return score
    except:
        return 0

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

# 2차 우회 수집 루트
def fetch_fallback_tickers():
    tickers = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36'}
    try:
        print("⚠️ 위키피디아 수집 실패. 우회 루트(CSV/대체사이트)로 명단 수집을 시도합니다.")
        sp500_csv_url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        sp500_df = pd.read_csv(sp500_csv_url)
        if 'Symbol' in sp500_df.columns: tickers.extend(sp500_df['Symbol'].tolist())
        
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
    
    # 1. 유니버스 구성
    for attempt in range(1, max_retries + 1):
        sp500 = fetch_wiki_tickers_safe('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        nasdaq100 = fetch_wiki_tickers_safe('https://en.wikipedia.org/wiki/Nasdaq-100')
        tickers = list(set(sp500 + nasdaq100))
        if len(tickers) > 400: break
        print(f"⚠️ 위키피디아 {attempt}차 수집 실패...")
        time.sleep(5)
        
    if len(tickers) < 400:
        tickers = list(set(fetch_fallback_tickers()))

    tickers = [t.replace('.', '-') for t in tickers]

    if len(tickers) < 100:
        send_telegram("⚠️ <b>데이터 수집 최종 실패</b>\n메인/우회 루트 모두 명단 확보에 실패했습니다.")
        return

    # 벤치마크(SPY) 데이터 사전 로드
    start_date = "2023-01-01"
    print("벤치마크(SPY) 데이터 다운로드 중...")
    spy_df = yf.download("SPY", start=start_date, progress=False)
    if isinstance(spy_df.columns, pd.MultiIndex): spy_df.columns = spy_df.columns.get_level_values(0)

    total_scan = len(tickers)
    step1_pass, step2_pass, rs_pass, final_pass = 0, 0, 0, 0
    
    candidates_data = {} 
    rs_scores = {}

    print("종목 스캔 및 기술적/RS 지표 계산 중...")
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, progress=False)
            if df.empty or len(df) < 260: continue 
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

            curr_price = float(df['Close'].iloc[-1])
            curr_vol = float(df['Volume'].iloc[-1])
            avg_vol_20 = float(df['Volume'].rolling(20).mean().iloc[-1])
            turnover = curr_price * avg_vol_20

            # 1단계: 유동성 및 가격 필터
            if not (10 <= curr_price <= 300) or turnover < 20000000: continue
            step1_pass += 1

            df['MA20'], df['MA50'] = ta.sma(df['Close'], 20), ta.sma(df['Close'], 50)
            df['BB_MID'] = ta.bbands(df['Close'], 20, 2.0)['BBM_20_2.0']
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], 14)
            rsi_val = ta.rsi(df['Close'], 14).iloc[-1]
            current_atr = float(df['ATR'].iloc[-1])

            # 2단계: 수급 상태 체크 (RSI 및 거래량 급감)
            if curr_vol >= (avg_vol_20 * 0.8) or rsi_val <= 35: continue
            step2_pass += 1

            # 추세 및 타점 확인 (ADX, PDI, MDI 삭제됨)
            c1 = df['MA20'].iloc[-1] > df['MA50'].iloc[-1]
            c3 = (df['Close'].iloc[-1] <= df['BB_MID'].iloc[-1])
            
            df['Buy_Signal_Historical'] = (df['MA20'] > df['MA50']) & (df['Close'] <= df['BB_MID'])

            if c1 and c3:
                score = calc_rs_score(df, spy_df)
                if score > 0:
                    rs_scores[ticker] = score
                    candidates_data[ticker] = {
                        'df': df, 'curr_price': curr_price, 'current_atr': current_atr
                    }
        except: continue

    # RS 랭킹 산출 및 최종 필터링
    msg_list = []
    if rs_scores:
        rs_series = pd.Series(rs_scores)
        rs_ranks = rs_series.rank(pct=True) * 100 
        
        for ticker, rank in rs_ranks.items():
            if rank >= 80:
                rs_pass += 1
                data = candidates_data[ticker]
                df = data['df']
                curr_price = data['curr_price']
                current_atr = data['current_atr']
                
                opt_mult, max_gap_limit = get_optimal_metrics(df)
                if opt_mult is None: continue
                
                final_pass += 1
                cnt_total = int(df.loc[start_date:, 'Buy_Signal_Historical'].sum())

                stop_l = curr_price - (opt_mult * current_atr)
                qty = int(200 // (curr_price - stop_l)) if curr_price > stop_l else 0
                
                entry_limit_p = curr_price * (1 + max_gap_limit / 100)
                limit_stop_l = entry_limit_p - (opt_mult * current_atr)

                msg_list.append(
                    f"🚀 <b>[매수 포착] {ticker}</b> (RS Rank: <b>{rank:.1f}</b>)\n"
                    f"- 과거기회 : 총 {cnt_total}회 (23년~)\n"
                    f"- ATR : <b>${current_atr:.2f}</b>\n"
                    f"\n"
                    f"- 현재가 : ${curr_price:.2f}\n"
                    f"- <b>진입 제한가 : ${entry_limit_p:.2f} (갭 {max_gap_limit:.1f}% 이내)</b>\n"
                    f"\n"
                    f"- 현재가 진입시, 손절가 : ${stop_l:.2f} (ATR x {opt_mult:.2f}배)\n"
                    f"- 제한가 진입시, 손절가 : <b>${limit_stop_l:.2f}</b>\n"
                    f"\n"
                    f"- 추천수량 : <b>{qty}주</b>\n"
                )

    header = f"<b>📅 {datetime.now().date()} 퀀트 스캔 보고서</b>\n\n"
    body = "\n".join(msg_list) if final_pass > 0 else "❌ <b>오늘은 조건에 맞는 1급(RS 80+) 눌림목 종목이 없습니다.</b>\n"
    
    footer = (f"\n<b>[진단 결과]</b>\n"
              f"* 총 스캔 종목: {total_scan}개\n"
              f"* 가격/유동성 통과: {step1_pass}개\n"
              f"* <b>수급 상태 체크 통과: {step2_pass}개</b>\n"
              f"* RS 80+ 통과: {rs_pass}개\n"
              f"* 최종 매수(데이터 검증) 통과: {final_pass}개")
    
    send_telegram(header + body + footer)

if __name__ == "__main__": analyze()
