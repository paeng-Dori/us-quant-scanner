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

warnings.filterwarnings('ignore') # Pandas 경고 숨김

# --- [1. 자산 및 리스크 설정] ---
BOT_TOKEN = os.environ.get('TG_TOKEN')
CHAT_ID = os.environ.get('TG_CHAT_ID')

def send_telegram(message):
    if not BOT_TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    requests.post(url, data=data)

def get_optimal_metrics(df):
    """최적 ATR 배수(손절), 허용 갭, 그리고 '종목별 최소 반등 강도' 산출"""
    mae_list = []
    historical_gaps = []
    reversal_strengths = []
    
    signals = df[df['Buy_Signal_Historical']].index
    
    for idx in signals:
        loc = df.index.get_loc(idx)
        if loc + 1 >= len(df): continue
        
        close_p = df.iloc[loc]['Close']
        low_p = df.iloc[loc]['Low']
        atr_p = df.iloc[loc]['ATR']
        next_open_p = df.iloc[loc+1]['Open']
        
        # 1. 갭 데이터 수집
        gap_pct = ((next_open_p - close_p) / close_p) * 100
        historical_gaps.append(gap_pct)

        # 2. MAE(최대 역행) 및 성공한 반등 강도 수집
        if loc + 10 >= len(df): continue
        future_low = df.iloc[loc+1 : loc+11]['Low'].min()
        future_max = df.iloc[loc+1 : loc+11]['High'].max()
        
        drawdown = close_p - future_low
        if drawdown > 0 and atr_p > 0:
            mae_list.append(drawdown / atr_p)
            
        # [핵심] 10일 내에 진입가 이상 수익을 준 '성공 사례'일 때, 반등 첫날의 강도 측정
        if future_max > close_p and atr_p > 0:
            rev_strength = (close_p - low_p) / atr_p
            reversal_strengths.append(rev_strength)
    
    # 데이터 부족 시 탈락 (과거기회 10번, 성공반등 5번 이상 필수)
    if len(mae_list) < 10 or len(reversal_strengths) < 5:
        return None, None, None
        
    opt_mult = np.percentile(mae_list, 90)
    max_gap_threshold = np.percentile(historical_gaps, 80)
    # 이 종목이 반등에 성공할 때 보여준 최소한의 힘 (하위 25% 지점)
    min_reversal_factor = np.percentile(reversal_strengths, 25) 
    
    return opt_mult, max_gap_threshold, min_reversal_factor

def calc_rs_score(df, spy_df):
    """가중 누적 수익률을 활용한 개별 종목의 RS 점수 계산"""
    try:
        periods = [63, 126, 189, 252]
        weights = [0.4, 0.2, 0.2, 0.2] # 최근 3개월에 가장 높은 가중치(40%)
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

# 2차 우회 수집 루트
def fetch_fallback_tickers():
    tickers = []
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
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
    except: pass
    return tickers

def analyze():
    start_date = "2023-01-01"
    
    # --- 🛑 [시장 생존 2중 필터] S&P 500(SPY) 상태 확인 ---
    print("시장 상태(SPY) 확인 중...")
    spy_df = yf.download("SPY", start=start_date, progress=False)
    if isinstance(spy_df.columns, pd.MultiIndex): spy_df.columns = spy_df.columns.get_level_values(0)
    
    spy_df['MA200'] = ta.sma(spy_df['Close'], 200)
    spy_df['MA5'] = ta.sma(spy_df['Close'], 5)
    
    spy_curr_close = float(spy_df['Close'].iloc[-1])
    spy_prev_close = float(spy_df['Close'].iloc[-2])
    spy_change_pct = ((spy_curr_close / spy_prev_close) - 1) * 100
    
    # [방어 로직] 200일선 장기 추세 유지 & (5일선 단기 유지 OR 당일 폭락이 아닐 것)
    market_is_safe = (spy_curr_close > spy_df['MA200'].iloc[-1]) and \
                     (spy_curr_close > spy_df['MA5'].iloc[-1] or spy_change_pct > -1.5)

    if not market_is_safe:
        send_telegram(f"⚠️ <b>시장 필터 작동 (매수 중단)</b>\nS&P 500 지수가 장/단기 지지선을 이탈했거나 급락했습니다. 현금을 보호합니다.\n(변동률: {spy_change_pct:.2f}%)")
        return

    # --- 1. 유니버스 구성 ---
    tickers = []
    for attempt in range(1, 4):
        sp500 = fetch_wiki_tickers_safe('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        nasdaq100 = fetch_wiki_tickers_safe('https://en.wikipedia.org/wiki/Nasdaq-100')
        tickers = list(set(sp500 + nasdaq100))
        if len(tickers) > 400: break
        time.sleep(5)
        
    if len(tickers) < 400:
        tickers = list(set(fetch_fallback_tickers()))

    tickers = [t.replace('.', '-') for t in tickers]

    if len(tickers) < 100:
        send_telegram("⚠️ <b>데이터 수집 최종 실패</b>\n메인/우회 루트 모두 명단 확보에 실패했습니다.")
        return

    total_scan = len(tickers)
    step1_pass, step2_pass, step3_pass, rs_pass, final_pass = 0, 0, 0, 0, 0
    
    candidates_data = {} 
    rs_scores = {}

    # --- 🚀 [속도 최적화] 대량 일괄 다운로드 (약 1~2분 소요) ---
    print("종목 데이터 일괄 다운로드 중...")
    raw_data = yf.download(tickers, start=start_date, group_by='ticker', threads=True, progress=False)

    for ticker in tickers:
        try:
            # MultiIndex 데이터프레임에서 개별 종목 추출
            if isinstance(raw_data.columns, pd.MultiIndex):
                if ticker not in raw_data.columns.get_level_values(0): continue
                df = raw_data[ticker].copy()
            else:
                df = raw_data.copy()
            
            df.dropna(inplace=True)
            if df.empty or len(df) < 260: continue # 1년치 이상 데이터 필수

            curr_price = float(df['Close'].iloc[-1])
            curr_vol = float(df['Volume'].iloc[-1])
            avg_vol_20 = float(df['Volume'].rolling(20).mean().iloc[-1])
            turnover = curr_price * avg_vol_20

            # [1단계] 유동성 및 가격 필터
            if not (10 <= curr_price <= 300) or turnover < 20000000: continue
            step1_pass += 1

            df['MA20'] = ta.sma(df['Close'], 20)
            df['MA50'] = ta.sma(df['Close'], 50)
            df['MA200'] = ta.sma(df['Close'], 200)
            df['BB_MID'] = ta.bbands(df['Close'], 20, 2.0)['BBM_20_2.0']
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], 14)
            rsi_val = ta.rsi(df['Close'], 14).iloc[-1]
            current_atr = float(df['ATR'].iloc[-1])

            # 🛑 [2단계] 종목 200일선 방어 (개별 종목 지하실 회피)
            if df['Close'].iloc[-1] < df['MA200'].iloc[-1]: continue
            step2_pass += 1

            # [3단계] 수급 진공 & 구역(Zone) & 찐반등 트리거(Trigger) 결합
            if curr_vol >= (avg_vol_20 * 0.8) or rsi_val <= 35: continue
            
            is_uptrend = df['MA20'].iloc[-1] > df['MA50'].iloc[-1]
            is_in_pullback = df['Close'].iloc[-1] <= df['BB_MID'].iloc[-1]
            is_green_candle = df['Close'].iloc[-1] > df['Open'].iloc[-1] # 양봉
            is_low_held = df['Low'].iloc[-1] > df['Low'].iloc[-2] # 직전 저점 지지
            
            # 백테스트용 과거 시그널 (Zone 진입 기준)
            df['Buy_Signal_Historical'] = (df['MA20'] > df['MA50']) & (df['Close'] <= df['BB_MID'])
            
            # 구역 + 트리거 동시 만족 시에만 통과
            if is_uptrend and is_in_pullback and is_green_candle and is_low_held:
                step3_pass += 1
                # 조건을 통과한 후보군만 RS 점수 계산
                score = calc_rs_score(df, spy_df)
                if score > 0:
                    rs_scores[ticker] = score
                    candidates_data[ticker] = {
                        'df': df, 'curr_price': curr_price, 'current_atr': current_atr,
                        'curr_low': float(df['Low'].iloc[-1])
                    }
        except: continue

    # --- 🎯 [4단계] RS 랭킹 산출 및 '반등 트리거' 최종 검증 ---
    msg_list = []
    if rs_scores:
        rs_series = pd.Series(rs_scores)
        rs_ranks = rs_series.rank(pct=True) * 100 # 백분위 산출 (0~100)
        
        for ticker, rank in rs_ranks.items():
            # [조건 1] 상위 20% (Rank 80 이상) 주도주만 선별
            if rank >= 80:
                rs_pass += 1
                data = candidates_data[ticker]
                df = data['df']
                curr_price = data['curr_price']
                current_atr = data['current_atr']
                curr_low = data['curr_low']
                
                # 과거 데이터 분석을 통한 종목별 최적화 수치 산출
                opt_mult, max_gap_limit, min_rev_factor = get_optimal_metrics(df)
                if opt_mult is None: continue
                
                # ⚡ [조건 2] 종목별 맞춤형 반등 트리거 강도 확인
                # 오늘 저점 대비 종가의 상승폭이, 과거 성공했던 최소 반등폭보다 커야 함
                current_rev_strength = (curr_price - curr_low) / current_atr
                
                if current_rev_strength >= min_rev_factor:
                    final_pass += 1
                    cnt_total = int(df['Buy_Signal_Historical'].sum())

                    # 리스크 수치 계산
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
                        f"💡 <i>반등강도: {current_rev_strength:.2f} (최소기준 {min_rev_factor:.2f} 통과)</i>\n"
                    )

    # --- 텔레그램 발송 ---
    header = f"<b>📅 {datetime.now().date()} 퀀트 스캔 보고서</b>\n\n"
    body = "\n".join(msg_list) if final_pass > 0 else "❌ <b>오늘은 '반등 트리거'가 작동한 1급(RS 80+) 주도주가 없습니다.</b>\n"
    
    footer = (f"\n<b>[진단 결과]</b>\n"
              f"* 총 스캔 종목: {total_scan}개\n"
              f"* 가격/유동성 통과: {step1_pass}개\n"
              f"* 종목 200일선 방어 통과: {step2_pass}개\n"
              f"* 수급 진공/구역(Zone)/양봉 트리거 통과: {step3_pass}개\n"
              f"* RS 80+ 주도주 랭킹 통과: {rs_pass}개\n"
              f"* <b>최종(데이터 검증 & 찐반등 강도 확인): {final_pass}개</b>")
    
    send_telegram(header + body + footer)

if __name__ == "__main__": analyze()
