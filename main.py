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

# 불필요한 경고문구 숨김 처리
warnings.filterwarnings('ignore')

# --- [1. 자산 및 리스크 설정] ---
BOT_TOKEN = os.environ.get('TG_TOKEN')
CHAT_ID = os.environ.get('TG_CHAT_ID')

def send_telegram(message):
    """텔레그램 메시지 발송 함수"""
    if not BOT_TOKEN or not CHAT_ID: 
        print("텔레그램 토큰 또는 CHAT_ID가 설정되지 않았습니다.")
        print(message)
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, data=data, timeout=10)
    except Exception as e:
        print(f"텔레그램 발송 실패: {e}")

def get_optimal_metrics(df):
    """최적 ATR 배수(손절), 허용 갭, 종목별 최소 반등 강도 산출 (터틀 2.0 ATR 플로어 적용)"""
    mae_list = []
    historical_gaps = []
    reversal_strengths = []
    
    signals = df[df['Buy_Signal_Historical']].index
    
    for idx in signals:
        loc = df.index.get_loc(idx)
        if loc + 1 >= len(df): continue
        
        close_p = float(df.iloc[loc]['Close'])
        low_p = float(df.iloc[loc]['Low'])
        atr_p = float(df.iloc[loc]['ATR'])
        next_open_p = float(df.iloc[loc+1]['Open'])
        
        # 1. 갭 데이터 수집
        gap_pct = ((next_open_p - close_p) / close_p) * 100
        historical_gaps.append(gap_pct)

        # 2. MAE(최대 역행) 및 성공 반등 강도 수집
        if loc + 10 >= len(df): continue
        future_low = float(df.iloc[loc+1 : loc+11]['Low'].min())
        future_max = float(df.iloc[loc+1 : loc+11]['High'].max())
        
        # 손절폭 데이터 축적
        if (close_p - future_low) > 0 and atr_p > 0:
            mae_list.append((close_p - future_low) / atr_p)
            
        # 성공 사례(10일 내 수익 구간 발생) 시, 반등 첫날의 강도 측정
        if future_max > close_p and atr_p > 0:
            reversal_strengths.append((close_p - low_p) / atr_p)
    
    # 데이터 부족 시 방어
    if len(mae_list) < 10 or len(reversal_strengths) < 5:
        return None, None, None
        
    # [터틀 트레이딩 가드레일] 데이터 산출값과 2.0 ATR 중 큰 값 선택 (휩소 완벽 방어)
    opt_mult = max(np.percentile(mae_list, 90), 2.0) 
    
    max_gap_threshold = np.percentile(historical_gaps, 80)
    min_reversal_factor = np.percentile(reversal_strengths, 25) 
    
    return opt_mult, max_gap_threshold, min_reversal_factor

def calc_rs_score(df, spy_df):
    """가중 누적 수익률을 활용한 개별 종목의 RS(상대강도) 점수 계산"""
    try:
        periods = [63, 126, 189, 252]
        weights = [0.4, 0.2, 0.2, 0.2] # 최근 3개월 탄력에 40% 가중치
        score = 0
        
        for p, w in zip(periods, weights):
            if len(df) > p and len(spy_df) > p:
                stock_ret = float(df['Close'].iloc[-1]) / float(df['Close'].iloc[-p])
                spy_ret = float(spy_df['Close'].iloc[-1]) / float(spy_df['Close'].iloc[-p])
                relative_ret = stock_ret / spy_ret 
                score += relative_ret * w
        return score
    except:
        return 0

def fetch_wiki_tickers_safe(url):
    """위키피디아 티커 수집 (S&P 500, Nasdaq 100)"""
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
    """우회 루트 티커 수집"""
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
    print(f"🚀 퀀트 스캔 시작: {datetime.now()}")
    
    # --- 🛑 [시장 3중 필터] SPY 추세 & VIX 변동성 체크 ---
    print("시장 상태(SPY/VIX) 확인 중...")
    spy_df = yf.download("SPY", start=start_date, progress=False)
    vix_df = yf.download("^VIX", period="1mo", progress=False)
    
    if spy_df.empty or vix_df.empty:
        send_telegram("⚠️ <b>시장 데이터 로드 실패</b>\n지수 데이터를 가져오지 못해 스캔을 중단합니다.")
        return
        
    if isinstance(spy_df.columns, pd.MultiIndex): spy_df.columns = spy_df.columns.get_level_values(0)
    if isinstance(vix_df.columns, pd.MultiIndex): vix_df.columns = vix_df.columns.get_level_values(0)
    
    spy_df['MA200'] = ta.sma(spy_df['Close'], 200)
    spy_df['MA5'] = ta.sma(spy_df['Close'], 5)
    
    if len(spy_df) < 200: return
    
    spy_curr = float(spy_df['Close'].iloc[-1])
    spy_ma200 = float(spy_df['MA200'].iloc[-1])
    spy_ma5 = float(spy_df['MA5'].iloc[-1])
    vix_curr = float(vix_df['Close'].iloc[-1])
    
    # [방어 로직] 200일선 위 (대세 상승) AND 5일선 위 (단기 추세) AND 공포지수 25 미만 (패닉장 회피)
    market_is_safe = (spy_curr > spy_ma200) and (spy_curr > spy_ma5) and (vix_curr < 25)

    if not market_is_safe:
        send_telegram(f"⚠️ <b>시장 필터 작동 (현금 보호)</b>\nS&P 500 추세 이탈 또는 VIX 지수({vix_curr:.2f}) 급등으로 매수 스캔을 전면 중단합니다.")
        return

    # --- 1. 유니버스 구성 ---
    print("종목 유니버스 구성 중...")
    tickers = []
    for attempt in range(1, 4):
        sp500 = fetch_wiki_tickers_safe('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        nasdaq100 = fetch_wiki_tickers_safe('https://en.wikipedia.org/wiki/Nasdaq-100')
        tickers = list(set(sp500 + nasdaq100))
        if len(tickers) > 400: break
        time.sleep(5)
        
    if len(tickers) < 400: tickers = list(set(fetch_fallback_tickers()))
    tickers = [t.replace('.', '-') for t in tickers]

    if len(tickers) < 100:
        send_telegram("⚠️ <b>데이터 수집 실패</b>\n명단 확보에 실패했습니다.")
        return

    total_scan = len(tickers)
    step1_pass, step2_pass, step3_pass, rs_pass, final_pass = 0, 0, 0, 0, 0
    candidates_data, rs_scores = {}, {}

    # --- 🚀 [속도 최적화] 대량 일괄 다운로드 ---
    print(f"총 {total_scan}개 종목 일괄 다운로드 중... (약 1~2분 소요)")
    raw_data = yf.download(tickers, start=start_date, group_by='ticker', threads=True, progress=False)

    for ticker in tickers:
        try:
            if isinstance(raw_data.columns, pd.MultiIndex):
                if ticker not in raw_data.columns.get_level_values(0): continue
                df = raw_data[ticker].copy()
            else: df = raw_data.copy()
            
            df.dropna(inplace=True)
            if df.empty or len(df) < 260: continue

            cp = float(df['Close'].iloc[-1])
            cv = float(df['Volume'].iloc[-1])
            avg_v20 = float(df['Volume'].rolling(20).mean().iloc[-1])

            # [1단계] 유동성 필터 (가격 상한선 300불 제한 철폐, 동전주만 컷)
            if cp < 10 or (cp * avg_v20 < 20000000): continue
            step1_pass += 1

            # 기술적 지표
            df['MA20'], df['MA50'], df['MA200'] = ta.sma(df['Close'], 20), ta.sma(df['Close'], 50), ta.sma(df['Close'], 200)
            df['BB_MID'] = ta.bbands(df['Close'], 20, 2.0)['BBM_20_2.0']
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], 14)
            rsi_val = float(ta.rsi(df['Close'], 14).iloc[-1])
            current_atr = float(df['ATR'].iloc[-1])

            # 🛑 [2단계] 종목 장/중기 추세 필터 (주도주는 200일, 50일선 위에 존재)
            if cp < float(df['MA200'].iloc[-1]) or cp < float(df['MA50'].iloc[-1]): continue
            step2_pass += 1

            # [3단계] 수급 진공 & 찐반등(망치형) 트리거
            # 3-1. 수급 조건: 어제보다 거래량이 늘었으나, 비정상적 광기(평균 3배 초과)는 아닐 것 & RSI 30 이상
            is_volume_ok = (cv > float(df['Volume'].iloc[-2])) and (cv < avg_v20 * 3.0)
            if not is_volume_ok or rsi_val < 30: continue
            
            # 3-2. 구역(Zone) 조건
            is_uptrend = float(df['MA20'].iloc[-1]) > float(df['MA50'].iloc[-1])
            is_in_pullback = cp <= float(df['BB_MID'].iloc[-1])
            
            # 3-3. 반등 트리거: 양봉 & 꼬리 말아올림(해머형 캔들 등 종가가 상단 40% 이내 안착)
            candle_range = float(df['High'].iloc[-1]) - float(df['Low'].iloc[-1])
            reversal_pos = (cp - float(df['Low'].iloc[-1])) / candle_range if candle_range > 0 else 0
            
            is_green_candle = cp > float(df['Open'].iloc[-1])
            is_strong_close = reversal_pos >= 0.6 # 종가가 고가에 가깝게 강하게 마감되었는지 확인
            
            # 백테스트 기록용 (Zone 진입 기준)
            df['Buy_Signal_Historical'] = (df['MA20'] > df['MA50']) & (df['Close'] <= df['BB_MID'])
            
            if is_uptrend and is_in_pullback and is_green_candle and is_strong_close:
                step3_pass += 1
                score = calc_rs_score(df, spy_df)
                if score > 0:
                    rs_scores[ticker] = score
                    candidates_data[ticker] = {'df': df, 'cp': cp, 'atr': current_atr, 'low': float(df['Low'].iloc[-1])}
        except Exception as e:
            continue

    # --- 🎯 [4단계] RS 랭킹 산출 및 데이터 검증 ---
    msg_list = []
    if rs_scores:
        rs_series = pd.Series(rs_scores)
        rs_ranks = rs_series.rank(pct=True) * 100 
        
        for ticker, rank in rs_ranks.items():
            if rank >= 80: # 상위 20% 주도주
                rs_pass += 1
                data = candidates_data[ticker]
                df, cp, atr, low = data['df'], data['cp'], data['atr'], data['low']
                
                opt_mult, max_gap_limit, min_rev_factor = get_optimal_metrics(df)
                if opt_mult is None: continue
                
                # 종목별 맞춤형 반등 강도 확인
                current_rev_strength = (cp - low) / atr
                
                if current_rev_strength >= min_rev_factor:
                    final_pass += 1
                    cnt_total = int(df['Buy_Signal_Historical'].sum())

                    # 리스크 산출 (고정 $200 리스크 기준)
                    stop_l = cp - (opt_mult * atr)
                    qty = int(200 // (cp - stop_l)) if cp > stop_l else 0
                    
                    entry_limit_p = cp * (1 + max_gap_limit / 100)
                    limit_stop_l = entry_limit_p - (opt_mult * atr)

                    msg_list.append(
                        f"🚀 <b>[매수 포착] {ticker}</b> (RS Rank: <b>{rank:.1f}</b>)\n"
                        f"- 과거기회 : 총 {cnt_total}회 (23년~)\n"
                        f"- ATR : <b>${atr:.2f}</b>\n"
                        f"\n"
                        f"- 현재가 : ${cp:.2f}\n"
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
    body = "\n".join(msg_list) if final_pass > 0 else "❌ <b>오늘은 '찐반등 트리거'가 작동한 1급(RS 80+) 주도주가 없습니다.</b>\n"
    
    footer = (f"\n<b>[진단 결과]</b>\n"
              f"* 총 스캔 종목: {total_scan}개\n"
              f"* 유동성 필터 통과: {step1_pass}개\n"
              f"* 추세(50/200일선) 통과: {step2_pass}개\n"
              f"* 타점(Zone/망치형 캔들/수급) 통과: {step3_pass}개\n"
              f"* RS 80+ 주도주 랭킹 통과: {rs_pass}개\n"
              f"* <b>최종(데이터 검증 & 찐반등 강도): {final_pass}개</b>")
    
    send_telegram(header + body + footer)

if __name__ == "__main__": 
    print("🚀 PRO 버전 퀀트 스캐너 가동을 시작합니다...")
    analyze()
    print("✅ 스캔 및 알림 프로세스가 정상 종료되었습니다.")
