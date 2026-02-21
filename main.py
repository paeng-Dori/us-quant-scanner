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
    """텔레그램 단일 메시지 발송 함수"""
    if not BOT_TOKEN or not CHAT_ID: 
        print("⚠️ 텔레그램 토큰 또는 CHAT_ID가 설정되지 않았습니다.")
        print(message)
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, data=data, timeout=10)
    except Exception as e:
        print(f"텔레그램 발송 실패: {e}")

def send_telegram_chunks(msg_list, header, footer):
    """메시지 길이 초과 방지를 위한 분할 발송 (5개 단위)"""
    if not msg_list:
        send_telegram(header + "❌ <b>오늘은 조건에 맞는 1급 주도주가 없습니다.</b>\n" + footer)
        return
        
    chunk_size = 5
    for i in range(0, len(msg_list), chunk_size):
        chunk = msg_list[i:i + chunk_size]
        body = "\n".join(chunk)
        title = f"{header} (파트 {i//chunk_size + 1})\n\n"
        send_telegram(title + body + (footer if i + chunk_size >= len(msg_list) else ""))
        time.sleep(1) # API 도배 방지 딜레이

# --- [2. 핵심 퀀트 엔진] ---
def get_optimal_metrics(df):
    """과거 시그널을 실전 타점 로직(캔들+수급 트리거)과 100% 동기화하여 정밀도 향상"""
    mae_list = []
    historical_gaps = []
    reversal_strengths = []
    
    df['avg_v20'] = ta.sma(df['Volume'], 20)
    df['prev_v'] = df['Volume'].shift(1)
    
    # [수급 트리거 보완] 전일 패닉셀/대량거래(1.5배 초과) 기저효과 방어 로직 적용
    cond_increase = df['Volume'] > df['prev_v']
    cond_exception = (df['prev_v'] > df['avg_v20'] * 1.5) & (df['Volume'] > df['avg_v20'])
    df['is_vol_ok'] = (cond_increase | cond_exception) & (df['Volume'] < df['avg_v20'] * 3.0)
    
    df['is_green'] = df['Close'] > df['Open']
    df['c_range'] = df['High'] - df['Low']
    df['rev_pos'] = np.where(df['c_range'] > 0, (df['Close'] - df['Low']) / df['c_range'], 0)
    
    # 완벽하게 동기화된 과거 매수 시그널
    df['Sync_Signal'] = (df['MA20'] > df['MA50']) & \
                        (df['Close'] <= df['BB_MID']) & \
                        (df['is_green']) & \
                        (df['rev_pos'] >= 0.6) & \
                        (df['is_vol_ok']) 
    
    signals = df[df['Sync_Signal']].index
    
    for idx in signals:
        loc = df.index.get_loc(idx)
        if loc + 11 >= len(df): continue 
        
        close_p = float(df.iloc[loc]['Close'])
        atr_p = float(df.iloc[loc]['ATR'])
        low_p = float(df.iloc[loc]['Low'])
        next_open_p = float(df.iloc[loc+1]['Open'])
        
        historical_gaps.append(((next_open_p - close_p) / close_p) * 100)

        f_low = float(df.iloc[loc+1 : loc+11]['Low'].min())
        f_max = float(df.iloc[loc+1 : loc+11]['High'].max())
        
        if (close_p - f_low) > 0 and atr_p > 0: 
            mae_list.append((close_p - f_low) / atr_p)
        if f_max > close_p and atr_p > 0: 
            reversal_strengths.append((close_p - low_p) / atr_p)
    
    # 데이터가 부족한 '슈퍼 스톡'을 버리지 않고 기본값 부여 (강력한 주도주 보호)
    if len(mae_list) < 10 or len(reversal_strengths) < 5: 
        return 2.0, 2.0, 0.5 
        
    opt_mult = max(np.percentile(mae_list, 90), 2.0) 
    max_gap_threshold = np.percentile(historical_gaps, 80)
    min_reversal_factor = np.percentile(reversal_strengths, 25) 
    
    return opt_mult, max_gap_threshold, min_reversal_factor

def calc_rs_score(df, spy_df):
    """가중 누적 수익률을 활용한 상대강도(RS) 점수 산출"""
    try:
        periods = [63, 126, 189, 252]
        weights = [0.4, 0.2, 0.2, 0.2]
        score = 0
        for p, w in zip(periods, weights):
            if len(df) > p and len(spy_df) > p:
                stock_ret = float(df['Close'].iloc[-1]) / float(df['Close'].iloc[-p])
                spy_ret = float(spy_df['Close'].iloc[-1]) / float(spy_df['Close'].iloc[-p])
                score += (stock_ret / spy_ret) * w
        return score
    except: 
        return 0

# --- [3. 유니버스 데이터 수집 함수] ---
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

# --- [4. 메인 분석 로직] ---
def analyze():
    # 항상 실행 시점 기준 '최근 3년 치' 동적 다운로드 (속도 및 유지보수 최적화)
    start_date = (pd.Timestamp.now() - pd.DateOffset(years=3)).strftime('%Y-%m-%d')
    
    print(f"🚀 스캔 시작: {datetime.now()} (데이터 수집 기준일: {start_date})")
    
    # 1. 시장 필터 (SPY & VIX)
    print("시장 상태(SPY/VIX) 확인 중...")
    m_data = yf.download(["SPY", "^VIX"], start=start_date, progress=False)['Close']
    if m_data.empty:
        send_telegram("⚠️ <b>시장 데이터 로드 실패</b>\n지수 데이터를 가져오지 못해 스캔을 중단합니다.")
        return
        
    spy = m_data['SPY'].dropna()
    vix = m_data['^VIX'].dropna()
    
    if len(spy) < 200: return
    
    spy_ma200 = ta.sma(spy, 200)
    spy_ma5 = ta.sma(spy, 5)
    
    spy_curr = float(spy.iloc[-1])
    vix_curr = float(vix.iloc[-1])
    
    if not (spy_curr > float(spy_ma200.iloc[-1]) and spy_curr > float(spy_ma5.iloc[-1]) and vix_curr < 25):
        send_telegram(f"⚠️ <b>시장 필터 작동 (매수 중단)</b>\nS&P 500 추세 이탈 또는 VIX 지수({vix_curr:.2f}) 불안정으로 현금을 보호합니다.")
        return

    # 2. 티커 수집
    print("유니버스 구성 중...")
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
        send_telegram("⚠️ <b>데이터 수집 최종 실패</b>\n티커 명단 확보에 실패했습니다.")
        return

    # 3. 데이터 일괄 다운로드
    print(f"총 {len(tickers)}개 종목 일괄 다운로드 중...")
    raw_data = yf.download(tickers, start=start_date, group_by='ticker', threads=True, progress=False)

    rs_scores_global = {}

    # [1차 패스] 전체 유니버스 대상 RS 랭킹 산출
    print("1차 패스: 전체 유니버스 RS 점수 계산 중...")
    for ticker in tickers:
        try:
            if isinstance(raw_data.columns, pd.MultiIndex):
                if ticker not in raw_data.columns.get_level_values(0): continue
                df = raw_data[ticker].copy()
            else:
                df = raw_data.copy()
                
            df.dropna(inplace=True)
            if len(df) < 260: continue
            
            cp = float(df['Close'].iloc[-1])
            avg_v20 = float(df['Volume'].rolling(20).mean().iloc[-1])
            
            if cp < 10 or (cp * avg_v20 < 20000000): continue
            
            df['MA200'] = ta.sma(df['Close'], 200)
            df['MA50'] = ta.sma(df['Close'], 50)
            
            if cp > float(df['MA200'].iloc[-1]) and cp > float(df['MA50'].iloc[-1]):
                score = calc_rs_score(df, spy)
                if score > 0:
                    rs_scores_global[ticker] = score
        except Exception:
            continue

    if not rs_scores_global:
        send_telegram("⚠️ <b>조건을 충족하는 종목이 없어 스캔을 종료합니다.</b>")
        return

    # RS 점수를 바탕으로 상위 20% 주도주 명단 추출
    rs_ranks = pd.Series(rs_scores_global).rank(pct=True) * 100
    leading_stocks = rs_ranks[rs_ranks >= 80].index.tolist()

    # [2차 패스] 상위 20% 주도주 안에서 타점 검사
    print(f"2차 패스: 상위 20% 주도주({len(leading_stocks)}개) 타점 스캔 중...")
    msg_list = []
    final_pass_count = 0

    for ticker in leading_stocks:
        try:
            df = raw_data[ticker].dropna()
            df['MA20'] = ta.sma(df['Close'], 20)
            df['MA50'] = ta.sma(df['Close'], 50)
            df['BB_MID'] = ta.bbands(df['Close'], 20, 2.0)['BBM_20_2.0']
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], 14)
            
            cp = float(df['Close'].iloc[-1])
            cv = float(df['Volume'].iloc[-1])
            prev_v = float(df['Volume'].iloc[-2])
            avg_v20 = float(df['Volume'].rolling(20).mean().iloc[-1])
            
            # 1. 구역(Zone)
            is_zone = float(df['MA20'].iloc[-1]) > float(df['MA50'].iloc[-1]) and cp <= float(df['BB_MID'].iloc[-1])
            
            # 2. 수급(Volume) 트리거 보완 - 전일 1.5배 초과시 오늘 평균 이상만 되어도 패스
            cond_increase = cv > prev_v
            cond_exception = (prev_v > avg_v20 * 1.5) and (cv > avg_v20)
            is_vol_ok = (cond_increase or cond_exception) and (cv < avg_v20 * 3.0)
            
            # 3. 캔들(Hammer) 트리거
            c_range = float(df['High'].iloc[-1]) - float(df['Low'].iloc[-1])
            rev_pos = (cp - float(df['Low'].iloc[-1])) / c_range if c_range > 0 else 0
            is_trigger = cp > float(df['Open'].iloc[-1]) and rev_pos >= 0.6
            
            if is_zone and is_vol_ok and is_trigger:
                opt_mult, max_gap_limit, min_rev_factor = get_optimal_metrics(df)
                if opt_mult is None: continue
                
                curr_rev_strength = (cp - float(df['Low'].iloc[-1])) / float(df['ATR'].iloc[-1])
                
                if curr_rev_strength >= min_rev_factor:
                    final_pass_count += 1
                    
                    stop_l = cp - (opt_mult * float(df['ATR'].iloc[-1]))
                    qty = int(200 // (cp - stop_l)) if cp > stop_l else 0
                    
                    entry_limit_p = cp * (1 + max_gap_limit / 100)
                    limit_stop_l = entry_limit_p - (opt_mult * float(df['ATR'].iloc[-1]))

                    msg_list.append(
                        f"🚀 <b>[매수 포착] {ticker}</b> (RS Rank: <b>{rs_ranks[ticker]:.1f}</b>)\n"
                        f"- ATR : <b>${float(df['ATR'].iloc[-1]):.2f}</b>\n"
                        f"\n"
                        f"- 현재가 : ${cp:.2f}\n"
                        f"- <b>진입 제한가 : ${entry_limit_p:.2f} (갭 {max_gap_limit:.1f}% 이내)</b>\n"
                        f"\n"
                        f"- 현재가 진입시 손절가 : ${stop_l:.2f} (ATR x {opt_mult:.2f}배)\n"
                        f"- 제한가 진입시 손절가 : <b>${limit_stop_l:.2f}</b>\n"
                        f"\n"
                        f"- 추천수량 : <b>{qty}주</b>\n"
                        f"💡 <i>반등강도: {curr_rev_strength:.2f} (최소기준 {min_rev_factor:.2f} 통과)</i>\n"
                    )
        except Exception:
            continue

    # 4. 분할 발송 로직 실행
    header = f"<b>📅 {datetime.now().date()} 퀀트 보고서 (PRO-MASTER)</b>\n\n"
    footer = f"\n<b>[진단 결과]</b>\n스캔:{len(tickers)}개 / 주도주(RS 80+):{len(leading_stocks)}개 / 최종 타점:{final_pass_count}개"
    
    send_telegram_chunks(msg_list, header, footer)

if __name__ == "__main__":
    print("🚀 PRO-MASTER 버전 퀀트 스캐너 가동을 시작합니다...")
    analyze()
    print("✅ 스캔 및 알림 프로세스가 정상 종료되었습니다.")
