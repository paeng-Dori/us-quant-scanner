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
    """메시지 길이 초과 방지를 위한 분할 발송 (3개 단위로 축소하여 가독성 확보)"""
    if not msg_list:
        send_telegram(header + "❌ <b>오늘은 조건에 맞는 1급 주도주가 없습니다.</b>\n" + footer)
        return
        
    chunk_size = 3 
    for i in range(0, len(msg_list), chunk_size):
        chunk = msg_list[i:i + chunk_size]
        body = "\n".join(chunk)
        title = f"{header} (파트 {i//chunk_size + 1})\n\n"
        send_telegram(title + body + (footer if i + chunk_size >= len(msg_list) else ""))
        time.sleep(1) # API 도배 방지 딜레이

# --- [2. 핵심 퀀트 엔진: 매수 & 매도] ---
def get_optimal_metrics(df):
    """과거 시그널을 바탕으로 매수 방어선(ATR 배수) 및 갭 한도 도출"""
    mae_list = []
    historical_gaps = []
    reversal_strengths = []
    
    df['avg_v20'] = ta.sma(df['Volume'], 20)
    df['prev_v'] = df['Volume'].shift(1)
    
    cond_increase = df['Volume'] > df['prev_v']
    cond_exception = (df['prev_v'] > df['avg_v20'] * 1.5) & (df['Volume'] > df['avg_v20'])
    df['is_vol_ok'] = (cond_increase | cond_exception) & (df['Volume'] < df['avg_v20'] * 3.0)
    
    df['is_green'] = df['Close'] > df['Open']
    df['c_range'] = df['High'] - df['Low']
    df['rev_pos'] = np.where(df['c_range'] > 0, (df['Close'] - df['Low']) / df['c_range'], 0)
    
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
    
    if len(mae_list) < 10 or len(reversal_strengths) < 5: 
        return 2.0, 2.0, 0.5, True 
        
    raw_opt_mult = np.percentile(mae_list, 90)
    is_defense = raw_opt_mult <= 2.0 
    opt_mult = max(raw_opt_mult, 2.0) 
    
    max_gap_threshold = np.percentile(historical_gaps, 80)
    min_reversal_factor = np.percentile(reversal_strengths, 25) 
    
    return opt_mult, max_gap_threshold, min_reversal_factor, is_defense

def get_optimized_sell_params(df):
    """과거 매수 타점들을 바탕으로 최적의 매도 타겟(전고점)과 추세선(SMA) 도출"""
    if 'Sync_Signal' not in df.columns: return 20, 20
        
    signals = df[df['Sync_Signal']].index[:-1] # 오늘 발생한 신호는 미래 결과가 없으므로 제외
    if len(signals) < 3: return 20, 20 # 과거 타점이 부족하면 기본값 배정
        
    target_lookbacks = [10, 15, 20] # 전고점 탐색 기간 후보
    sma_periods = [10, 20]          # 이탈 기준 추세선 후보
    
    for s in sma_periods:
        if f'SMA_{s}' not in df.columns:
            df[f'SMA_{s}'] = ta.sma(df['Close'], s)
            
    best_pnl = -float('inf')
    best_lookback = 20
    best_sma = 20
    
    for l in target_lookbacks:
        df[f'Target_High_{l}'] = df['High'].rolling(window=l).max().shift(1)
        
    for l in target_lookbacks:
        for s in sma_periods:
            pnl = 0
            for idx in signals:
                loc = df.index.get_loc(idx)
                if loc + 1 >= len(df): continue 
                
                entry_p = float(df.iloc[loc]['Close'])
                atr_p = float(df.iloc[loc]['ATR'])
                initial_sl = entry_p - (atr_p * 2.0)
                
                target_p = float(df.iloc[loc][f'Target_High_{l}'])
                if pd.isna(target_p) or target_p <= entry_p: 
                    target_p = entry_p + (atr_p * 2.0)
                    
                qty = 200 // (entry_p - initial_sl) if (entry_p - initial_sl) > 0 else 10
                half_qty = qty // 2
                half_sold = False
                current_sl = initial_sl
                
                # 최대 40일 추적 (무한 루프 방지)
                for j in range(loc + 1, min(loc + 41, len(df))):
                    curr = df.iloc[j]
                    
                    # 1. 1차 익절 도달
                    if not half_sold and curr['High'] >= target_p:
                        pnl += (target_p - entry_p) * half_qty
                        half_sold = True
                        current_sl = entry_p # 무적 모드 발동 (본전 스탑 상향)
                        continue
                        
                    # 2. 추세 이탈 또는 스탑로스 터치
                    is_sma_broken = curr['Close'] < curr[f'SMA_{s}']
                    is_sl_hit = curr['Low'] <= current_sl
                    
                    if is_sma_broken or is_sl_hit:
                        exit_p = current_sl if is_sl_hit else float(curr['Close'])
                        remaining_qty = (qty - half_qty) if half_sold else qty
                        pnl += (exit_p - entry_p) * remaining_qty
                        break
                        
            # 최고 수익을 안겨준 파라미터 저장
            if pnl > best_pnl:
                best_pnl = pnl
                best_lookback = l
                best_sma = s
                
    return best_lookback, best_sma

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
    start_date = (pd.Timestamp.now() - pd.DateOffset(years=3)).strftime('%Y-%m-%d')
    
    print(f"🚀 스캔 시작: {datetime.now()} (데이터 수집 기준일: {start_date})")
    
    # 1. 시장 필터
    print("시장 상태(SPY/VIX) 확인 중...")
    try:
        m_data = yf.download(["SPY", "^VIX"], start=start_date, progress=False)['Close']
    except Exception as e:
        print(f"⚠️ 시장 데이터 다운로드 실패: {e}")
        return
        
    if m_data.empty or 'SPY' not in m_data or '^VIX' not in m_data:
        print("⚠️ 시장 데이터를 불러올 수 없습니다.")
        return
        
    spy = m_data['SPY'].dropna()
    vix = m_data['^VIX'].dropna()
    
    if len(spy) < 200 or len(vix) < 1: 
        print("⚠️ 지수 데이터 누락으로 스캔 중단.")
        return
    
    spy_ma200 = ta.sma(spy, 200)
    spy_ma5 = ta.sma(spy, 5)
    
    spy_curr = float(spy.iloc[-1])
    vix_curr = float(vix.iloc[-1])
    
    if not (spy_curr > float(spy_ma200.iloc[-1]) and spy_curr > float(spy_ma5.iloc[-1]) and vix_curr < 25):
        send_telegram(f"⚠️ <b>시장 필터 작동 (매수 중단)</b>\nS&P 500 역배열 또는 VIX({vix_curr:.2f}) 불안정.")
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
        send_telegram("⚠️ <b>데이터 수집 최종 실패</b>\n티커 명단 확보 실패.")
        return

    # 3. 데이터 일괄 다운로드
    print(f"총 {len(tickers)}개 종목 일괄 다운로드 중...")
    raw_data = yf.download(tickers, start=start_date, group_by='ticker', threads=True, progress=False)

    rs_scores_global = {}

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
        send_telegram("⚠️ <b>조건을 충족하는 종목이 없습니다.</b>")
        return

    rs_ranks = pd.Series(rs_scores_global).rank(pct=True) * 100
    leading_stocks = rs_ranks[rs_ranks >= 80].index.tolist()

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
            
            is_zone = float(df['MA20'].iloc[-1]) > float(df['MA50'].iloc[-1]) and cp <= float(df['BB_MID'].iloc[-1])
            
            cond_increase = cv > prev_v
            cond_exception = (prev_v > avg_v20 * 1.5) and (cv > avg_v20)
            is_vol_ok = (cond_increase or cond_exception) and (cv < avg_v20 * 3.0)
            
            c_range = float(df['High'].iloc[-1]) - float(df['Low'].iloc[-1])
            rev_pos = (cp - float(df['Low'].iloc[-1])) / c_range if c_range > 0 else 0
            is_trigger = cp > float(df['Open'].iloc[-1]) and rev_pos >= 0.6
            
            if is_zone and is_vol_ok and is_trigger:
                # 1. 매수 파라미터 최적화 (기존 로직)
                opt_mult, max_gap_limit, min_rev_factor, is_defense = get_optimal_metrics(df)
                if opt_mult is None: continue
                
                curr_rev_strength = (cp - float(df['Low'].iloc[-1])) / float(df['ATR'].iloc[-1])
                
                if curr_rev_strength >= min_rev_factor:
                    final_pass_count += 1
                    
                    # 2. 매도 파라미터 최적화 도출 (신규 추가)
                    best_lookback, best_sma = get_optimized_sell_params(df)
                    
                    stop_l = cp - (opt_mult * float(df['ATR'].iloc[-1]))
                    qty = int(200 // (cp - stop_l)) if cp > stop_l else 0
                    
                    entry_limit_p = cp * (1 + max_gap_limit / 100)
                    limit_stop_l = entry_limit_p - (opt_mult * float(df['ATR'].iloc[-1]))
                    
                    # 맞춤형 익절 타겟가 계산 (최근 N일 최고점)
                    target_price = float(df['High'].iloc[-best_lookback:].max())
                    if target_price <= entry_limit_p: 
                        target_price = entry_limit_p + ((entry_limit_p - limit_stop_l) * 1.5)

                    atr_label = "하한선 방어" if is_defense else "동적 계산"

                    msg_list.append(
                        f"🚀 <b>[실전 주문] {ticker}</b> (RS Rank: {rs_ranks[ticker]:.1f})\n"
                        f"━━━━━━━━━━━━━━━━━━\n"
                        f"<b>[진입 플랜]</b>\n"
                        f"🎯 <b>지정가 매수 : ${entry_limit_p:.2f}</b> (이하 체결)\n"
                        f"🛑 <b>초기 손절가 : ${limit_stop_l:.2f}</b>\n"
                        f"📦 <b>매수 수량 : {qty}주</b> (리스크 $200)\n"
                        f"🛡️ 방어 기준 : ATR {opt_mult:.2f}배 ({atr_label})\n"
                        f"💡 반등 강도 : {curr_rev_strength:.2f} (최소 {min_rev_factor:.2f})\n"
                        f"━━━━━━━━━━━━━━━━━━\n"
                        f"<b>[매도 작전 명령서]</b>\n"
                        f"💰 <b>1차 익절(50%) : ${target_price:.2f}</b> ({best_lookback}일 전고점)\n"
                        f"📈 <b>추세 청산(50%) : 종가 SMA {best_sma} 이탈 시 매도</b>\n"
                        f"💡 <i>(Tip: 1차 익절 도달 시 남은 수량 손절가를 진입가로 변경)</i>\n\n"
                    )
        except Exception:
            continue

    # 4. 분할 발송 로직 실행
    header = f"<b>📅 {datetime.now().date()} 퀀트 보고서 (완전체)</b>\n\n"
    footer = f"\n<b>[진단 결과]</b>\n스캔:{len(tickers)}개 / 주도주(RS 80+):{len(leading_stocks)}개 / 최종 타점:{final_pass_count}개"
    
    send_telegram_chunks(msg_list, header, footer)

if __name__ == "__main__":
    print("🚀 PRO-MASTER 버전 완전체 퀀트 스캐너 가동을 시작합니다...")
    analyze()
    print("✅ 스캔 및 알림 프로세스가 정상 종료되었습니다.")
