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
from functools import lru_cache

# 불필요한 경고문구 숨김 처리
warnings.filterwarnings('ignore')

# --- [0. 계좌/리스크(운용 안정화용 가드레일)] ---
ACCOUNT_CAPITAL = 10000  # ✅ 사용자 계좌
RISK_AMOUNT = 200        # 1회 타점당 고정 리스크 ($200)

MAX_OPEN_POSITIONS = 5                       # 동시 보유 최대 포지션(스윙 권장 4~5)
MAX_TOTAL_RISK = int(ACCOUNT_CAPITAL * 0.08) # 총 리스크 상한(예: 8% = $800)
MAX_POSITION_DOLLARS = int(ACCOUNT_CAPITAL * 0.30)  # 종목당 최대 투자금(예: 30% = $3,000)

MAX_PER_SECTOR = 2        # 동일 섹터 최대 진입 허용 개수

# --- [1. 텔레그램 설정] ---
BOT_TOKEN = os.environ.get('TG_TOKEN')
CHAT_ID = os.environ.get('TG_CHAT_ID')

def send_telegram(message: str):
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
        time.sleep(1)  # API 도배 방지

# --- [2. yfinance 메타데이터 조회 안정화(캐시)] ---
@lru_cache(maxsize=4096)
def get_sector_cached(ticker_symbol: str) -> str:
    try:
        return yf.Ticker(ticker_symbol).info.get('sector', 'Unknown') or 'Unknown'
    except:
        return 'Unknown'

@lru_cache(maxsize=4096)
def is_earnings_near_cached(ticker_symbol: str) -> bool:
    """실적 발표일이 3일 이내인지(캐시 + 예외 방어). 데이터 없으면 False로 보고 진행."""
    try:
        cal = yf.Ticker(ticker_symbol).calendar
        if cal is None or getattr(cal, "empty", True):
            return False

        ed = None
        if hasattr(cal, "index") and ('Earnings Date' in cal.index):
            vals = cal.loc['Earnings Date'].values
            ed = vals[0] if len(vals) else None
        else:
            try:
                ed = cal.iloc[0, 0]
            except:
                ed = None

        if ed is None:
            return False

        if isinstance(ed, (list, tuple, np.ndarray)) and len(ed) > 0:
            ed = ed[0]

        if hasattr(ed, "date"):
            days = (ed.date() - datetime.now().date()).days
            return 0 <= days <= 3

    except:
        pass
    return False

# --- [3. 유니버스: 정적 CSV 우선 + 안전한 폴백] ---
def fetch_wiki_tickers_safe(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code != 200:
            return []
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(res.text)
            tmp_path = f.name
        tables = pd.read_html(tmp_path)
        os.remove(tmp_path)

        for df in tables:
            if 'Symbol' in df.columns:
                return df['Symbol'].astype(str).tolist()
            if 'Ticker' in df.columns:
                return df['Ticker'].astype(str).tolist()
    except:
        pass
    return []

def load_universe_from_static_csvs():
    """
    ✅ A: 유니버스 수집 안정화
    - 로컬 CSV가 있으면 최우선 사용
    - 없으면 GitHub 정적 CSV(raw)에서 로드 시도
    CSV 형식: 최소한 'Symbol' 컬럼을 포함
    """
    local_sp500 = "sp500.csv"
    local_nasdaq100 = "nasdaq100.csv"

    # 1) 로컬 우선
    tickers = []
    try:
        if os.path.exists(local_sp500):
            df = pd.read_csv(local_sp500)
            if 'Symbol' in df.columns:
                tickers += df['Symbol'].astype(str).tolist()
    except:
        pass

    try:
        if os.path.exists(local_nasdaq100):
            df = pd.read_csv(local_nasdaq100)
            if 'Symbol' in df.columns:
                tickers += df['Symbol'].astype(str).tolist()
    except:
        pass

    tickers = list({t.strip() for t in tickers if isinstance(t, str) and t.strip()})
    if len(tickers) >= 200:
        return tickers

    # 2) 원격 정적 CSV(예시): 원하는 저장소로 바꿔도 됨
    #    - S&P500은 datasets repo가 비교적 안정적
    #    - NASDAQ100은 별도 소스/자체 CSV를 권장(여기선 폴백만 둠)
    tickers = []
    try:
        sp500_csv_url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        sp500_df = pd.read_csv(sp500_csv_url)
        if 'Symbol' in sp500_df.columns:
            tickers += sp500_df['Symbol'].astype(str).tolist()
    except:
        pass

    # NASDAQ100은 “네가 관리하는 정적 CSV”를 추천.
    # 여기서는 위키 폴백에 의존하게끔 최소 구성.
    return list({t.strip() for t in tickers if isinstance(t, str) and t.strip()})

def build_universe():
    # 1) 정적 CSV 우선
    tickers = load_universe_from_static_csvs()

    # 2) NASDAQ100 보완(없으면 위키 폴백)
    if len(tickers) < 350:
        sp500 = fetch_wiki_tickers_safe('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        nasdaq100 = fetch_wiki_tickers_safe('https://en.wikipedia.org/wiki/Nasdaq-100')
        tickers = list(set(tickers + sp500 + nasdaq100))

    tickers = [t.replace('.', '-') for t in tickers if isinstance(t, str)]
    tickers = list({t.strip() for t in tickers if t.strip()})
    return tickers

# --- [4. yfinance 다운로드 안정화: 청크 + 재시도] ---
def _ensure_multiindex(one_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """단일 티커 다운로드 시 컬럼이 단층이면, (ticker, field) 멀티인덱스로 맞춤."""
    if isinstance(one_df.columns, pd.MultiIndex):
        return one_df
    # one_df: columns = ['Open','High',...]
    one_df = one_df.copy()
    one_df.columns = pd.MultiIndex.from_product([[ticker], one_df.columns])
    return one_df

def download_price_data_chunked(tickers, start_date, chunk_size=100, pause=1.2, retries=2, auto_adjust=True):
    """
    ✅ B: 대량 다운로드 누락 방지
    - 100개 단위 청크 다운로드
    - 청크 실패/누락 티커 재시도
    """
    all_parts = []
    tickers = list(tickers)

    def download_chunk(chunk):
        # group_by='ticker'로 받아 멀티인덱스 구조 유지
        df = yf.download(
            chunk,
            start=start_date,
            group_by='ticker',
            threads=True,
            progress=False,
            auto_adjust=auto_adjust
        )
        # 단일 티커면 멀티인덱스가 아닐 수 있어 보정
        if len(chunk) == 1:
            df = _ensure_multiindex(df, chunk[0])
        return df

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]

        got = None
        for attempt in range(retries + 1):
            try:
                got = download_chunk(chunk)
                break
            except Exception as e:
                if attempt >= retries:
                    print(f"⚠️ 청크 다운로드 실패(최종): {chunk[:3]}... ({len(chunk)}개) / 에러: {e}")
                else:
                    time.sleep(pause * (attempt + 1))

        if got is None or got.empty:
            time.sleep(pause)
            continue

        # 누락 티커 검사(멀티인덱스 기준)
        present = set(got.columns.get_level_values(0)) if isinstance(got.columns, pd.MultiIndex) else set()
        missing = [t for t in chunk if t not in present]

        # 누락 티커 재시도(개별/소수 묶음)
        if missing:
            retry_ok = []
            for t in missing:
                one = None
                for attempt in range(retries + 1):
                    try:
                        one = download_chunk([t])
                        break
                    except:
                        time.sleep(pause * (attempt + 1))
                if one is not None and not one.empty:
                    all_parts.append(one)
                    retry_ok.append(t)

            if retry_ok:
                print(f"✅ 누락 복구: {len(retry_ok)}/{len(missing)}개")

        all_parts.append(got)
        time.sleep(pause)

    if not all_parts:
        return pd.DataFrame()

    merged = pd.concat(all_parts, axis=1)
    # 중복 컬럼(같은 티커가 여러 번 붙는 케이스) 정리: 마지막 값 유지
    merged = merged.loc[:, ~merged.columns.duplicated(keep='last')]
    return merged

# --- [5. 데이터 전처리: sanity check] ---
def sanitize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    ✅ C: 비정상 캔들/이상치 제거로 ATR/손절 왜곡 방어
    - High < Low 제거
    - 가격 <= 0 제거
    - 일일 변동폭(High-Low)/Close 과도(예: 80% 초과) 제거
    - 전일 대비 갭/급변(예: 60% 초과) 제거 (데이터 오류 방어 목적)
    """
    df = df.copy()
    need_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if any(c not in df.columns for c in need_cols):
        return df

    # 기본 형식 오류 제거
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Open', 'High', 'Low', 'Close'])
    df = df[(df['High'] >= df['Low'])]
    df = df[(df['Open'] > 0) & (df['High'] > 0) & (df['Low'] > 0) & (df['Close'] > 0)]

    # 일중 변동폭 과도 제거(데이터 오류/지연 반영 방어)
    intraday_range = (df['High'] - df['Low']) / df['Close']
    df = df[intraday_range <= 0.80]  # 80% 초과는 비정상치로 간주(원하면 조정)

    # 전일 대비 급변(데이터 오류 방어) - 너무 빡세면 완화 가능
    ret1 = df['Close'].pct_change().abs()
    df = df[(ret1.isna()) | (ret1 <= 0.60)]  # 60% 초과 급변 제거

    # Volume 음수/NaN 방어
    if 'Volume' in df.columns:
        df['Volume'] = df['Volume'].fillna(0)
        df = df[df['Volume'] >= 0]

    return df

# --- [6. 핵심 퀀트 엔진] ---
def get_optimal_metrics(df):
    """과거 시그널을 실전 타점 로직(캔들+수급 트리거)과 동기화하여 정밀도 향상"""
    mae_list = []
    historical_gaps = []
    reversal_strengths = []

    df = df.copy()

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
        if loc + 11 >= len(df):
            continue

        close_p = float(df.iloc[loc]['Close'])
        atr_p = float(df.iloc[loc]['ATR'])
        low_p = float(df.iloc[loc]['Low'])
        next_open_p = float(df.iloc[loc + 1]['Open'])

        if close_p <= 0 or atr_p <= 0:
            continue

        historical_gaps.append(((next_open_p - close_p) / close_p) * 100)

        f_low = float(df.iloc[loc + 1: loc + 11]['Low'].min())
        f_max = float(df.iloc[loc + 1: loc + 11]['High'].max())

        if (close_p - f_low) > 0:
            mae_list.append((close_p - f_low) / atr_p)
        if f_max > close_p:
            reversal_strengths.append((close_p - low_p) / atr_p)

    if len(mae_list) < 10 or len(reversal_strengths) < 5:
        return 2.0, 2.0, 0.5, True

    raw_opt_mult = float(np.percentile(mae_list, 90))
    is_defense = raw_opt_mult <= 2.0
    opt_mult = max(raw_opt_mult, 2.0)

    max_gap_threshold = float(np.percentile(historical_gaps, 80)) if historical_gaps else 2.0
    min_reversal_factor = float(np.percentile(reversal_strengths, 25))

    return opt_mult, max_gap_threshold, min_reversal_factor, is_defense

def calc_rs_score(df, spy_close):
    """가중 누적 수익률을 활용한 상대강도(RS) 점수 산출"""
    try:
        periods = [63, 126, 189, 252]
        weights = [0.4, 0.2, 0.2, 0.2]
        score = 0.0
        for p, w in zip(periods, weights):
            if len(df) > p and len(spy_close) > p:
                stock_ret = float(df['Close'].iloc[-1]) / float(df['Close'].iloc[-p])
                spy_ret = float(spy_close.iloc[-1]) / float(spy_close.iloc[-p])
                if spy_ret > 0:
                    score += (stock_ret / spy_ret) * w
        return float(score)
    except:
        return 0.0

# --- [7. 메인 분석 로직] ---
def analyze():
    start_date = (pd.Timestamp.now() - pd.DateOffset(years=3)).strftime('%Y-%m-%d')
    print(f"🚀 스캔 시작: {datetime.now()} (데이터 수집 기준일: {start_date})")

    # 1) 시장 데이터 (SPY / VIX) — ✅ auto_adjust로 왜곡 방어
    print("시장 상태(SPY/VIX) 확인 중...")
    try:
        m_raw = yf.download(["SPY", "^VIX"], start=start_date, progress=False, auto_adjust=True)
        m_close = m_raw['Close'] if isinstance(m_raw.columns, pd.MultiIndex) else m_raw['Close']
    except Exception as e:
        print(f"⚠️ 시장 데이터 다운로드 실패: {e}")
        return

    if m_close is None or len(m_close) == 0 or not isinstance(m_close, pd.DataFrame):
        print("⚠️ 시장 데이터를 불러올 수 없습니다.")
        return

    if 'SPY' not in m_close.columns or '^VIX' not in m_close.columns:
        print("⚠️ 시장 데이터 컬럼이 누락되었습니다.")
        return

    spy = m_close['SPY'].dropna()
    vix = m_close['^VIX'].dropna()

    if len(spy) < 200 or len(vix) < 1:
        print("⚠️ 지수 데이터가 부족하거나 누락되어 안전을 위해 스캔을 중단합니다.")
        return

    spy_ma200 = ta.sma(spy, 200)
    spy_ma5 = ta.sma(spy, 5)

    spy_curr = float(spy.iloc[-1])
    vix_curr = float(vix.iloc[-1])

    if not (spy_curr > float(spy_ma200.iloc[-1]) and spy_curr > float(spy_ma5.iloc[-1]) and vix_curr < 25):
        send_telegram(
            f"⚠️ <b>시장 필터 작동 (매수 중단)</b>\n"
            f"S&P 500 추세 이탈 또는 VIX 지수({vix_curr:.2f}) 불안정으로 현금을 보호합니다."
        )
        return

    # 2) 유니버스 구성 — ✅ 정적 CSV 우선
    print("유니버스 구성 중...")
    tickers = build_universe()

    if len(tickers) < 100:
        send_telegram("⚠️ <b>데이터 수집 최종 실패</b>\n티커 명단 확보에 실패했습니다.")
        return

    # 3) 종목 데이터 다운로드 — ✅ 청크 다운로드 + 재시도 + auto_adjust
    print(f"총 {len(tickers)}개 종목 청크 다운로드 중...")
    raw_data = download_price_data_chunked(
        tickers=tickers,
        start_date=start_date,
        chunk_size=100,
        pause=1.2,
        retries=2,
        auto_adjust=True
    )

    if raw_data is None or raw_data.empty:
        send_telegram("⚠️ <b>가격 데이터 다운로드 실패</b>\nraw_data가 비어있습니다.")
        return

    rs_scores_global = {}

    # [1차 패스] 전체 유니버스 대상 RS 랭킹 산출
    print("1차 패스: 전체 유니버스 RS 점수 계산 중...")
    for ticker in tickers:
        try:
            if not isinstance(raw_data.columns, pd.MultiIndex):
                continue
            if ticker not in raw_data.columns.get_level_values(0):
                continue

            df = raw_data[ticker].copy()
            df.dropna(inplace=True)

            # ✅ C: 전처리(이상치 제거)
            df = sanitize_ohlcv(df)

            if len(df) < 260:
                continue

            cp = float(df['Close'].iloc[-1])
            avg_v20 = float(df['Volume'].rolling(20).mean().iloc[-1])

            # 유동성 필터
            if cp < 10 or (cp * avg_v20 < 20000000):
                continue

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
    leading_stocks = rs_ranks[rs_ranks >= 80].sort_values(ascending=False).index.tolist()

    # [2차 패스] 상위 20% 주도주 안에서 타점 검사
    print(f"2차 패스: 상위 20% 주도주({len(leading_stocks)}개) 타점 스캔 중...")
    msg_list = []
    final_pass_count = 0
    sector_counts = {}
    total_risk_used = 0

    for ticker in leading_stocks:
        # 계좌 안정화: 포지션/총리스크 상한
        if final_pass_count >= MAX_OPEN_POSITIONS:
            break
        if total_risk_used + RISK_AMOUNT > MAX_TOTAL_RISK:
            break

        try:
            if ticker not in raw_data.columns.get_level_values(0):
                continue

            df = raw_data[ticker].dropna().copy()
            df = sanitize_ohlcv(df)
            if len(df) < 260:
                continue

            df['MA20'] = ta.sma(df['Close'], 20)
            df['MA50'] = ta.sma(df['Close'], 50)

            bb = ta.bbands(df['Close'], 20, 2.0)
            if bb is None or bb.empty or 'BBM_20_2.0' not in bb.columns:
                continue
            df['BB_MID'] = bb['BBM_20_2.0']

            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], 14)

            cp = float(df['Close'].iloc[-1])
            cv = float(df['Volume'].iloc[-1])
            prev_v = float(df['Volume'].iloc[-2])
            avg_v20 = float(df['Volume'].rolling(20).mean().iloc[-1])
            atr_val = float(df['ATR'].iloc[-1])

            if not np.isfinite(atr_val) or atr_val <= 0:
                continue

            # --- 단기 이격도(Pullback) 필터 ---
            recent_high = float(df['High'].rolling(20).max().iloc[-1])
            if not np.isfinite(recent_high) or recent_high <= 0:
                continue

            pullback_pct = (recent_high - cp) / recent_high
            pullback_dist = recent_high - cp

            if pullback_pct < 0.03 or pullback_pct > 0.12:
                continue
            if pullback_dist < atr_val * 1.0 or pullback_dist > atr_val * 6.0:
                continue
            # -----------------------------------

            # 1) 구역(Zone)
            is_zone = float(df['MA20'].iloc[-1]) > float(df['MA50'].iloc[-1]) and cp <= float(df['BB_MID'].iloc[-1])

            # 2) 수급(Volume) 트리거
            cond_increase = cv > prev_v
            cond_exception = (prev_v > avg_v20 * 1.5) and (cv > avg_v20)
            is_vol_ok = (cond_increase or cond_exception) and (cv < avg_v20 * 3.0)

            # 3) 캔들(Hammer) 트리거
            c_range = float(df['High'].iloc[-1]) - float(df['Low'].iloc[-1])
            rev_pos = (cp - float(df['Low'].iloc[-1])) / c_range if c_range > 0 else 0
            is_trigger = cp > float(df['Open'].iloc[-1]) and rev_pos >= 0.6

            if is_zone and is_vol_ok and is_trigger:
                # 실적 회피(캐시)
                if is_earnings_near_cached(ticker):
                    continue

                # 섹터 분산(캐시)
                sector = get_sector_cached(ticker)
                if sector != 'Unknown' and sector_counts.get(sector, 0) >= MAX_PER_SECTOR:
                    print(f"⏭️ {ticker} 스킵 (섹터 집중 방지: {sector} 이미 {MAX_PER_SECTOR}개 확보)")
                    continue

                opt_mult, max_gap_limit, min_rev_factor, is_defense = get_optimal_metrics(df)
                if opt_mult is None:
                    continue

                curr_rev_strength = (cp - float(df['Low'].iloc[-1])) / atr_val
                if curr_rev_strength < min_rev_factor:
                    continue

                # 손절/수량
                stop_l = cp - (opt_mult * atr_val)
                risk_per_share = cp - stop_l
                if risk_per_share <= 0:
                    continue

                qty = int(RISK_AMOUNT // risk_per_share)
                if qty < 1:
                    continue

                # 종목당 최대 투자금 제한
                max_qty_by_dollars = int(MAX_POSITION_DOLLARS // cp) if cp > 0 else 0
                if max_qty_by_dollars <= 0:
                    continue
                qty = min(qty, max_qty_by_dollars)
                if qty < 1:
                    continue

                # 진입 제한가(기존 로직 유지)
                entry_limit_p = cp * (1 + max_gap_limit / 100)
                limit_stop_l = entry_limit_p - (opt_mult * atr_val)

                atr_label = "하한선 방어" if is_defense else "동적 계산"
                sector_display = f"[{sector}]" if sector != "Unknown" else ""

                # 카운트 반영
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
                final_pass_count += 1
                total_risk_used += RISK_AMOUNT

                # ✅ 텔레그램 메시지 “완전 동일” 유지 (요청사항)
                msg_list.append(
                    f"🚀 <b>[실전 주문] {ticker}</b> {sector_display} (RS: 상위 {100-rs_ranks[ticker]:.1f}%)\n"
                    f"━━━━━━━━━━━━━━━━━━\n"
                    f"🎯 <b>지정가 매수 : ${entry_limit_p:.2f}</b> (이하 체결)\n"
                    f"🛑 <b>스탑로스(SL): ${limit_stop_l:.2f}</b>\n"
                    f"📦 <b>매수 수량 : {qty}주</b> (리스크 ${RISK_AMOUNT} 고정)\n"
                    f"━━━━━━━━━━━━━━━━━━\n"
                    f"📉 참조 종가 : ${cp:.2f} (눌림목 {pullback_pct*100:.1f}%)\n"
                    f"🛡️ 방어 기준 : ATR {opt_mult:.2f}배 적용 ({atr_label})\n"
                    f"💡 반등 강도 : {curr_rev_strength:.2f} (최소 {min_rev_factor:.2f})\n\n"
                )

        except Exception:
            continue

    # 분할 발송
    header = f"<b>📅 {datetime.now().date()} 퀀트 보고서 (PRO-MASTER)</b>\n\n"
    footer = f"\n<b>[진단 결과]</b>\n스캔:{len(tickers)}개 / 주도주:{len(leading_stocks)}개 / 최종 타점:{final_pass_count}개"
    send_telegram_chunks(msg_list, header, footer)

if __name__ == "__main__":
    print("🚀 PRO-MASTER 버전 퀀트 스캐너 가동을 시작합니다...")
    analyze()
    print("✅ 스캔 및 알림 프로세스가 정상 종료되었습니다.")
