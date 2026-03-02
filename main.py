import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import requests
import os
import time
import json
from datetime import datetime, timedelta
import warnings
from functools import lru_cache

# 불필요한 경고문구 숨김 처리
warnings.filterwarnings('ignore')

# =========================
# [0. 계좌/리스크(운용 안정화용 가드레일)]
# =========================
ACCOUNT_CAPITAL = 10000  # 사용자 계좌 기본금 ($10,000)
RISK_AMOUNT = 200        # 1회 타점당 고정 리스크 ($200)

MAX_OPEN_POSITIONS = 5                       # 동시 보유 최대 포지션(스윙 권장 4~5)
MAX_TOTAL_RISK = int(ACCOUNT_CAPITAL * 0.08) # 총 리스크 상한(예: 8% = $800)
MAX_POSITION_DOLLARS = int(ACCOUNT_CAPITAL * 0.30)  # 종목당 최대 투자금(예: 30% = $3,000)
MAX_PER_SECTOR = 2        # 동일 섹터 최대 진입 허용 개수

# =========================
# [1. PRO-MASTER 실전 최적화 파라미터]
# =========================
PARTIAL_TP_R = 2.0
ORDER_VALID_DAYS = 1
VIX_MAX = 25
TRAIL_MULT = 5.0                 # 넉넉한 추세 스윙 유지
VOLUME_OVERHEAT_MULT = 3.0       # 거래량 3배 이상 폭발 배제
RS_CUTOFF = 75.0                 # RS 컷 75 유지 (상위 25% 타겟)

OPT_MULT_MIN = 2.0
OPT_MULT_MAX = 5.0               # 변동성에 맞춰 넉넉하게 스탑로스 숨통 트여주기
MIN_MAE_SAMPLES = 10
MIN_REV_SAMPLES = 5

MIN_SAMPLE_ENTRY = 5             # 샘플 5건 미만은 진입 제외
SAMPLE_DEFENSE_MAX = 9           # 샘플 5~9는 방어모드(샘플 부족)

EARNINGS_WARNING_DAYS = 3        # 실적발표 경고일수

FOMC_WARNING_DAYS = 3            # FOMC 회의 며칠 전부터 경고할지
FOMC_DATES = [
    "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16"
]

# 위키피디아 스크래핑 실패 시 최후의 보루 (에러 방지용)
FALLBACK_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AVGO", "TSLA", "COST", "NFLX"]

# =========================
# [2. 상태 파일 (실전 포지션 관리 및 월간 캐싱)]
# =========================
PENDING_ORDERS_PATH = "pending_orders.json"
POSITIONS_STATE_PATH = "positions_state.json"
FILLS_PATH = "fills.json"
MONTHLY_METRICS_PATH = "monthly_metrics.json"
TELEGRAM_OFFSET_PATH = "telegram_offset.json"
CACHE_SP = "tickers_cache_sp500.txt"
CACHE_NQ = "tickers_cache_nasdaq100.txt"

# =========================
# [3. 텔레그램 설정 및 수신 로직]
# =========================
BOT_TOKEN = os.environ.get('TG_TOKEN')
CHAT_ID = os.environ.get('TG_CHAT_ID')

def send_telegram(message: str):
    if not BOT_TOKEN or not CHAT_ID:
        print(message)
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, data=data, timeout=10)
    except:
        pass

def send_telegram_chunks(msg_list, header, footer):
    if not msg_list:
        send_telegram(header + "❌ <b>오늘은 조건에 맞는 1급 주도주가 없습니다.</b>\n" + footer)
        return
    chunk_size = 5
    for i in range(0, len(msg_list), chunk_size):
        chunk = msg_list[i:i + chunk_size]
        body = "\n".join(chunk)
        title = f"{header} (파트 {i//chunk_size + 1})\n\n"
        send_telegram(title + body + (footer if i + chunk_size >= len(msg_list) else ""))
        time.sleep(1)

def _load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return default

def _save_json(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except:
        pass

def load_pending_orders(): return _load_json(PENDING_ORDERS_PATH, {})
def save_pending_orders(data): _save_json(PENDING_ORDERS_PATH, data)
def load_positions(): return _load_json(POSITIONS_STATE_PATH, {})
def save_positions(data): _save_json(POSITIONS_STATE_PATH, data)
def load_fills(): return _load_json(FILLS_PATH, [])
def load_monthly_metrics(): return _load_json(MONTHLY_METRICS_PATH, {})
def save_monthly_metrics(data): _save_json(MONTHLY_METRICS_PATH, data)

def process_telegram_commands():
    if not BOT_TOKEN:
        return

    offset_data = _load_json(TELEGRAM_OFFSET_PATH, {"offset": 0})
    offset = offset_data.get("offset", 0)

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"

    try:
        res = requests.get(url, params={"offset": offset, "timeout": 5}, timeout=10)
        data = res.json()
        if not data.get("ok"):
            return

        fills = load_fills()
        new_offset = offset
        updates = data.get("result", [])

        for item in updates:
            update_id = item["update_id"]
            new_offset = max(new_offset, update_id + 1)

            msg = item.get("message", {})
            text = msg.get("text", "").strip()

            if text.startswith("매수"):
                parts = text.split()
                if len(parts) >= 3:
                    ticker = parts[1].upper()
                    try:
                        price = float(parts[2])
                        fills.append({
                            "ticker": ticker,
                            "fill_date": str(datetime.now().date()),
                            "fill_price": price
                        })
                        send_telegram(
                            f"📥 <b>[매수 수신 완료]</b>\n"
                            f"{ticker} 종목 매수가 ${price:.2f} 기록 완료!\n"
                            f"잠시 후 스캔이 진행되며 포지션으로 전환됩니다."
                        )
                    except:
                        pass
            
            elif text.startswith("1차매도"):
                parts = text.split()
                if len(parts) >= 2:
                    ticker = parts[1].upper()
                    positions = load_positions()
                    if ticker in positions:
                        pos = positions[ticker]
                        if not pos.get("sold_partial"):
                            sell_qty = pos["remaining"] // 2
                            if sell_qty > 0:
                                pos["remaining"] -= sell_qty
                            pos["sold_partial"] = True
                            save_positions(positions)
                            send_telegram(
                                f"💰 <b>[1차 익절 수신 완료] {ticker}</b>\n"
                                f"절반 매도 처리되어 잔여 수량이 {pos['remaining']}주로 업데이트되었습니다."
                            )
                        else:
                            send_telegram(f"ℹ️ {ticker} 종목은 이미 1차 매도 완료 상태입니다.")
                    else:
                        send_telegram(f"⚠️ <b>[오류]</b> {ticker} 종목은 현재 보유 장부에 없습니다.")
            
            elif text.startswith("매도"):
                parts = text.split()
                if len(parts) >= 2:
                    ticker = parts[1].upper()
                    positions = load_positions()
                    if ticker in positions:
                        del positions[ticker]
                        save_positions(positions)
                        send_telegram(
                            f"🗑️ <b>[청산 수신 완료]</b>\n"
                            f"{ticker} 종목이 보유 장부에서 완전히 삭제되었습니다.\n"
                            f"(새로운 매수 슬롯 1개 확보 완료)"
                        )
                    else:
                        send_telegram(f"⚠️ <b>[오류]</b> {ticker} 종목은 현재 보유 장부에 없습니다.")

        if new_offset > offset:
            _save_json(TELEGRAM_OFFSET_PATH, {"offset": new_offset})
            _save_json(FILLS_PATH, fills)

    except:
        pass

# =========================
# [4. 위키피디아 스크래핑 및 매크로 경고 모듈]
# =========================
UA_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def normalize_ticker(sym: str) -> str:
    return str(sym).replace(".", "-").strip()

def load_cache(path: str):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f.readlines() if line.strip()]
        except:
            pass
    return None

def save_cache(path: str, syms):
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(syms))
    except:
        pass

def fetch_html(url: str, retries=3, timeout=30) -> str:
    for k in range(retries):
        try:
            r = requests.get(url, headers=UA_HEADERS, timeout=timeout)
            r.raise_for_status()
            return r.text
        except:
            time.sleep(1.0 + k)
    raise RuntimeError(f"HTML fetch failed: {url}")

def wiki_tickers_sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        html = fetch_html(url)
        tables = pd.read_html(html, attrs={"id": "constituents"}, flavor="lxml")
        syms = tables[0]["Symbol"].astype(str).tolist()
        save_cache(CACHE_SP, syms)
        return syms
    except:
        cached = load_cache(CACHE_SP)
        return cached if cached else FALLBACK_TICKERS

def wiki_tickers_nasdaq100():
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    try:
        html = fetch_html(url)
        tables = pd.read_html(html, flavor="lxml")
        best = max(tables, key=lambda x: x.shape[0])
        col = next((c for c in best.columns if str(c).lower() in ["ticker", "symbol"]), best.columns[0])
        syms = best[col].astype(str).tolist()
        save_cache(CACHE_NQ, syms)
        return syms
    except:
        cached = load_cache(CACHE_NQ)
        return cached if cached else FALLBACK_TICKERS

def build_universe_backtest_style():
    sp = [normalize_ticker(s) for s in wiki_tickers_sp500()]
    nq = [normalize_ticker(s) for s in wiki_tickers_nasdaq100()]
    return sorted(set(sp) | set(nq))

@lru_cache(maxsize=4096)
def get_sector_cached(ticker_symbol: str) -> str:
    try:
        return yf.Ticker(ticker_symbol).info.get('sector', 'Unknown') or 'Unknown'
    except:
        return 'Unknown'

@lru_cache(maxsize=4096)
def get_earnings_warning_cached(ticker_symbol: str) -> str:
    try:
        cal = yf.Ticker(ticker_symbol).calendar
        if cal is not None and not getattr(cal, "empty", True):
            if hasattr(cal, "index") and ('Earnings Date' in cal.index):
                ed = cal.loc['Earnings Date'].values[0]
            else:
                ed = cal.iloc[0, 0]

            if isinstance(ed, (list, tuple, np.ndarray)) and len(ed) > 0:
                ed = ed[0]

            ed_ts = pd.to_datetime(ed)
            if hasattr(ed_ts, "date"):
                days_left = (ed_ts.date() - datetime.now().date()).days
                if 0 <= days_left <= EARNINGS_WARNING_DAYS:
                    d_str = "오늘" if days_left == 0 else f"{days_left}일 뒤"
                    return f"\n⚠️ <b>[주의] 실적발표 임박: {d_str} ({ed_ts.strftime('%Y-%m-%d')})</b> - 매수 시 참고 바랍니다."
    except:
        pass
    return ""

def get_fomc_warning() -> str:
    today = datetime.now().date()
    for date_str in FOMC_DATES:
        try:
            fomc_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            days_left = (fomc_date - today).days
            if 0 <= days_left <= FOMC_WARNING_DAYS:
                d_str = "오늘" if days_left == 0 else f"{days_left}일 뒤"
                return f"\n🚨 <b>[매크로 주의] FOMC 금리결정 임박: {d_str} ({date_str})</b> - 시장 변동성에 주의하세요."
        except:
            continue
    return ""

# =========================
# [5. yfinance 청크 다운로드 및 전처리 (안정성 강화)]
# =========================
def _ensure_multiindex(one_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if one_df is None or one_df.empty:
        return pd.DataFrame()
    if isinstance(one_df.columns, pd.MultiIndex):
        return one_df
    one_df = one_df.copy()
    one_df.columns = pd.MultiIndex.from_product([[ticker], one_df.columns])
    return one_df

def download_price_data_chunked(tickers, start_date, chunk_size=100, pause=1.2):
    all_parts = []
    tickers = list(tickers)

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        try:
            df = yf.download(
                chunk, start=start_date,
                group_by='ticker', threads=True,
                progress=False, auto_adjust=True
            )
            
            if df is None or df.empty:
                continue

            if len(chunk) == 1:
                df = _ensure_multiindex(df, chunk[0])
            elif not isinstance(df.columns, pd.MultiIndex):
                continue
                
            all_parts.append(df)
        except:
            pass
        time.sleep(pause)

    if not all_parts:
        return pd.DataFrame()

    try:
        merged = pd.concat(all_parts, axis=1)
        return merged.loc[:, ~merged.columns.duplicated(keep='last')]
    except:
        return pd.DataFrame()

def sanitize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Open', 'High', 'Low', 'Close'])
    df = df[(df['High'] >= df['Low']) & (df['Open'] > 0) & (df['High'] > 0) & (df['Low'] > 0) & (df['Close'] > 0)]
    df = df[(df['High'] - df['Low']) / df['Close'] <= 0.80]
    ret1 = df['Close'].pct_change().abs()
    df = df[(ret1.isna()) | (ret1 <= 0.60)]
    if 'Volume' in df.columns:
        df['Volume'] = df['Volume'].fillna(0)
        df = df[df['Volume'] >= 0]
    return df

# =========================
# [6. 벡터화된 RS 계산]
# =========================
def calc_rs_score_df(close_all: pd.DataFrame, spy_close: pd.Series) -> pd.DataFrame:
    periods = [63, 126, 189, 252]
    weights = [0.4, 0.2, 0.2, 0.2]
    rs_total = None

    for p, w in zip(periods, weights):
        stock_ret = close_all / close_all.shift(p)
        spy_ret = spy_close / spy_close.shift(p)
        part = stock_ret.div(spy_ret, axis=0) * w
        rs_total = part if rs_total is None else (rs_total + part)

    return rs_total

def build_rs_rank(close_all: pd.DataFrame, spy_close: pd.Series) -> pd.DataFrame:
    return calc_rs_score_df(close_all, spy_close).rank(axis=1, pct=True) * 100.0

# =========================
# [7. 동적 메트릭 클램핑 + 샘플/원인 추적]
# =========================
def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))

def get_optimal_metrics_backtest(df: pd.DataFrame):
    mae_list, historical_gaps, reversal_strengths = [], [], []
    temp_df = df.copy()

    temp_df["MA20"] = ta.sma(temp_df["Close"], 20)
    temp_df["EMA20"] = ta.ema(temp_df["Close"], 20)
    temp_df["MA50"] = ta.sma(temp_df["Close"], 50)
    temp_df["ATR"] = ta.atr(temp_df["High"], temp_df["Low"], temp_df["Close"], 14)
    temp_df["avg_v20"] = ta.sma(temp_df["Volume"], 20)
    temp_df["prev_v"] = temp_df["Volume"].shift(1)

    cond_inc = temp_df["Volume"] > temp_df["prev_v"]
    cond_exc = (temp_df["prev_v"] > temp_df["avg_v20"] * 1.5) & (temp_df["Volume"] > temp_df["avg_v20"])
    temp_df["is_vol_ok"] = (cond_inc | cond_exc) & (temp_df["Volume"] < temp_df["avg_v20"] * VOLUME_OVERHEAT_MULT)

    temp_df["is_green"] = temp_df["Close"] > temp_df["Open"]
    temp_df["c_range"] = temp_df["High"] - temp_df["Low"]
    temp_df["rev_pos"] = np.where(temp_df["c_range"] > 0, (temp_df["Close"] - temp_df["Low"]) / temp_df["c_range"], 0)

    temp_df["Sync_Signal"] = (
        (temp_df["MA20"] > temp_df["MA50"]) &
        (temp_df["Close"] <= temp_df["EMA20"]) &
        (temp_df["is_green"]) &
        (temp_df["rev_pos"] >= 0.6) &
        (temp_df["is_vol_ok"])
    )
    signals = temp_df[temp_df["Sync_Signal"]].index

    for idx in signals:
        loc = temp_df.index.get_loc(idx)
        if loc + 11 >= len(temp_df):
            continue

        close_p = float(temp_df.iloc[loc]["Close"])
        atr_p = float(temp_df.iloc[loc]["ATR"])
        low_p = float(temp_df.iloc[loc]["Low"])
        next_open_p = float(temp_df.iloc[loc + 1]["Open"])

        if close_p <= 0 or atr_p <= 0:
            continue

        historical_gaps.append(((next_open_p - close_p) / close_p) * 100)

        f_low = float(temp_df.iloc[loc + 1: loc + 11]["Low"].min())
        f_max = float(temp_df.iloc[loc + 1: loc + 11]["High"].max())

        if (close_p - f_low) > 0:
            mae_list.append((close_p - f_low) / atr_p)

        if f_max > close_p:
            reversal_strengths.append((close_p - low_p) / atr_p)

    sample_count = int(min(len(mae_list), len(reversal_strengths)))

    if len(mae_list) < MIN_MAE_SAMPLES or len(reversal_strengths) < MIN_REV_SAMPLES:
        raw_opt_mult = float(OPT_MULT_MIN)
        opt_mult = float(OPT_MULT_MIN)
        max_gap_threshold = 2.0
        min_reversal_factor = 0.5
        return opt_mult, raw_opt_mult, max_gap_threshold, min_reversal_factor, sample_count

    raw_opt_mult = float(np.percentile(mae_list, 90))
    opt_mult = clamp(raw_opt_mult, OPT_MULT_MIN, OPT_MULT_MAX)

    max_gap_threshold = max(float(np.percentile(historical_gaps, 80)) if historical_gaps else 2.0, 0.5)
    min_reversal_factor = float(np.percentile(reversal_strengths, 25))

    return opt_mult, raw_opt_mult, max_gap_threshold, min_reversal_factor, sample_count

# =========================
# [8. 주문 및 포지션 관리 엔진 (실전형)]
# =========================
def expire_pending_orders(pending: dict):
    today = datetime.now().date()
    to_remove = [
        t for t, od in pending.items()
        if today > datetime.strptime(od.get("expires_on", "2099-01-01"), "%Y-%m-%d").date()
    ]
    for t in to_remove:
        pending.pop(t, None)
    return pending

def apply_fills_to_positions(pending: dict, positions: dict):
    fills = load_fills()
    if not fills:
        return pending, positions

    new_fills_msgs = []
    for fill in fills:
        ticker = str(fill.get("ticker", "")).strip().upper()
        fill_price = float(fill.get("fill_price", 0))

        if not ticker or fill_price <= 0 or ticker in positions or ticker not in pending:
            continue

        od = pending[ticker]
        qty = int(od.get("qty", 0))
        opt_mult = float(od.get("opt_mult", OPT_MULT_MIN))
        atr_entry = float(od.get("atr_entry", 0))

        if qty < 1 or atr_entry <= 0:
            continue

        one_r = opt_mult * atr_entry
        sl_price = fill_price - one_r
        tp2_price = fill_price + PARTIAL_TP_R * one_r

        positions[ticker] = {
            "entry_date": fill.get("fill_date", str(datetime.now().date())),
            "entry_price": fill_price,
            "qty": qty,
            "remaining": qty,

            "opt_mult": opt_mult,
            "atr_entry": atr_entry,
            "one_r": one_r,

            "sl_price": sl_price,
            "tp2_price": tp2_price,

            "sold_partial": False,
            "trail_mult": float(TRAIL_MULT),

            "sl_alerted": False,
            "trail_alerted": False,
            "tp2_alerted": False  
        }

        pending.pop(ticker, None)

        new_fills_msgs.append(
            f"✅ <b>[체결확정] {ticker}</b>\n"
            f"- 체결가: ${fill_price:.2f} ({qty}주)\n"
            f"- SL: ${sl_price:.2f} | 1차 매도 (2R): ${tp2_price:.2f}"
        )

    if new_fills_msgs:
        send_telegram_chunks(new_fills_msgs, f"<b>📌 {datetime.now().date()} 체결확정 반영</b>\n\n", "")

    _save_json(FILLS_PATH, [])  
    return pending, positions

def evaluate_exits_and_alert(positions: dict, raw_data: pd.DataFrame):
    if not positions:
        return positions

    alerts = []

    for ticker, pos in positions.items():
        try:
            if ticker not in raw_data.columns.get_level_values(0):
                continue

            df = sanitize_ohlcv(raw_data[ticker].dropna().copy())
            if len(df) < 260:
                continue

            df["EMA20"] = ta.ema(df["Close"], 20)
            df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], 14)

            day = df.iloc[-1]
            day_high, day_low, day_close = float(day["High"]), float(day["Low"]), float(day["Close"])

            sl_price = float(pos.get("sl_price", 0))
            tp2_price = float(pos.get("tp2_price", 0))

            ema20 = float(df["EMA20"].iloc[-1])
            atr_now = float(df["ATR"].iloc[-1])
            trail_line = 0
            if np.isfinite(ema20) and atr_now > 0:
                trail_line = ema20 - float(TRAIL_MULT) * atr_now
                pos["current_trail"] = trail_line

            if sl_price > 0 and day_low <= sl_price and not pos.get("sl_alerted"):
                alerts.append(
                    f"🛑 <b>[손절가(SL) 이탈] {ticker}</b>\n"
                    f"- 오늘 저가(${day_low:.2f})가 SL(${sl_price:.2f}) 이탈\n"
                    f"- HTS 체결 확인 및 '매도 {ticker}' 명령 전송 바랍니다."
                )
                pos["sl_alerted"] = True

            if (not pos.get("sold_partial")) and tp2_price > 0 and day_high >= tp2_price:
                if not pos.get("tp2_alerted"):
                    alerts.append(
                        f"🎯 <b>[1차 목표가 도달] {ticker}</b>\n"
                        f"- 1차 목표가(${tp2_price:.2f})에 도달했습니다.\n"
                        f"- HTS에서 실제 체결이 완료되었다면 <b>'1차매도 {ticker}'</b> 라고 보내주세요.\n"
                        f"(봇은 수동 컨펌 전까지 장부를 수정하지 않습니다.)"
                    )
                    pos["tp2_alerted"] = True

            if trail_line > 0:
                if day_close < trail_line and not pos.get("trail_alerted"):
                    alerts.append(
                        f"📉 <b>[2차 매도 (TRAIL) 이탈] {ticker}</b>\n"
                        f"- 종가(${day_close:.2f})가 트레일선(${trail_line:.2f}) 이탈\n"
                        f"- 전량 청산 후 '매도 {ticker}' 명령 전송 바랍니다."
                    )
                    pos["trail_alerted"] = True

        except:
            continue

    if alerts:
        send_telegram_chunks(alerts, f"<b>📌 {datetime.now().date()} 매도 시그널 (PRO-MASTER)</b>\n\n", "")
    return positions

# =========================
# [9. 메인 엔진]
# =========================
def analyze():
    print(f"🚀 스캔 및 수신 시작: {datetime.now()}")
    process_telegram_commands()

    start_date = (pd.Timestamp.now() - pd.DateOffset(years=5)).strftime('%Y-%m-%d')
    print(f"데이터 수집 기준일: {start_date} (5년치)")

    try:
        m_raw = yf.download(["SPY", "^VIX"], start=start_date, progress=False, auto_adjust=True)
        m_close = m_raw["Close"] if isinstance(m_raw.columns, pd.MultiIndex) else m_raw["Close"]
        spy, vix = m_close["SPY"].dropna(), m_close["^VIX"].dropna()
    except:
        return send_telegram("⚠️ 시장 데이터 다운로드 실패")

    if len(spy) < 200:
        return send_telegram("⚠️ 지수 데이터 부족")

    spy_curr = float(spy.iloc[-1])
    vix_curr = float(vix.iloc[-1])
    spy_ma200 = float(ta.sma(spy, 200).iloc[-1])
    spy_ma5 = float(ta.sma(spy, 5).iloc[-1])

    if not (spy_curr > spy_ma200 and spy_curr > spy_ma5 and vix_curr < VIX_MAX):
        return send_telegram(
            f"⚠️ <b>시장 필터 작동 (매수 중단)</b>\n"
            f"VIX({vix_curr:.2f}) 또는 추세 이탈로 현금 보호."
        )

    print("유니버스 구성 중 (위키피디아)...")
    tickers = build_universe_backtest_style()
    if len(tickers) < 100:
        return send_telegram("⚠️ 유니버스 확보 실패")

    print(f"총 {len(tickers)}개 종목 가격 데이터 다운로드 중 (5년치)...")
    raw_data = download_price_data_chunked(tickers, start_date)
    if raw_data.empty:
        return send_telegram("⚠️ 가격 데이터 다운로드 실패")

    pending = expire_pending_orders(load_pending_orders())
    positions = load_positions()

    pending, positions = apply_fills_to_positions(pending, positions)
    positions = evaluate_exits_and_alert(positions, raw_data)

    save_pending_orders(pending)
    save_positions(positions)

    monthly_metrics = load_monthly_metrics()
    current_month = datetime.now().strftime("%Y-%m")
    if monthly_metrics.get("month") != current_month:
        print(f"🔄 월 변경 감지({current_month}). 메트릭 캐시를 초기화합니다.")
        monthly_metrics = {"month": current_month, "metrics": {}}

    print("RS 랭킹 계산 중 (벡터 연산)...")
    close_all = pd.DataFrame(index=spy.index)

    for t in tickers:
        try:
            if t in raw_data.columns.get_level_values(0):
                close_all = close_all.join(raw_data[t]["Close"].dropna().rename(t), how="left")
        except:
            pass

    rs_rank_df = build_rs_rank(close_all, spy)
    rs_today = rs_rank_df.iloc[-1] if not rs_rank_df.empty else pd.Series(dtype=float)

    fomc_msg = get_fomc_warning()

    msg_list = []
    final_pass_count = 0
    total_risk_used = 0
    sector_counts = {}

    print("타점 스캔 중...")

    for ticker in tickers:
        if final_pass_count >= MAX_OPEN_POSITIONS or total_risk_used + RISK_AMOUNT > MAX_TOTAL_RISK:
            break
        if ticker in positions or ticker in pending:
            continue

        try:
            rs_val = float(rs_today.get(ticker, np.nan))
            if not np.isfinite(rs_val) or rs_val < RS_CUTOFF:
                continue

            if ticker not in raw_data.columns.get_level_values(0):
                continue

            df = sanitize_ohlcv(raw_data[ticker].dropna().copy())
            if len(df) < 260:
                continue

            df["MA20"] = ta.sma(df["Close"], 20)
            df["EMA20"] = ta.ema(df["Close"], 20)
            df["MA50"] = ta.sma(df["Close"], 50)
            df["MA200"] = ta.sma(df["Close"], 200)
            df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], 14)
            df["AVG_V20"] = ta.sma(df["Volume"], 20)

            row, prev_row = df.iloc[-1], df.iloc[-2]
            cp = float(row["Close"])
            cv = float(row["Volume"])
            prev_v = float(prev_row["Volume"])
            avg_v20 = float(row["AVG_V20"])
            atr_val = float(row["ATR"])

            if not np.isfinite(atr_val) or atr_val <= 0 or cp <= 0:
                continue

            cond_inc = cv > prev_v
            cond_exc = (prev_v > avg_v20 * 1.5) and (cv > avg_v20)
            if not ((cond_inc or cond_exc) and cv < avg_v20 * VOLUME_OVERHEAT_MULT):
                continue

            if not (float(row["MA20"]) > float(row["MA50"]) and cp > float(row["MA200"]) and cp <= float(row["EMA20"])):
                continue

            c_range = float(row["High"]) - float(row["Low"])
            rev_pos = (cp - float(row["Low"])) / c_range if c_range > 0 else 0
            if not (cp > float(row["Open"]) and rev_pos >= 0.6):
                continue

            # ==========================================
            # ✅ 메트릭 캐시를 먼저 확인/계산 (API 호출 전 필터링)
            # ==========================================
            if ticker in monthly_metrics["metrics"]:
                cached = monthly_metrics["metrics"][ticker]
                opt_mult = float(cached.get("opt_mult", OPT_MULT_MIN))
                raw_opt_mult = float(cached.get("raw_opt_mult", OPT_MULT_MIN))
                max_gap_limit = float(cached.get("max_gap_limit", 2.0))
                min_rev_factor = float(cached.get("min_rev_factor", 0.5))
                sample_count = int(cached.get("sample_count", 0))
            else:
                opt_mult, raw_opt_mult, max_gap_limit, min_rev_factor, sample_count = get_optimal_metrics_backtest(df)
                monthly_metrics["metrics"][ticker] = {
                    "opt_mult": opt_mult,
                    "raw_opt_mult": raw_opt_mult,
                    "max_gap_limit": max_gap_limit,
                    "min_rev_factor": min_rev_factor,
                    "sample_count": sample_count
                }

            if sample_count < MIN_SAMPLE_ENTRY:
                continue

            if (cp - float(row["Low"])) / atr_val < float(min_rev_factor):
                continue

            # ==========================================
            # ✅ 끝까지 통과한 종목에 대해서만 API 정보 수집 (병목 방지)
            # ==========================================
            sector = get_sector_cached(ticker)
            if sector != "Unknown" and sector_counts.get(sector, 0) >= MAX_PER_SECTOR:
                continue

            earnings_msg = get_earnings_warning_cached(ticker)

            if sample_count <= SAMPLE_DEFENSE_MAX:
                mode_msg = f"🛡️ 방어모드 : ATR {opt_mult:.2f}배 적용 (샘플 {sample_count}건, 샘플 데이터 부족)"
                is_defense = True
            elif raw_opt_mult <= OPT_MULT_MIN + 1e-9:
                mode_msg = f"🛡️ 방어모드 : ATR {opt_mult:.2f}배 적용 (샘플 {sample_count}건, ATR 하한선 적용)"
                is_defense = True
            else:
                mode_msg = f"🧠 동적모드 : ATR {opt_mult:.2f}배 적용 (샘플 {sample_count}건)"
                is_defense = False

            stop_dist = atr_val * opt_mult
            if stop_dist <= 0:
                continue

            entry_limit_p = cp * (1 + max_gap_limit / 100.0)

            qty_risk = int(RISK_AMOUNT // stop_dist)
            qty_cap = int(MAX_POSITION_DOLLARS // entry_limit_p) if entry_limit_p > 0 else 0
            qty = min(qty_risk, qty_cap)

            if qty < 1:
                continue

            sl_price = entry_limit_p - stop_dist
            tp2_price = entry_limit_p + (PARTIAL_TP_R * stop_dist)

            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            final_pass_count += 1
            total_risk_used += RISK_AMOUNT

            pending[ticker] = {
                "created_on": str(datetime.now().date()),
                "expires_on": str((datetime.now().date() + timedelta(days=ORDER_VALID_DAYS))),
                "limit_price": entry_limit_p,
                "qty": qty,
                "opt_mult": opt_mult,
                "atr_entry": atr_val
            }

            msg_list.append(
                f"🚀 <b>[실전 주문] {ticker}</b> [{sector}] (RS: {rs_val:.1f} / 컷 {RS_CUTOFF:.0f}+)\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"🎯 <b>지정가 매수 : ${entry_limit_p:.2f}</b> (이하 체결)\n"
                f"🛑 <b>스탑로스(SL): ${sl_price:.2f}</b> (ATR {opt_mult:.2f}배)\n"
                f"✅ <b>1차 매도 (2R): ${tp2_price:.2f}</b>\n"
                f"📦 <b>매수 수량 : {qty}주</b> (리스크 ${RISK_AMOUNT})\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"📉 <b>트레일 기준:</b> EMA20 - {TRAIL_MULT}*ATR\n"
                f"{mode_msg}\n"
                f"{earnings_msg}"
                f"{fomc_msg}\n\n"
            )

        except:
            continue

    save_pending_orders(pending)
    save_monthly_metrics(monthly_metrics)

    pos_summary = "\n━━━━━━━━━━━━━━━━━━\n<b>[📊 현재 보유 포지션 대시보드]</b>\n"
    if positions:
        for t, p in positions.items():
            qty = p.get("remaining", p.get("qty", 0))
            tp2 = p.get("tp2_price", 0)
            trail = p.get("current_trail", 0)
            sl = p.get("sl_price", 0)
            
            status_1r = "✅ 완료" if p.get("sold_partial") else f"${tp2:.2f}"
            
            pos_summary += (
                f"🔹 <b>{t}</b> ({qty}주 보유 중)\n"
                f" ┣ 1차 매도(2R) : {status_1r}\n"
                f" ┗ 2차 매도(TRAIL) : ${trail:.2f} (손절가: ${sl:.2f})\n\n"
            )
    else:
        pos_summary += "보유 중인 종목이 없습니다.\n\n"

    pos_summary += "━━━━━━━━━━━━━━━━━━\n"

    header = f"<b>📅 {datetime.now().date()} 퀀트 보고서 (PRO-MASTER 통합본)</b>\n\n"
    footer = f"{pos_summary}<b>[스캔 결과]</b> 유니버스:{len(tickers)}개 / 타점:{final_pass_count}개 / 슬롯잔여:{MAX_OPEN_POSITIONS - len(positions)}개"
    
    send_telegram_chunks(msg_list, header, footer)

if __name__ == "__main__":
    print("🚀 PRO-MASTER 최종 통합 스캐너 가동 시작...")
    analyze()
    print("✅ 프로세스 정상 종료.")
