# bot.py —— 美股信号机器人
# 特性：5m→15m→30m→1d 自动降级、重试、盘前后支持、中文名、仓位/止损止盈、最强/最弱摘要
# 标的：SPY/QQQ/AAPL/MSFT/NVDA/AMD/MU/TXN/NEE/JNJ/TSLA/META/GOOGL/TSM/UNH
# 依赖：yfinance pandas numpy pytz requests
# GitHub Actions Secrets：BOT_TOKEN / CHAT_ID

import os, math, time, requests, pytz, datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf

# ============ Telegram ============
BOT_TOKEN = os.environ.get("BOT_TOKEN", "").strip()
CHAT_ID   = os.environ.get("CHAT_ID", "").strip()
TG_URL    = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

def send(msg: str):
    if not (BOT_TOKEN and CHAT_ID):
        print("Telegram 未配置"); return
    try:
        requests.post(TG_URL, data={"chat_id": CHAT_ID, "text": msg})
    except Exception as e:
        print("Telegram error:", e)

# ============ 标的 ============
TICKERS = [
    "SPY","QQQ","AAPL","MSFT","NVDA","AMD","MU","TXN","NEE","JNJ",
    "TSLA","META","GOOGL","TSM","UNH"
]

# ============ 策略与参数 ============
RSI_LEN   = 14
EMA_FAST  = 20
EMA_SLOW  = 50
MACD_FAST, MACD_SLOW, MACD_SIG = 12, 26, 9
ATR_LEN   = 14
VOL_SMA   = 20
MIN_BARS_MAP = {"5m": 40, "15m": 24, "30m": 16, "1d": 20}
VOLUME_BOOST = 1.2

RUN_WINDOW = os.environ.get("RUN_WINDOW", "regular").lower()     # regular / extended
STRATEGY   = os.environ.get("STRATEGY", "conservative").lower()  # conservative / aggressive
eastern    = pytz.timezone("US/Eastern")

# ============ 名称映射（中文） ============
NAME_MAP = {
    "SPY": "SPDR S&P 500 ETF（标普500）",
    "QQQ": "Invesco QQQ（纳指100）",
    "AAPL": "Apple Inc.（苹果）",
    "MSFT": "Microsoft Corp.（微软）",
    "NVDA": "NVIDIA Corp.（英伟达）",
    "AMD": "Advanced Micro Devices（超微）",
    "MU":   "Micron Technology（美光）",
    "TXN":  "Texas Instruments（德仪）",
    "NEE":  "NextEra Energy（新纪元能源）",
    "JNJ":  "Johnson & Johnson（强生）",
    "TSLA":"Tesla, Inc.（特斯拉）",
    "META":"Meta Platforms（脸书）",
    "GOOGL":"Alphabet Inc. A（谷歌A）",
    "TSM": "Taiwan Semiconductor（台积电）",
    "UNH": "UnitedHealth Group（联合健康）",
}

def company_name(ticker: str) -> str:
    name = NAME_MAP.get(ticker.upper())
    if name: return name
    try:
        info = yf.Ticker(ticker).get_info()
        return info.get("shortName") or info.get("longName") or ticker
    except Exception:
        return ticker

# ============ 指标工具 ============
def _series(x):
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            x = x.iloc[:, 0]
        else:
            for col in ("Close","Adj Close","close","adjclose","price"):
                if col in x.columns:
                    x = x[col]; break
            else:
                x = x.iloc[:, 0]
    return pd.Series(x, index=getattr(x, "index", None), dtype="float64").astype(float)

def ema(s, n): return _series(s).ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    s = _series(s); d = s.diff()
    up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up, index=s.index).rolling(n, min_periods=n).mean()
    rd = pd.Series(dn, index=s.index).rolling(n, min_periods=n).mean()
    rs = ru / (rd + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))

def macd(s):
    s = _series(s)
    ef = ema(s, MACD_FAST); es = ema(s, MACD_SLOW)
    line = ef - es
    sig  = line.ewm(span=MACD_SIG, adjust=False).mean()
    hist = line - sig
    return line, sig, hist

def atr(df, n=14):
    h = _series(df["High"]); l = _series(df["Low"]); c = _series(df["Close"])
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def clean_df(df):
    if df is None or len(df) == 0: return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    for col in ["Open","High","Low","Close","Adj Close","Volume"]:
        if col in df.columns: df[col] = _series(df[col])
    return df.dropna(how="any")

# ============ 稳健下载：5m→15m→30m→1d 自动降级 ============
def download_bars(ticker: str, tries: int = 3):
    intervals = [("5m","10d"), ("15m","30d"), ("30m","60d"), ("1d","200d")]
    for interval, period in intervals:
        for i in range(tries):
            try:
                df = yf.download(
                    ticker,
                    period=period,
                    interval=interval,
                    prepost=True,
                    auto_adjust=False,
                    progress=False,
                    timeout=30,
                    threads=False,
                    repair=True
                )
                if isinstance(df, pd.DataFrame) and len(df) > 0:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df = df.copy()
                    df["interval"] = interval
                    return df
            except Exception as e:
                print(f"{ticker} {interval} attempt {i+1} failed: {e}")
            time.sleep(1.5)
    return pd.DataFrame()

# ============ 时间窗 ============
def within_window():
    now = dt.datetime.now(dt.timezone.utc).astimezone(eastern)
    if RUN_WINDOW == "regular":
        s = now.replace(hour=9, minute=30, second=0, microsecond=0)
        e = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return s <= now <= e
    else:
        pre_s = now.replace(hour=7,  minute=0, second=0, microsecond=0)
        pre_e = now.replace(hour=9,  minute=30, second=0, microsecond=0)
        reg_s = now.replace(hour=9,  minute=30, second=0, microsecond=0)
        reg_e = now.replace(hour=16, minute=0, second=0, microsecond=0)
        aft_s = now.replace(hour=16, minute=0, second=0, microsecond=0)
        aft_e = now.replace(hour=20, minute=0, second=0, microsecond=0)
        return (pre_s <= now <= pre_e) or (reg_s <= now <= reg_e) or (aft_s <= now <= aft_e)

# ============ 策略 ============
def signal_logic(price, ema20, ema50, rsi_v, macd_up, macd_dn, vol_ok):
    if STRATEGY == "aggressive":
        bull = (price > ema20) and macd_up and (rsi_v > 48)
        bear = (price < ema20) and macd_dn and (rsi_v < 52)
    else:  # conservative
        bull = (price > ema20 > ema50) and macd_up and (rsi_v > 50) and vol_ok
        bear = (price < ema20 < ema50) and macd_dn and (rsi_v < 50) and vol_ok
    if bull: return "buy",  ("强势" if rsi_v > 60 else "普通")
    if bear: return "sell", ("强势" if rsi_v < 40 else "普通")
    return "hold", None

def position_advice(tag):
    if tag == "强势": return "重仓50%"
    if tag == "普通": return "中仓30%"
    return "轻仓10%"

# ============ 评分（用于“今日最强/最弱”） ============
def strength_score(price, ema20, ema50, rsi_v, macd_hist, vol_ok, action):
    # 归一到相对比率，避免量纲影响
    eps = 1e-9
    trend = (price - ema20) / max(price, eps) + (ema20 - ema50) / max(price, eps)
    momentum = (rsi_v - 50)/50.0 + np.tanh((macd_hist)/(abs(price)*0.001 + eps))
    vol = 0.3 if vol_ok else 0.0
    score = trend + momentum + vol
    if action == "sell":
        score = -score  # 卖出信号越强，score 越负；为了统一“最强买”和“最弱=最强卖”取 min
    return float(score)

# ============ 单标的分析（返回结构化结果，便于摘要排名） ============
def analyze_one(ticker: str):
    cname = company_name(ticker)

    df = download_bars(ticker)
    if df.empty:
        return {"ticker": ticker, "cname": cname, "text": f"⚠️数据源暂不可用（bars=0）",
                "action": "none", "score": None}

    interval = str(df["interval"].iloc[0]) if "interval" in df.columns else "?"
    df = clean_df(df)

    need_bars = MIN_BARS_MAP.get(interval, 24)
    if len(df) < need_bars:
        return {"ticker": ticker, "cname": cname,
                "text": f"⚪️观望 | 理由：{interval} 数据不足（bars={len(df)}）",
                "action": "hold", "score": None}

    # 指标
    df["EMA20"]  = ema(df["Close"], EMA_FAST)
    df["EMA50"]  = ema(df["Close"], EMA_SLOW)
    df["RSI"]    = rsi(df["Close"], RSI_LEN)
    mline, msig, mhist = macd(df["Close"])
    df["MACD"]   = mline; df["MACDsig"] = msig; df["MACDhist"] = mhist
    df["ATR"]    = atr(df, ATR_LEN)
    df["VolSMA"] = _series(df["Volume"]).rolling(VOL_SMA, min_periods=5).mean()

    last, prev = df.iloc[-1], df.iloc[-2]
    price  = float(last["Close"])
    ema20  = float(last["EMA20"])
    ema50  = float(last["EMA50"])
    rsi_v  = float(last["RSI"])
    macd_hist = float(last["MACDhist"])

    macd_up = (last["MACD"] > last["MACDsig"]) and \
              (prev["MACD"] <= prev["MACDsig"] or last["MACDhist"] > prev["MACDhist"])
    macd_dn = (last["MACD"] < last["MACDsig"]) and \
              (prev["MACD"] >= prev["MACDsig"] or last["MACDhist"] < prev["MACDhist"])

    try:
        vol_ok = float(last["Volume"]) >= float(last["VolSMA"]) * VOLUME_BOOST
    except Exception:
        vol_ok = False

    action, tag = signal_logic(price, ema20, ema50, rsi_v, macd_up, macd_dn, vol_ok)

    # 风险：ATR 兜底
    atr_v = float(last["ATR"]) if not math.isnan(float(last["ATR"])) else price * 0.008
    risk  = max(atr_v * 1.5, price * 0.012)

    suffix = "" if interval in ("5m","15m","30m") else "（⚠️已降级为日线）"

    if action == "buy":
        stop = price - risk; tp = price + 2 * risk
        pos  = position_advice(tag)
        text = (f"🟢{'强势' if tag=='强势' else ''}买入{suffix} | {interval} | "
                f"价:{price:.2f} 止损:{stop:.2f} 止盈:{tp:.2f} | 仓位：{pos}\n"
                f"理由：价/均线 {price:.2f}>{ema20:.2f}>{ema50:.2f} + MACD转强 + RSI:{int(rsi_v)}"
                f"{' + 放量' if vol_ok else ''}")
    elif action == "sell":
        stop = price + risk; tp = price - 2 * risk
        pos  = position_advice(tag)
        text = (f"🔴{'强势' if tag=='强势' else ''}卖出{suffix} | {interval} | "
                f"价:{price:.2f} 止损:{stop:.2f} 止盈:{tp:.2f} | 仓位：{pos}\n"
                f"理由：价/均线 {price:.2f}<{ema20:.2f}<{ema50:.2f} + MACD转弱 + RSI:{int(rsi_v)}"
                f"{' + 放量' if vol_ok else ''}")
    else:
        text = (f"⚪️观望{suffix} | {interval} | "
                f"价:{price:.2f} EMA20:{ema20:.2f} EMA50:{ema50:.2f} RSI:{int(rsi_v)}")

    score = strength_score(price, ema20, ema50, rsi_v, macd_hist, vol_ok, action) if action in ("buy","sell") else None

    return {"ticker": ticker, "cname": cname, "text": text, "action": action,
            "score": score, "interval": interval}

# ============ 主流程 ============
def within_header():
    return "常规盘" if RUN_WINDOW == "regular" else "盘前/常规/盘后"

def build_summary(results):
    # 选出最强买入（score 最大的正值）与最强卖出（score 最小的负值）
    buys = [r for r in results if r["action"] == "buy" and r["score"] is not None]
    sells = [r for r in results if r["action"] == "sell" and r["score"] is not None]
    best_buy  = max(buys, key=lambda r: r["score"]) if buys else None
    best_sell = min(sells, key=lambda r: r["score"]) if sells else None  # 更负代表更强的做空

    parts = []
    if best_buy:
        parts.append(f"最强买入：{best_buy['ticker']}/{best_buy['cname']}（{best_buy['interval']} | 评分：{best_buy['score']:.2f}）")
    if best_sell:
        parts.append(f"最强卖出：{best_sell['ticker']}/{best_sell['cname']}（{best_sell['interval']} | 评分：{best_sell['score']:.2f}）")
    if not parts:
        return "今日最强/最弱：暂无明确强信号（以观望为主）"
    return "；".join(parts)

def main():
    if not within_window():
        send(f"⏱ 当前不在设定时间窗（{within_header()}），本次不推送。")
        return

    results, lines = [], []
    for t in TICKERS:
        try:
            r = analyze_one(t)
        except Exception as e:
            r = {"ticker": t, "cname": company_name(t), "text": f"⚠️数据异常：{e}",
                 "action": "none", "score": None}
        results.append(r)
        lines.append(r["text"])

    summary = build_summary(results)
    header  = f"5分钟信号（策略：{STRATEGY}，窗口：{RUN_WINDOW}）\n🏁 {summary}"
    send(header + "\n" + "\n".join(lines))

if __name__ == "__main__":
    main()
