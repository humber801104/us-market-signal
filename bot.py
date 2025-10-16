# bot.py —— 每次运行抓取一次并推送（5分钟级别）
# 依赖：yfinance pandas numpy pytz requests
# 机密：从 GitHub Actions Secrets 读取 BOT_TOKEN / CHAT_ID

import os, math, requests, pytz, datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf

# === 机密从环境读取 ===
BOT_TOKEN = os.environ.get("BOT_TOKEN", "").strip()
CHAT_ID   = os.environ.get("CHAT_ID", "").strip()
TG_URL    = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

# === 配置 ===
TICKERS   = ["SPY","QQQ","AAPL","MSFT","NVDA","AMD","MU","TXN","NEE","JNJ","TSLA","META","GOOGL","TSM","UNH"]
RSI_LEN   = 14
EMA_FAST  = 20
EMA_SLOW  = 50
MACD_FAST, MACD_SLOW, MACD_SIG = 12, 26, 9
ATR_LEN   = 14
VOL_SMA   = 20
MIN_BARS  = 40            # 盘中前半段数据较少；40根更稳
VOLUME_BOOST = 1.2        # 放量阈值
eastern   = pytz.timezone("US/Eastern")

# === 工具函数（全部保证一维 Series）===
def _series(x):
    """确保返回一维 float Series，修复 (n,1)/多层列等情况"""
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            x = x.iloc[:, 0]
        else:
            # 取 Close/High/Low 等常用列名
            for col in ("Close","close","Adj Close","adjclose","price"):
                if col in x.columns:
                    x = x[col]
                    break
            else:
                x = x.iloc[:, 0]
    return pd.Series(x, index=getattr(x, "index", None), dtype="float64").astype(float)

def ema(s, n):
    s = _series(s)
    return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    s = _series(s)
    d  = s.diff()
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up, index=s.index).rolling(n, min_periods=n).mean()
    rd = pd.Series(dn, index=s.index).rolling(n, min_periods=n).mean()
    rs = ru / (rd + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))

def macd(s):
    s = _series(s)
    ef = ema(s, MACD_FAST)
    es = ema(s, MACD_SLOW)
    line = ef - es
    sig  = line.ewm(span=MACD_SIG, adjust=False).mean()
    hist = line - sig
    return line, sig, hist

def atr(df, n=14):
    h = _series(df["High"])
    l = _series(df["Low"])
    c = _series(df["Close"])
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def send(msg):
    if not (BOT_TOKEN and CHAT_ID): 
        print("Telegram 未配置")
        return
    try:
        requests.post(TG_URL, data={"chat_id": CHAT_ID, "text": msg})
    except Exception as e:
        print("Telegram error:", e)

def within_trading_hours():
    now = dt.datetime.now(dt.timezone.utc).astimezone(eastern)
    start = now.replace(hour=9, minute=30, second=0, microsecond=0)
    end   = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return start <= now <= end

def clean_df(df):
    """修复 yfinance 的 MultiIndex 列、二维列、类型问题"""
    if df is None or len(df) == 0:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        # 取顶层列名（Open/High/Low/Close/Adj Close/Volume）
        df.columns = df.columns.get_level_values(0)
    # 确保关键列存在且是一维
    for col in ["Open","High","Low","Close","Adj Close","Volume"]:
        if col in df.columns:
            s = _series(df[col])
            df[col] = s
    # 去 NaN
    df = df.dropna(how="any")
    return df

def analyze_one(t):
    try:
        df = yf.download(t, period="5d", interval="5m", auto_adjust=False, progress=False)
    except Exception as e:
        return f"⚠️下载异常：{e}"

    df = clean_df(df)
    if df.empty or len(df) < MIN_BARS:
        return f"⚪️观望 | 理由：数据不足（bars={len(df)})"

    # 指标
    df["EMA20"]   = ema(df["Close"], EMA_FAST)
    df["EMA50"]   = ema(df["Close"], EMA_SLOW)
    df["RSI"]     = rsi(df["Close"], RSI_LEN)
    mline, msig, mhist = macd(df["Close"])
    df["MACD"]    = mline
    df["MACDsig"] = msig
    df["MACDhist"]= mhist
    df["ATR"]     = atr(df, ATR_LEN)
    df["VolSMA"]  = _series(df["Volume"]).rolling(VOL_SMA, min_periods=5).mean()

    last, prev = df.iloc[-1], df.iloc[-2]
    price  = float(last["Close"])
    ema20  = float(last["EMA20"])
    ema50  = float(last["EMA50"])
    rsi_v  = float(last["RSI"])
    macd_up  = (last["MACD"] > last["MACDsig"]) and (prev["MACD"] <= prev["MACDsig"] or last["MACDhist"] > prev["MACDhist"])
    macd_dn  = (last["MACD"] < last["MACDsig"]) and (prev["MACD"] >= prev["MACDsig"] or last["MACDhist"] < prev["MACDhist"])
    vol_ok = False
    try:
        vol_ok = float(last["Volume"]) >= float(last["VolSMA"]) * VOLUME_BOOST
    except Exception:
        vol_ok = False

    bull = (price > ema20 > ema50) and macd_up and (rsi_v > 50) and vol_ok
    bear = (price < ema20 < ema50) and macd_dn and (rsi_v < 50) and vol_ok

    # 风险控制（ATR兜底 1.2%）
    atr_v = float(last["ATR"]) if not math.isnan(float(last["ATR"])) else price * 0.008
    risk  = max(atr_v * 1.5, price * 0.012)

    if bull:
        stop = price - risk
        tp   = price + 2 * risk
        return f"🟢买入 | 价:{price:.2f} 止损:{stop:.2f} 止盈:{tp:.2f}\n理由：价>EMA20>EMA50 + MACD转强 + RSI>{int(rsi_v)} + 放量"
    if bear:
        stop = price + risk
        tp   = price - 2 * risk
        return f"🔴卖出 | 价:{price:.2f} 止损:{stop:.2f} 止盈:{tp:.2f}\n理由：价<EMA20<EMA50 + MACD转弱 + RSI<{int(rsi_v)} + 放量"

    return f"⚪️观望 | 理由：条件未齐（价:{price:.2f} EMA20:{ema20:.2f} EMA50:{ema50:.2f} RSI:{int(rsi_v)})"

def main():
    if not within_trading_hours():
        send("⏱ 当前不在美股盘中（09:30–16:00 ET），本次不推送。")
        return
    lines = []
    for t in TICKERS:
        try:
            lines.append(analyze_one(t))
        except Exception as e:
            lines.append(f"⚠️数据异常：{e}")
    send("5分钟信号\n" + "\n".join(lines))

if __name__ == "__main__":
    main()
