# bot.py —— 美股信号机器人（5分钟级别）
# 功能：SPY/QQQ/AAPL/MSFT/NVDA/AMD/MU/TXN/NEE/JNJ/TSLA/META/GOOGL/TSM/UNH
# 依赖：yfinance pandas numpy pytz requests
# 密钥：从 GitHub Actions Secrets 读取 BOT_TOKEN / CHAT_ID

import os, math, requests, pytz, datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf

# ====== 机密（来自 Actions → Secrets）======
BOT_TOKEN = os.environ.get("BOT_TOKEN", "").strip()
CHAT_ID   = os.environ.get("CHAT_ID", "").strip()
TG_URL    = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

# ====== 监控标的 ======
TICKERS = [
    "SPY","QQQ","AAPL","MSFT","NVDA","AMD","MU","TXN","NEE","JNJ",
    "TSLA","META","GOOGL","TSM","UNH"
]

# ====== 策略参数 ======
RSI_LEN   = 14
EMA_FAST  = 20
EMA_SLOW  = 50
MACD_FAST, MACD_SLOW, MACD_SIG = 12, 26, 9
ATR_LEN   = 14
VOL_SMA   = 20
MIN_BARS  = 40              # 盘中前半段较少K线，用40根确保能计算
VOLUME_BOOST = 1.2          # 放量阈值：成交量 >= 20均量 * 1.2

# ====== 运行窗口：regular(常规09:30–16:00 ET) / extended(含盘前07:00–09:30与盘后16:00–20:00) ======
RUN_WINDOW = os.environ.get("RUN_WINDOW", "regular").lower()
eastern    = pytz.timezone("US/Eastern")

# ====== 名称映射（中英混排，快且稳定）======
NAME_MAP = {
    "SPY": "SPDR S&P 500 ETF（标普500）",
    "QQQ": "Invesco QQQ（纳指100）",
    "AAPL": "Apple Inc.（苹果）",
    "MSFT": "Microsoft Corp.（微软）",
    "NVDA": "NVIDIA Corp.（英伟达）",
    "AMD": "Advanced Micro Devices（超微）",
    "MU":   "Micron Technology（美光）",
    "TXN":  "Texas Instruments（德州仪器）",
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
    # 兜底：偶发请求，避免频繁调用导致限流
    try:
        info = yf.Ticker(ticker).get_info()
        return info.get("shortName") or info.get("longName") or ticker
    except Exception:
        return ticker

# ====== 工具函数：保证一维 Series，修复 (n,1) 形状 ======
def _series(x):
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            x = x.iloc[:, 0]
        else:
            for col in ("Close","close","Adj Close","adjclose","price"):
                if col in x.columns:
                    x = x[col]; break
            else:
                x = x.iloc[:, 0]
    return pd.Series(x, index=getattr(x, "index", None), dtype="float64").astype(float)

def ema(s, n):
    s = _series(s); return s.ewm(span=n, adjust=False).mean()

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
        if col in df.columns:
            df[col] = _series(df[col])
    return df.dropna(how="any")

# ====== 时间窗口判断 ======
def within_window():
    now = dt.datetime.now(dt.timezone.utc).astimezone(eastern)
    if RUN_WINDOW == "regular":
        start = now.replace(hour=9, minute=30, second=0, microsecond=0)
        end   = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return start <= now <= end
    else:  # extended
        pre_s = now.replace(hour=7, minute=0, second=0, microsecond=0)
        pre_e = now.replace(hour=9, minute=30, second=0, microsecond=0)
        reg_s = now.replace(hour=9, minute=30, second=0, microsecond=0)
        reg_e = now.replace(hour=16, minute=0, second=0, microsecond=0)
        aft_s = now.replace(hour=16, minute=0, second=0, microsecond=0)
        aft_e = now.replace(hour=20, minute=0, second=0, microsecond=0)
        return (pre_s <= now <= pre_e) or (reg_s <= now <= reg_e) or (aft_s <= now <= aft_e)

# ====== 推送 ======
def send(msg: str):
    if not (BOT_TOKEN and CHAT_ID):
        print("Telegram 未配置"); return
    try:
        requests.post(TG_URL, data={"chat_id": CHAT_ID, "text": msg})
    except Exception as e:
        print("Telegram error:", e)

# ====== 双策略：稳健 conservative / 进取 aggressive （用环境变量 STRATEGY 选择）======
STRATEGY = os.environ.get("STRATEGY", "conservative").lower()

def signal_logic(price, ema20, ema50, rsi_v, macd_up, macd_dn, vol_ok):
    """
    返回 ('buy'/'sell'/'hold', strength_tag)
    strength_tag: "强势"/"普通"/None
    """
    if STRATEGY == "aggressive":
        bull = (price > ema20) and macd_up and (rsi_v > 48)
        bear = (price < ema20) and macd_dn and (rsi_v < 52)
    else:  # conservative
        bull = (price > ema20 > ema50) and macd_up and (rsi_v > 50) and vol_ok
        bear = (price < ema20 < ema50) and macd_dn and (rsi_v < 50) and vol_ok

    if bull:
        return "buy", ("强势" if rsi_v > 60 else "普通")
    if bear:
        return "sell", ("强势" if rsi_v < 40 else "普通")
    return "hold", None

def position_advice(tag):
    # 结合强度给出仓位
    if tag == "强势": return "重仓50%"
    if tag == "普通": return "中仓30%"
    return "轻仓10%"

# ====== 单标的分析 ======
def analyze_one(ticker: str) -> str:
    cname = company_name(ticker)
    try:
        df = yf.download(ticker, period="5d", interval="5m", auto_adjust=False, progress=False)
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
    df["MACD"]    = mline; df["MACDsig"] = msig; df["MACDhist"] = mhist
    df["ATR"]     = atr(df, ATR_LEN)
    df["VolSMA"]  = _series(df["Volume"]).rolling(VOL_SMA, min_periods=5).mean()

    last, prev = df.iloc[-1], df.iloc[-2]
    price  = float(last["Close"])
    ema20  = float(last["EMA20"])
    ema50  = float(last["EMA50"])
    rsi_v  = float(last["RSI"])
    macd_up = (last["MACD"] > last["MACDsig"]) and \
              (prev["MACD"] <= prev["MACDsig"] or last["MACDhist"] > prev["MACDhist"])
    macd_dn = (last["MACD"] < last["MACDsig"]) and \
              (prev["MACD"] >= prev["MACDsig"] or last["MACDhist"] < prev["MACDhist"])
    try:
        vol_ok = float(last["Volume"]) >= float(last["VolSMA"]) * VOLUME_BOOST
    except Exception:
        vol_ok = False

    # 生成信号
    action, tag = signal_logic(price, ema20, ema50, rsi_v, macd_up, macd_dn, vol_ok)

    # 风险控制（ATR 兜底：≥1.2%）
    atr_v = float(last["ATR"]) if not math.isnan(float(last["ATR"])) else price * 0.008
    risk  = max(atr_v * 1.5, price * 0.012)

    if action == "buy":
        stop = price - risk; tp = price + 2 * risk
        pos  = position_advice(tag)
        strength = "强势买入信号" if tag == "强势" else "买入"
        return (f"🟢{strength} | 价:{price:.2f} 止损:{stop:.2f} "
                f"止盈:{tp:.2f} | 仓位建议：{pos}\n"
                f"理由：价/均线:{price:.2f}>{ema20:.2f}>{ema50:.2f} "
                f"+ MACD转强 + RSI:{int(rsi_v)}{' + 放量' if vol_ok else ''}")

    if action == "sell":
        stop = price + risk; tp = price - 2 * risk
        pos  = position_advice(tag)
        strength = "强势卖出信号" if tag == "强势" else "卖出"
        return (f"🔴{strength} | 价:{price:.2f} 止损:{stop:.2f} "
                f"止盈:{tp:.2f} | 仓位建议：{pos}\n"
                f"理由：价/均线:{price:.2f}<{ema20:.2f}<{ema50:.2f} "
                f"+ MACD转弱 + RSI:{int(rsi_v)}{' + 放量' if vol_ok else ''}")

    return (f"⚪️观望 | 理由：条件未齐（价:{price:.2f} "
            f"EMA20:{ema20:.2f} EMA50:{ema50:.2f} RSI:{int(rsi_v)})")

# ====== 主流程 ======
def main():
    if not within_window():
        window_note = "常规盘" if RUN_WINDOW=="regular" else "盘前/常规/盘后"
        send(f"⏱ 当前不在设定时间窗（{window_note}），本次不推送。")
        return
    lines = []
    for t in TICKERS:
        try:
            lines.append(analyze_one(t))
        except Exception as e:
            lines.append(f"⚠️数据异常：{e}")
    header = f"5分钟信号（策略：{STRATEGY}，窗口：{RUN_WINDOW}）"
    send(header + "\n" + "\n".join(lines))

if __name__ == "__main__":
    main()
