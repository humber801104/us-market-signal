# bot.py  —— 每次运行抓取一次并推送
import math, os, requests, pytz, datetime as dt
import pandas as pd, numpy as np
import yfinance as yf

BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
CHAT_ID   = os.environ.get("CHAT_ID", "")
TG_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

TICKERS = ["SPY","QQQ","AAPL","MSFT","NVDA","AMD","MU","TXN","NEE","JNJ","TSLA","META","GOOGL","TSM","UNH"]
RSI_LEN=14; EMA_FAST=20; EMA_SLOW=50
MACD_FAST, MACD_SLOW, MACD_SIG = 12,26,9
ATR_LEN=14; VOL_SMA=20
eastern = pytz.timezone("US/Eastern")

def within_trading_hours():
    now = dt.datetime.now(dt.timezone.utc).astimezone(eastern)
    start = now.replace(hour=9, minute=30, second=0, microsecond=0)
    end   = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return start <= now <= end

def send(msg):
    if not BOT_TOKEN or not CHAT_ID: return
    try:
        requests.post(TG_URL, data={"chat_id": CHAT_ID, "text": msg})
    except Exception as e:
        print("TG error:", e)

def rsi(s, n=14):
    d = s.diff(); up = np.where(d>0, d, 0.0); dn = np.where(d<0, -d, 0.0)
    ru = pd.Series(up, index=s.index).rolling(n).mean()
    rd = pd.Series(dn, index=s.index).rolling(n).mean()
    rs = ru/(rd+1e-9); return 100 - (100/(1+rs))

def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def macd(s):
    ef, es = ema(s, MACD_FAST), ema(s, MACD_SLOW)
    line = ef - es; sig = line.ewm(span=MACD_SIG, adjust=False).mean()
    hist = line - sig; return line, sig, hist

def atr(df, n=14):
    h,l,c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def analyze_one(t):
    df = yf.download(t, period="5d", interval="5m", auto_adjust=False, progress=False)
    if df is None or df.empty or len(df)<60:
        return f"⚪️观望 | 理由：数据不足"
    df = df.dropna().copy()
    df["EMA20"] = ema(df["Close"], EMA_FAST)
    df["EMA50"] = ema(df["Close"], EMA_SLOW)
    df["RSI"]   = rsi(df["Close"], RSI_LEN)
    mline, msig, mhist = macd(df["Close"])
    df["MACD"], df["MACDsig"], df["MACDhist"] = mline, msig, mhist
    df["ATR"] = atr(df, ATR_LEN)
    df["VolSMA"] = df["Volume"].rolling(VOL_SMA).mean()
    last, prev = df.iloc[-1], df.iloc[-2]
    price = float(last["Close"])
    vol_ok = last["Volume"] >= (last["VolSMA"]*1.2 if not math.isnan(last["VolSMA"]) else 0)

    bull = (price > last["EMA20"] > last["EMA50"]) and \
           ((last["MACD"]>last["MACDsig"] and prev["MACD"]<=prev["MACDsig"]) or last["MACDhist"]>prev["MACDhist"]) and \
           (last["RSI"]>50) and vol_ok

    bear = (price < last["EMA20"] < last["EMA50"]) and \
           ((last["MACD"]<last["MACDsig"] and prev["MACD"]>=prev["MACDsig"]) or last["MACDhist"]<prev["MACDhist"]) and \
           (last["RSI"]<50) and vol_ok

    if bull:
        risk = float(last["ATR"])*1.5 if not math.isnan(last["ATR"]) else price*0.015
        stop = price - risk; tp = price + 2*risk
        return f"🟢买入 | 价:{price:.2f} 止损:{stop:.2f} 止盈:{tp:.2f}\n理由：价>EMA20>EMA50 + MACD转强 + RSI>{int(last['RSI'])} + 放量"
    if bear:
        risk = float(last["ATR"])*1.5 if not math.isnan(last["ATR"]) else price*0.015
        stop = price + risk; tp = price - 2*risk
        return f"🔴卖出 | 价:{price:.2f} 止损:{stop:.2f} 止盈:{tp:.2f}\n理由：价<EMA20<EMA50 + MACD转弱 + RSI<{int(last['RSI'])} + 放量"
    return f"⚪️观望 | 理由：条件未齐"

def main():
    if not within_trading_hours():
        send("⏱ 当前不在美股盘中（09:30–16:00 ET），本次不推送。")
        return
    lines=[]
    for t in TICKERS:
        try:
            lines.append(analyze_one(t))
        except Exception as e:
            lines.append(f"⚠️数据异常：{e}")
    send("5分钟信号\n" + "\n".join(lines))

if __name__=="__main__":
    main()
