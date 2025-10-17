# bot.py —— US Market Signals (stable)
# 依赖：yfinance pandas numpy pytz requests
# GitHub Actions Secrets：BOT_TOKEN / CHAT_ID
# 环境变量（可选）：RUN_WINDOW=regular|extended, STRATEGY=conservative|aggressive

import os, time, math, pytz, datetime as dt, requests
import numpy as np
import pandas as pd
import yfinance as yf

# ---------- 全局稳定性设置 ----------
os.environ.setdefault("YF_TZ_FIX", "1")   # 修复某些 runner 的时区/空数据问题
# yfinance 某些版本多线程会触发空数据，这里统一关掉
yf.shared._config.config.get("threads", False)

# ---------- Telegram ----------
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

# ---------- 标的 ----------
TICKERS = [
    "SPY","QQQ","AAPL","MSFT","NVDA","AMD","MU","TXN","NEE","JNJ",
    "TSLA","META","GOOGL","TSM","UNH"
]

# ---------- 策略参数 ----------
RSI_LEN, EMA_FAST, EMA_SLOW = 14, 20, 50
MACD_FAST, MACD_SLOW, MACD_SIG = 12, 26, 9
ATR_LEN, VOL_SMA = 14, 20
MIN_BARS_MAP = {"5m": 40, "15m": 24, "30m": 16, "1d": 20}
VOLUME_BOOST = 1.2

RUN_WINDOW = os.environ.get("RUN_WINDOW", "regular").lower()     # regular / extended
STRATEGY   = os.environ.get("STRATEGY", "conservative").lower()  # conservative / aggressive
eastern    = pytz.timezone("US/Eastern")

# ---------- 中文名称 ----------
NAME_MAP = {
    "SPY":"SPDR S&P 500 ETF（标普500）", "QQQ":"Invesco QQQ（纳指100）",
    "AAPL":"Apple Inc.（苹果）", "MSFT":"Microsoft（微软）", "NVDA":"NVIDIA（英伟达）",
    "AMD":"AMD（超微）", "MU":"Micron（美光）", "TXN":"Texas Instruments（德仪）",
    "NEE":"NextEra Energy（新纪元能源）", "JNJ":"Johnson & Johnson（强生）",
    "TSLA":"Tesla（特斯拉）", "META":"Meta（脸书）", "GOOGL":"Alphabet A（谷歌A）",
    "TSM":"TSMC（台积电）", "UNH":"UnitedHealth（联合健康）",
}
def company_name(tk): return NAME_MAP.get(tk.upper(), tk)

# ---------- 指标工具 ----------
def _series(x):
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            x = x.iloc[:,0]
        else:
            for col in ("Close","Adj Close","close","adjclose","price"):
                if col in x.columns:
                    x = x[col]; break
            else:
                x = x.iloc[:,0]
    s = pd.Series(x, index=getattr(x,"index",None), dtype="float64")
    return s.astype(float)

def ema(s, n): return _series(s).ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    s = _series(s); d = s.diff()
    up = np.where(d>0, d, 0.0); dn = np.where(d<0, -d, 0.0)
    ru = pd.Series(up, index=s.index).rolling(n, min_periods=n).mean()
    rd = pd.Series(dn, index=s.index).rolling(n, min_periods=n).mean()
    rs = ru/(rd+1e-9); return 100.0 - 100.0/(1.0+rs)

def macd(s):
    s = _series(s); ef = ema(s, MACD_FAST); es = ema(s, MACD_SLOW)
    line = ef - es; sig = line.ewm(span=MACD_SIG, adjust=False).mean()
    hist = line - sig; return line, sig, hist

def atr(df, n=14):
    h,l,c = _series(df["High"]), _series(df["Low"]), _series(df["Close"])
    pc = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def clean_df(df):
    if df is None or len(df)==0: return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    for col in ["Open","High","Low","Close","Adj Close","Volume"]:
        if col in df.columns: df[col] = _series(df[col])
    return df.dropna(how="any")

# ---------- 稳健下载（含备用节点/重试/降级） ----------
def download_bars(ticker: str, tries: int = 3):
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    intervals = [("5m","10d"), ("15m","30d"), ("30m","60d"), ("1d","200d")]
    proxies = [None, "https://query2.finance.yahoo.com"]  # 备用节点
    for interval, period in intervals:
        for pxy in proxies:
            for i in range(tries):
                try:
                    df = yf.download(
                        ticker, period=period, interval=interval,
                        prepost=True, auto_adjust=False, progress=False,
                        timeout=30, threads=False, repair=True,
                        proxy=pxy
                    )
                    if isinstance(df, pd.DataFrame) and len(df) > 0:
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = df.columns.get_level_values(0)
                        df = df.copy(); df["interval"]=interval
                        return df
                except Exception as e:
                    print(f"{ticker} {interval} try{i+1} proxy={pxy}: {e}")
                time.sleep(1.2)
    return pd.DataFrame()

# ---------- 时间窗 ----------
def within_window():
    now = dt.datetime.now(dt.timezone.utc).astimezone(eastern)
    if RUN_WINDOW=="regular":
        s = now.replace(hour=9,minute=30,second=0,microsecond=0)
        e = now.replace(hour=16,minute=0,second=0,microsecond=0)
        return s <= now <= e
    else:
        pre_s = now.replace(hour=7,minute=0,second=0,microsecond=0)
        pre_e = now.replace(hour=9,minute=30,second=0,microsecond=0)
        reg_s = now.replace(hour=9,minute=30,second=0,microsecond=0)
        reg_e = now.replace(hour=16,minute=0,second=0,microsecond=0)
        aft_s = now.replace(hour=16,minute=0,second=0,microsecond=0)
        aft_e = now.replace(hour=20,minute=0,second=0,microsecond=0)
        return (pre_s<=now<=pre_e) or (reg_s<=now<=reg_e) or (aft_s<=now<=aft_e)

def within_header(): return "常规盘" if RUN_WINDOW=="regular" else "盘前/常规/盘后"

# ---------- 策略 ----------
def signal_logic(price, ema20, ema50, rsi_v, macd_up, macd_dn, vol_ok):
    if STRATEGY=="aggressive":
        bull = (price>ema20) and macd_up and (rsi_v>48)
        bear = (price<ema20) and macd_dn and (rsi_v<52)
    else:
        bull = (price>ema20>ema50) and macd_up and (rsi_v>50) and vol_ok
        bear = (price<ema20<ema50) and macd_dn and (rsi_v<50) and vol_ok
    if bull: return "buy","强势" if rsi_v>60 else "普通"
    if bear: return "sell","强势" if rsi_v<40 else "普通"
    return "hold",None

def pos_advice(tag): return "重仓50%" if tag=="强势" else ("中仓30%" if tag=="普通" else "轻仓10%")

def strength_score(price, ema20, ema50, rsi_v, macd_hist, vol_ok, action):
    eps = 1e-9
    trend = (price-ema20)/max(price,eps) + (ema20-ema50)/max(price,eps)
    momentum = (rsi_v-50)/50.0 + np.tanh((macd_hist)/(abs(price)*0.001+eps))
    vol = 0.3 if vol_ok else 0.0
    s = trend + momentum + vol
    return -s if action=="sell" else s

# ---------- 单标的分析 ----------
def analyze_one(tk: str):
    cname = company_name(tk)
    df = download_bars(tk)
    if df.empty:
        return {"ticker":tk,"cname":cname,"action":"none","score":None,
                "text":f"⚠️数据源暂不可用（bars=0）"}

    interval = str(df.get("interval","?").iloc[0]) if "interval" in df else "?"
    df = clean_df(df)
    need = MIN_BARS_MAP.get(interval,24)
    if len(df) < need:
        return {"ticker":tk,"cname":cname,"action":"hold","score":None,
                "text":f"⚪️观望 | 理由：{interval} 数据不足（bars={len(df)}）"}

    # 指标
    df["EMA20"]=ema(df["Close"],EMA_FAST); df["EMA50"]=ema(df["Close"],EMA_SLOW)
    df["RSI"]=rsi(df["Close"],RSI_LEN)
    mline,msig,mhist = macd(df["Close"])
    df["MACD"]=mline; df["MACDsig"]=msig; df["MACDhist"]=mhist
    df["ATR"]=atr(df,ATR_LEN)
    df["VolSMA"]=_series(df["Volume"]).rolling(VOL_SMA, min_periods=5).mean()

    last, prev = df.iloc[-1], df.iloc[-2]
    price = float(last["Close"]); ema20=float(last["EMA20"]); ema50=float(last["EMA50"]); rsi_v=float(last["RSI"])
    macd_up = (last["MACD"]>last["MACDsig"]) and (prev["MACD"]<=prev["MACDsig"] or last["MACDhist"]>prev["MACDhist"])
    macd_dn = (last["MACD"]<last["MACDsig"]) and (prev["MACD"]>=prev["MACDsig"] or last["MACDhist"]<prev["MACDhist"])
    try: vol_ok = float(last["Volume"]) >= float(last["VolSMA"]) * VOLUME_BOOST
    except: vol_ok=False

    action, tag = signal_logic(price, ema20, ema50, rsi_v, macd_up, macd_dn, vol_ok)

    atr_v = float(last["ATR"]) if not math.isnan(float(last["ATR"])) else price*0.008
    risk  = max(atr_v*1.5, price*0.012)
    suffix = "" if interval in ("5m","15m","30m") else "（⚠️已降级为日线）"

    if action=="buy":
        stop, tp = price-risk, price+2*risk
        text=(f"🟢{'强势' if tag=='强势' else ''}买入{suffix} | {interval} | "
              f"价:{price:.2f} 止损:{stop:.2f} 止盈:{tp:.2f} | 仓位：{pos_advice(tag)}\n"
              f"理由：价/均线 {price:.2f}>{ema20:.2f}>{ema50:.2f} + MACD转强 + RSI:{int(rsi_v)}"
              f"{' + 放量' if vol_ok else ''}")
    elif action=="sell":
        stop, tp = price+risk, price-2*risk
        text=(f"🔴{'强势' if tag=='强势' else ''}卖出{suffix} | {interval} | "
              f"价:{price:.2f} 止损:{stop:.2f} 止盈:{tp:.2f} | 仓位：{pos_advice(tag)}\n"
              f"理由：价/均线 {price:.2f}<{ema20:.2f}<{ema50:.2f} + MACD转弱 + RSI:{int(rsi_v)}"
              f"{' + 放量' if vol_ok else ''}")
    else:
        text=(f"⚪️观望{suffix} | {interval} | "
              f"价:{price:.2f} EMA20:{ema20:.2f} EMA50:{ema50:.2f} RSI:{int(rsi_v)}")

    score = strength_score(price, ema20, ema50, rsi_v, float(last["MACDhist"]), vol_ok, action) \
            if action in ("buy","sell") else None

    return {"ticker":tk,"cname":cname,"text":text,"action":action,
            "score":score,"interval":interval}

# ---------- 摘要 ----------
def build_summary(results):
    buys  = [r for r in results if r["action"]=="buy"  and r["score"] is not None]
    sells = [r for r in results if r["action"]=="sell" and r["score"] is not None]
    best_buy  = max(buys,  key=lambda r: r["score"]) if buys  else None
    best_sell = min(sells, key=lambda r: r["score"]) if sells else None
    if not best_buy and not best_sell:
        return "今日最强/最弱：暂无明确强信号（以观望为主）"
    parts=[]
    if best_buy:
        parts.append(f"最强买入：{best_buy['ticker']}/{best_buy['cname']}（{best_buy['interval']} | 评分:{best_buy['score']:.2f}）")
    if best_sell:
        parts.append(f"最强卖出：{best_sell['ticker']}/{best_sell['cname']}（{best_sell['interval']} | 评分:{best_sell['score']:.2f}）")
    return "；".join(parts)

# ---------- 主流程 ----------
def main():
    if not within_window():
        send(f"⏱ 当前不在设定时间窗（{within_header()}），本次不推送。"); return

    results, lines = [], []
    for tk in TICKERS:
        try:
            r = analyze_one(tk)
        except Exception as e:
            r = {"ticker":tk,"cname":company_name(tk),"text":f"⚠️数据异常：{e}",
                 "action":"none","score":None}
        results.append(r); lines.append(r["text"])

    summary = build_summary(results)
    header  = f"5分钟信号（策略：{STRATEGY}，窗口：{RUN_WINDOW}）\n🏁 {summary}"
    send(header + "\n" + "\n".join(lines))

if __name__ == "__main__":
    main()
