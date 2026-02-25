import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# =======================
# KONFIGURASI TETAP
# =======================
import os

TELEGRAM_TOKEN = os.getenv("BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("CHAT_ID")
CSV_URL = "https://docs.google.com/spreadsheets/d/1iZ2Iny1GsaZpmQHkEuXXVg8qzq4WSa-LrcW3SeKzvAw/export?format=csv&gid=0"

# =======================
# PARAMETER ADAPTIF (selaras Pine)
# =======================
VOL_LEN = 20
VOL_MULT_SPIKE = 5.0
VOL_MULT_BREAK = 1.2
ADX_LEN = 14
MIN_GAP = 50  # minimal jarak antar sinyal agar dianggap fresh

# =======================
# FILTER NILAI TRANSAKSI
# =======================
VALUE_THRESHOLD = 100_000_000_000  # 100 milyar rupiah

# =======================
# UTIL
# =======================
def send_telegram(msg: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}
    try:
        r = requests.post(url, data=data, timeout=15)
        if r.status_code != 200:
            print(f"[ERROR] Telegram error: {r.text}")
    except Exception as e:
        print(f"[ERROR] Telegram exception: {e}")


def get_fraksi_harga(h):
    if h < 200:
        return 1
    elif h < 500:
        return 2
    elif h < 2000:
        return 5
    elif h < 5000:
        return 10
    else:
        return 25


# TradingView RMA (Wilders)
def rma(series: pd.Series, length: int) -> pd.Series:
    # EWM alpha = 1/length (Wilders smoothing)
    return series.ewm(alpha=1.0 / length, adjust=False).mean()


# =======================
# INDIKATOR
# =======================
def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize columns if MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ensure required columns exist
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if c not in df.columns:
            raise ValueError(f"Missing column {c}")

    # Convert to numeric and drop rows with NaNs in core cols
    df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[
        ['Open', 'High', 'Low', 'Close', 'Volume']
    ].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume']).copy()

    close = df['Close']
    high = df['High']
    low = df['Low']
    vol = df['Volume']

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = rma(gain, 14)
    avg_loss = rma(loss, 14)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))

    # ATR(14)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    df['atr14'] = rma(tr, 14)

    # SMA
    df['ma20'] = close.rolling(20).mean()
    df['ma50'] = close.rolling(50).mean()
    df['ma200'] = close.rolling(200).mean()

    # Volume adaptif
    df['volMA'] = vol.rolling(VOL_LEN).mean()
    df['volSpike'] = vol > (df['volMA'] * VOL_MULT_SPIKE)
    df['volConfirm'] = df['volSpike'] | (vol > (df['volMA'] * VOL_MULT_BREAK))

    # Donchian & breakout (prev 5 closes)
    df['highest_close_prev5'] = close.shift(1).rolling(5).max()
    df['donHigh20'] = high.shift(1).rolling(20).max()
    df['validBreak'] = (close > df['highest_close_prev5']) & (vol > (df['volMA'] * VOL_MULT_BREAK))

    # Range lesu
    rng5 = high.rolling(5).max() - low.rolling(5).min()
    fraksi = close.apply(get_fraksi_harga)
    df['notIsLesu'] = rng5 >= (fraksi * 10)

    # ADX(14)
    upMove = high.diff()
    downMove = -low.diff()
    plusDM = np.where((upMove > downMove) & (upMove > 0), upMove, 0.0)
    minusDM = np.where((downMove > upMove) & (downMove > 0), downMove, 0.0)

    plusDI = 100 * rma(pd.Series(plusDM, index=df.index), ADX_LEN) / rma(tr, ADX_LEN)
    minusDI = 100 * rma(pd.Series(minusDM, index=df.index), ADX_LEN) / rma(tr, ADX_LEN)
    dx = 100 * (plusDI - minusDI).abs() / (plusDI + minusDI).replace(0, np.nan)
    df['adx'] = rma(dx, ADX_LEN)
    df['strongTrend'] = df['adx'] > 20

    return df


# =======================
# FUNGSI GAIN & STATUS
# =======================
def calc_gain_status(row, prev_close):
    if pd.isna(prev_close) or prev_close == 0:
        return 0.0, ""
    gain_pct = ((row['Close'] - prev_close) / prev_close) * 100
    status = "🟩Safe" if gain_pct <= 15 else ""
    return gain_pct, status


# =======================
# MESIN SINYAL (Daily / Weekly)
# =======================
def run_signal_engine(df: pd.DataFrame, timeframe: str):
    events = []
    last_signal_pos = None
    lastFL_pos = None
    lastDC_pos = None
    LOOKBACK_TREND = 50

    # iterate rows
    for idx_pos, (idx, row) in enumerate(df.iterrows()):
        volOk = (not pd.isna(row.get('volMA'))) and (row['Volume'] > row['volMA'])
        volSpike = bool(row.get('volSpike', False))
        volConfirm = bool(row.get('volConfirm', False))
        notIsLesu = bool(row.get('notIsLesu', False))
        validBreak = bool(row.get('validBreak', False))

        # minniBreak ~ validBreak equivalent (keamanan)
        minniBreak = False
        if 'highest_close_prev5' in row.index and not pd.isna(row['highest_close_prev5']):
            minniBreak = row['Close'] > row['highest_close_prev5']

        # BUY FL
        buyFL = (
            volOk
            and pd.notna(row.get('ma50')) and pd.notna(row.get('ma200'))
            and (row['ma50'] > row['ma200'])
            and row['strongTrend']
            and (row['Close'] > row['Open'])
            and pd.notna(row.get('rsi')) and (row['rsi'] <= 79)
            and (volSpike or notIsLesu)
            and pd.notna(row.get('ma20')) and (row['Close'] > row['ma20'])
            and minniBreak and validBreak
        )

        # BUY DC
        buyDC = (
            pd.notna(row.get('donHigh20', np.nan))
            and (row['Close'] > row['donHigh20'])
            and volConfirm
            and pd.notna(row.get('rsi')) and (row['rsi'] <= 78)
            and (row['Close'] >= row['Open'])
            and row['strongTrend']
            and pd.notna(row.get('ma200')) and (row['Close'] > row['ma200'])
            and validBreak and notIsLesu
        )

        if buyFL:
            lastFL_pos = idx_pos
        if buyDC:
            lastDC_pos = idx_pos

        validTrend = (
            (lastFL_pos is not None and (idx_pos - lastFL_pos) <= LOOKBACK_TREND)
            or (lastDC_pos is not None and (idx_pos - lastDC_pos) <= LOOKBACK_TREND)
        )

        # BUY SP
        buySP = (
            pd.notna(row.get('ma50'))
            and pd.notna(row.get('ma20'))
            and (row['Close'] > row['ma50'])
            and (row['ma20'] > row['ma50'])
            and pd.notna(row.get('rsi')) and (row['rsi'] < 55)
            and (row['Close'] > row['Open'])
            and (row['Volume'] > (row.get('volMA', np.nan) * 0.8))
            and validTrend
        )

        # === JIKA ADA SINYAL ===
        if buyFL or buyDC or buySP:
            is_fresh = (last_signal_pos is None) or ((idx_pos - last_signal_pos) >= MIN_GAP)
            last_signal_pos = idx_pos

            prev_close = df['Close'].iloc[idx_pos - 1] if idx_pos > 0 else np.nan
            gain_pct, ok_status = calc_gain_status(row, prev_close)
            fresh_tag = "🟩Fresh" if is_fresh else ""
            status_line = (ok_status + " " + fresh_tag).strip()

            strategies = []
            if buyFL:
                strategies.append("FL")
            if buyDC:
                strategies.append("DC")
            if buySP:
                strategies.append("SP")
            strategy = " | ".join(strategies) if strategies else None

            if strategy:
                events.append({
                    "i": idx_pos,
                    "ts": idx,
                    "type": "BUY",
                    "strategy": strategy,
                    "status": status_line,
                    "gain_pct": gain_pct,
                    "timeframe": timeframe,
                })

    return events


# =======================
# MAIN
# =======================
def main():
    print("Mengambil daftar ticker dari Google Sheets...")
    try:
        tickers = pd.read_csv(CSV_URL)['ticker'].dropna().tolist()
        print("Daftar ticker:", tickers)
    except Exception as e:
        print("[ERROR] baca CSV:", e)
        tickers = []

    if not tickers:
        print("Ticker kosong")
        return

    any_signal = False

    for tk in tickers:
        print(f"\nMemproses ticker {tk} ...")
        try:
            # === DAILY ===
            df_daily = yf.download(tk, period="12mo", interval="1d", auto_adjust=False, progress=False, threads=True)
            if not df_daily.empty:
                df_daily = calc_indicators(df_daily)
                if len(df_daily) >= 200:
                    events_daily = run_signal_engine(df_daily, "Daily")
                    last_i = len(df_daily) - 1
                    last_ts = df_daily.index[last_i]
                    price = round(df_daily['Close'].iloc[-1])

                    # hitung nilai transaksi bar terakhir (daily)
                    try:
                        total_value_daily = float(df_daily['Close'].iloc[-1]) * float(df_daily['Volume'].iloc[-1])
                    except Exception:
                        total_value_daily = 0.0

                    for ev in [e for e in events_daily if e['i'] == last_i]:
                        # cek filter nilai transaksi > threshold
                        if total_value_daily > VALUE_THRESHOLD:
                            any_signal = True
                            miliar = total_value_daily / 1e9
                            msg = (
                                f"{tk} - {last_ts.date()} - {ev['strategy']} [{ev['timeframe']}]\n"
                                f"Harga: {price} | {ev['status']}\nNilai transaksi: {miliar:.2f} M"
                            )
                            print("KIRIM:", msg)
                            send_telegram(msg)
                        else:
                            miliar = total_value_daily / 1e9
                            print(f"SKIP (nilai transaksi < 100M): {tk} {last_ts.date()} nilai={miliar:.2f} M")

            # === WEEKLY ===
            df_weekly = yf.download(tk, period="3y", interval="1wk", auto_adjust=False, progress=False, threads=True)
            if not df_weekly.empty:
                df_weekly = calc_indicators(df_weekly)
                if len(df_weekly) >= 100:
                    events_weekly = run_signal_engine(df_weekly, "Weekly")
                    last_i = len(df_weekly) - 1
                    last_ts = df_weekly.index[last_i]
                    price = round(df_weekly['Close'].iloc[-1])

                    # hitung nilai transaksi bar terakhir (weekly)
                    try:
                        total_value_weekly = float(df_weekly['Close'].iloc[-1]) * float(df_weekly['Volume'].iloc[-1])
                    except Exception:
                        total_value_weekly = 0.0

                    for ev in [e for e in events_weekly if e['i'] == last_i]:
                        if total_value_weekly > VALUE_THRESHOLD:
                            any_signal = True
                            miliar = total_value_weekly / 1e9
                            msg = (
                                f"{tk} - {last_ts.date()} - {ev['strategy']} [{ev['timeframe']}]\n"
                                f"Harga: {price} | {ev['status']}\nNilai transaksi: {miliar:.2f} M"
                            )
                            print("KIRIM:", msg)
                            send_telegram(msg)
                        else:
                            miliar = total_value_weekly / 1e9
                            print(f"SKIP (nilai transaksi < 100M): {tk} {last_ts.date()} nilai={miliar:.2f} M")

            if not any_signal:
                print(f"{tk}: tidak ada sinyal BUY")

        except Exception as e:
            print(f"[ERROR] Saat proses {tk}: {e}")

    if not any_signal:
        print("Tidak ada sinyal buy")
        send_telegram("Tidak ada sinyal buy")


if __name__ == "__main__":
    main()



