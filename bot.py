import time
import pandas as pd
import numpy as np
import telebot
from pybit.unified_trading import HTTP
import os

# Переменные из окружения
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
TG_TOKEN = os.getenv("TELEGRAM_TOKEN")
TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

SYMBOL = 'DOGEUSDT'
TRADE_AMOUNT = 100  # USD
TAKE_PROFIT_PERCENT = 0.20  # 20%
ENTRY_THRESHOLD = 0.0040  # 40 пунктов

bot = telebot.TeleBot(TG_TOKEN)
session = HTTP(api_key=API_KEY, api_secret=API_SECRET)

def send_telegram(msg):
    bot.send_message(TG_CHAT_ID, msg)

def get_klines(symbol, interval, limit):
    res = session.get_kline(category="linear", symbol=symbol, interval=interval, limit=limit)
    return res['result']['list']

def analyze_market():
    try:
        # Получение 6 месяцев данных (примерно 1080 свечей по 4H)
        candles_4h = get_klines(SYMBOL, '240', 1080)
        df_4h = pd.DataFrame(candles_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', '_'])
        df_4h['close'] = df_4h['close'].astype(float)
        df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'], unit='ms')

        # EMA & RSI
        df_4h['ema20'] = df_4h['close'].ewm(span=20).mean()
        df_4h['ema50'] = df_4h['close'].ewm(span=50).mean()
        delta = df_4h['close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(14).mean()
        avg_loss = pd.Series(loss).rolling(14).mean()
        rs = avg_gain / avg_loss
        df_4h['rsi'] = 100 - (100 / (1 + rs))

        trend = "bullish" if df_4h.iloc[-1]['ema20'] > df_4h.iloc[-1]['ema50'] else "bearish"

        # Анализ на 15M для входа
        candles_15m = get_klines(SYMBOL, '15', 96)
        df_15m = pd.DataFrame(candles_15m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', '_'])
        df_15m['close'] = df_15m['close'].astype(float)
        last_close = df_15m.iloc[-1]['close']
        prev_close = df_15m.iloc[-2]['close']

        price_change = last_close - prev_close

        signal = None
        if abs(price_change) >= ENTRY_THRESHOLD:
            if price_change > 0 and trend == 'bullish':
                signal = 'long'
            elif price_change < 0 and trend == 'bearish':
                signal = 'short'

        return signal, last_close, trend

    except Exception as e:
        send_telegram(f"Ошибка анализа рынка: {e}")
        return None, None, None

def place_order(side, entry_price):
    take_profit_price = entry_price * (1 + TAKE_PROFIT_PERCENT) if side == 'Buy' else entry_price * (1 - TAKE_PROFIT_PERCENT)
    try:
        order = session.place_order(
            category="linear",
            symbol=SYMBOL,
            side=side,
            orderType="Market",
            qty=TRADE_AMOUNT / entry_price,
            timeInForce="GoodTillCancel",
            reduceOnly=False
        )
        send_telegram(f"Открыта позиция {side} по цене {entry_price:.4f}\nТейк-профит: {take_profit_price:.4f}")
    except Exception as e:
        send_telegram(f"Ошибка при открытии позиции: {e}")

def main():
    send_telegram("Бот запущен и ждёт сигналы...")
    while True:
        signal, price, trend = analyze_market()
        if signal:
            send_telegram(f"Анализ 4H: тренд {trend}.\n15M сигнал на вход: {signal.upper()} по цене {price:.4f}")
            place_order('Buy' if signal == 'long' else 'Sell', price)
        else:
            send_telegram("Сигналов не найдено. Ждём следующего цикла.")
        time.sleep(4 * 3600)

if __name__ == "__main__":
    main()
