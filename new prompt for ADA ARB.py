#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyQt5 Cryptocurrency Trading App (Binance via CCXT)
- Real-time Trading (5m timeframe) with per-pair QThreads
- Backtesting with the same strategy engine
- Pairs populated from pairs.txt (one pair per line)
- API keys loaded from .env (BINANCE_API_KEY, BINANCE_API_SECRET)

Upgrades applied:
1) **Backtesting pagination**: fetches full historical span for requested days.
2) **Real-time incremental updates**: initial bulk load once; then append only new 5m candles.
3) **DRY strategy**: unified Strategy class returning BUY/SELL/HOLD decisions.
4) **Logging**: Python logging + in-app QTextEdit log pane; more specific ccxt error handling.
5) **Realistic Backtesting**: 0.1% trading fee simulation.
6) **Duplicate Prevention**: UI prevents running the same symbol in multiple slots.
7) **Enhanced UI**: Active rows have their inputs disabled.
8) **Detailed Log Summary**: Backtest results are shown in a detailed, single-line format.
"""

import os
import sys
import time
import math
import traceback
import logging
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow, QTabWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QDoubleSpinBox, QSpinBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QCheckBox, QTextEdit, QSplitter
)

# --- Third-party exchange lib ---
try:
    import ccxt
except Exception as e:
    print("Error: ccxt not installed. Please: pip install ccxt")
    raise


# ===================== Strategy Parameters (Default) =====================
TIMEFRAME = "5m"  # fixed for the entire app
RISK_PER_TRADE = 0.10      # 10% of slot capital
STOP_LOSS_PCT = 0.02       # 2% SL
SMA_FAST = 50
SMA_SLOW = 200
CONSEC_BULL = 2
RSI_PERIOD = 14
RSI_THRESHOLD = 60
TRAILING_STOP_PCT = 0.03
BINANCE_FEE = 0.001 # 0.1% fee for SPOT trading

# ========================================================================
# Logging setup
# ========================================================================
logger = logging.getLogger("crypto_trader")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


# ========================================================================
# Helpers
# ========================================================================

def load_pairs(file_path: str) -> List[str]:
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r') as f:
        pairs = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith('#')]
    return pairs


def ms_to_min5_boundary_delay(now_ms: int) -> float:
    """Return seconds until the next 5-minute candle CLOSE."""
    now_s = now_ms / 1000.0
    next_boundary = math.ceil(now_s / 300.0) * 300.0  # 5m = 300s
    delay = max(0.0, next_boundary - now_s)
    return delay + 1.0  # cushion


def ohlcv_to_df(ohlcv: List[List[float]]) -> pd.DataFrame:
    df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['sma_fast'] = df['close'].rolling(SMA_FAST).mean()
    df['sma_slow'] = df['close'].rolling(SMA_SLOW).mean()
    delta = df['close'].diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain, index=df.index).rolling(RSI_PERIOD).mean()
    avg_loss = pd.Series(loss, index=df.index).rolling(RSI_PERIOD).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    return df


def precision_amount(exchange, symbol: str, amount: float) -> float:
    try:
        return float(exchange.amount_to_precision(symbol, amount))
    except Exception:
        return amount


def precision_price(exchange, symbol: str, price: float) -> float:
    try:
        return float(exchange.price_to_precision(symbol, price))
    except Exception:
        return price


@dataclass
class Position:
    symbol: str
    size: float
    entry_price: float
    stop_loss: float
    trailing_stop: float
    invested: float


# ========================================================================
# Exchange Wrapper
# ========================================================================
class ExchangeWrapper:
    def __init__(self, paper_trade: bool = True):
        load_dotenv()
        api_key = os.getenv('BINANCE_API_KEY', '')
        api_secret = os.getenv('BINANCE_API_SECRET', '')
        self.paper = paper_trade
        self.client = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
            }
        })

    @property
    def rate_limit_s(self) -> float:
        try:
            return float(self.client.rateLimit) / 1000.0
        except Exception:
            return 0.2

    def fetch_ohlcv(self, symbol: str, timeframe: str, since: Optional[int] = None, limit: int = 1000) -> List[List[float]]:
        return self.client.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)

    def fetch_ticker_price(self, symbol: str) -> float:
        t = self.client.fetch_ticker(symbol)
        return float(t.get('last') or t.get('close'))

    def market_buy(self, symbol: str, amount: float) -> Dict:
        if self.paper:
            return {"info": {"paper": True}, "symbol": symbol, "amount": amount}
        return self.client.create_order(symbol, type='market', side='buy', amount=amount)

    def market_sell(self, symbol: str, amount: float) -> Dict:
        if self.paper:
            return {"info": {"paper": True}, "symbol": symbol, "amount": amount}
        return self.client.create_order(symbol, type='market', side='sell', amount=amount)


# ========================================================================
# Unified Strategy (Corrected to match original logic)
# ========================================================================
class Strategy:
    """Returns (action, details)"""

    def _get_running_bullish_count(self, df: pd.DataFrame, current_index: int) -> int:
        """Calculates the uninterrupted sequence of bullish candles ending at the current index."""
        count = 0
        for i in range(current_index, -1, -1):
            if i == 0: break
            row = df.iloc[i]
            if row['close'] > row['open']:
                count += 1
            else:
                break # Stop counting as soon as a non-bullish candle is found
        return count

    def decide(self, row: pd.Series, df: pd.DataFrame, position: Optional[Position]) -> Tuple[str, Dict]:
        i = row.name
        is_bullish = row['close'] > row['open']
        is_bearish = row['close'] < row['open']

        if position is None:
            # Use the corrected running count logic
            bullish_count = self._get_running_bullish_count(df, i)

            strong_trend = row['sma_fast'] > row['sma_slow']
            strong_momentum = row['rsi'] > RSI_THRESHOLD
            strong_candle = row['close'] > row['open'] + 0.75 * (row['high'] - row['low'])
            if (
                bullish_count >= CONSEC_BULL and
                row['close'] > row['sma_fast'] and
                strong_trend and strong_momentum and strong_candle
            ):
                stop_loss_price = float(row['close']) * (1 - STOP_LOSS_PCT)
                return "BUY", {"stop_loss": stop_loss_price}
            return "HOLD", {}

        # position is open -> check exits (This logic was already correct)
        details = {}
        if row['low'] <= position.stop_loss:
            details = {"sell_price": position.stop_loss, "reason": "stop_loss"}
            return "SELL", details
        if row['low'] <= position.trailing_stop:
            details = {"sell_price": position.trailing_stop, "reason": "trailing_stop"}
            return "SELL", details
        if is_bearish and row['close'] < row['sma_fast']:
            details = {"sell_price": float(row['close']), "reason": "bearish_exit"}
            return "SELL", details
        return "HOLD", {}


# ========================================================================
# Worker Base (common signals)
# ========================================================================
class BaseWorker(QThread):
    update_status = pyqtSignal(str)
    update_live = pyqtSignal(str, float, float)
    finished_pair = pyqtSignal(str)

    def __init__(self, symbol: str, capital: float, ex: ExchangeWrapper):
        super().__init__()
        self.symbol = symbol
        self.initial_capital = float(capital)
        self.exchange = ex
        self._running = True
        self.position: Optional[Position] = None
        self.trades: List[Dict] = []
        self.capital = float(capital)
        self.strategy = Strategy()

    def stop(self):
        self._running = False

    def _emit_live(self):
        pnl = 0.0
        invested = 0.0
        try:
            if self.position:
                current_price = self.exchange.fetch_ticker_price(self.symbol)
                pnl = (current_price - self.position.entry_price) * self.position.size
                invested = self.position.invested
        except Exception as e:
            logger.warning(f"Live price fetch failed for {self.symbol}: {e}")
        self.update_live.emit(self.symbol, pnl, invested)


# ========================================================================
# Real-time Trading Worker
# ========================================================================
class RealtimeWorker(BaseWorker):
    def run(self):
        try:
            self.update_status.emit(f"[RT] Starting {self.symbol} with capital {self.initial_capital:.2f} USDT")
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, TIMEFRAME, limit=max(1000, SMA_SLOW+5))
            df = compute_indicators(ohlcv_to_df(ohlcv))

            while self._running:
                now_ms = int(self.exchange.client.milliseconds())
                delay = ms_to_min5_boundary_delay(now_ms)
                slept = 0
                while self._running and slept < delay:
                    time.sleep(1)
                    slept += 1
                if not self._running:
                    break

                try:
                    latest_candles = self.exchange.fetch_ohlcv(self.symbol, TIMEFRAME, limit=2)
                    new_df = ohlcv_to_df(latest_candles)
                    last_ts = df['timestamp'].iloc[-1]
                    to_append = new_df[new_df['timestamp'] > last_ts]
                    if not to_append.empty:
                        df = pd.concat([df, to_append], ignore_index=True)
                        df = compute_indicators(df)

                    i = len(df) - 1
                    if i <= SMA_SLOW:
                        continue
                    row = df.iloc[i]

                    if self.position:
                        price_now = self.exchange.fetch_ticker_price(self.symbol)
                        self.position.trailing_stop = max(self.position.trailing_stop, price_now * (1 - TRAILING_STOP_PCT))

                    action, details = self.strategy.decide(row, df, self.position)
                    if action == "BUY" and self.position is None:
                        risk_amount = self.capital * RISK_PER_TRADE
                        stop_loss_price = float(details["stop_loss"])
                        if row['close'] - stop_loss_price <= 0:
                            continue
                        raw_size = risk_amount / (row['close'] - stop_loss_price)
                        size = precision_amount(self.exchange.client, self.symbol, raw_size)
                        if size <= 0:
                            continue
                        price = self.exchange.fetch_ticker_price(self.symbol)
                        try:
                            self.exchange.market_buy(self.symbol, size)
                        except ccxt.AuthenticationError:
                            self.update_status.emit(f"ERROR: Invalid API Keys for {self.symbol}. Check .env.")
                            break
                        except ccxt.NetworkError:
                            self.update_status.emit(f"ERROR: Network issue while buying {self.symbol}.")
                            continue
                        invested = price * size
                        trailing_stop = price * (1 - TRAILING_STOP_PCT)
                        self.position = Position(self.symbol, size, price, stop_loss_price, trailing_stop, invested)
                        self.capital -= invested
                        self.update_status.emit(f"[RT] BUY {self.symbol} size={size:.6f} @ {price:.6f} SL={stop_loss_price:.6f}")

                    elif action == "SELL" and self.position:
                        sell_price = float(details.get("sell_price", row['close']))
                        reason = details.get("reason", "exit")
                        size = self.position.size
                        try:
                            self.exchange.market_sell(self.symbol, size)
                        except ccxt.AuthenticationError:
                            self.update_status.emit(f"ERROR: Invalid API Keys for {self.symbol}. Check .env.")
                            break
                        except ccxt.NetworkError:
                            self.update_status.emit(f"ERROR: Network issue while selling {self.symbol}.")
                            continue
                        pnl = (sell_price - self.position.entry_price) * size
                        self.capital += sell_price * size
                        self.trades.append({'symbol': self.symbol, 'pnl': pnl, 'type': reason})
                        self.update_status.emit(f"[RT] SELL {self.symbol} size={size:.6f} @ {sell_price:.6f} ({reason}) PnL={pnl:.4f}")
                        self.position = None

                    self._emit_live()

                except ccxt.NetworkError as e:
                    self.update_status.emit(f"[RT] Network error {self.symbol}: {e}")
                    time.sleep(self.exchange.rate_limit_s)
                except Exception as e:
                    tb = traceback.format_exc()
                    self.update_status.emit(f"[RT] Loop error {self.symbol}: {e}\n{tb}")
                    time.sleep(self.exchange.rate_limit_s)

            if self.position:
                try:
                    price = self.exchange.fetch_ticker_price(self.symbol)
                    self.exchange.market_sell(self.symbol, self.position.size)
                    pnl = (price - self.position.entry_price) * self.position.size
                    self.update_status.emit(f"[RT] FORCE-SELL {self.symbol} size={self.position.size:.6f} @ {price:.6f} PnL={pnl:.4f}")
                except Exception as e:
                    self.update_status.emit(f"[RT] Error force-closing {self.symbol}: {e}")
                finally:
                    self.position = None
                    self._emit_live()

        except Exception as e:
            tb = traceback.format_exc()
            self.update_status.emit(f"[RT] Worker error {self.symbol}: {e}\n{tb}")
        finally:
            self.finished_pair.emit(self.symbol)


# ========================================================================
# Backtesting Worker
# ========================================================================
class BacktestWorker(BaseWorker):
    update_result = pyqtSignal(str, float, float)

    def __init__(self, symbol: str, capital: float, ex: ExchangeWrapper, lookback_days: int):
        super().__init__(symbol, capital, ex)
        self.lookback_days = int(lookback_days)

    def run(self):
        try:
            self.update_status.emit(f"[BT] Fetching data {self.symbol} for {self.lookback_days} days...")
            now_ms = int(self.exchange.client.milliseconds())
            since_ms = now_ms - self.lookback_days * 24 * 60 * 60 * 1000

            all_ohlcv: List[List[float]] = []
            since = since_ms
            while self._running:
                batch = self.exchange.fetch_ohlcv(self.symbol, TIMEFRAME, since=since, limit=1000)
                if not batch: break
                all_ohlcv.extend(batch)
                since = batch[-1][0] + 1
                if len(batch) < 1000: break
                time.sleep(self.exchange.rate_limit_s)

            if not all_ohlcv:
                self.update_status.emit(f"[BT] No data found for {self.symbol}")
                return

            df = compute_indicators(ohlcv_to_df(all_ohlcv))
            
            capital = self.initial_capital
            position: Optional[Position] = None
            trades: List[Dict] = []

            for i in range(SMA_SLOW, len(df)):
                if not self._running: break
                row = df.iloc[i]

                if position is not None:
                    position.trailing_stop = max(position.trailing_stop, float(row['close']) * (1 - TRAILING_STOP_PCT))

                action, details = self.strategy.decide(row, df, position)
                if action == "BUY" and position is None:
                    risk_amount = capital * RISK_PER_TRADE
                    stop_loss_price = float(details["stop_loss"])
                    if row['close'] - stop_loss_price <= 0: continue
                    
                    size = risk_amount / (row['close'] - stop_loss_price)
                    invested = float(row['close']) * size
                    size *= (1 - BINANCE_FEE) # Apply fee
                    
                    trades.append({'type': 'BUY', 'price': row['close'], 'size': size})
                    position = Position(self.symbol, size, float(row['close']), stop_loss_price, float(row['close'])*(1-TRAILING_STOP_PCT), invested)
                    capital -= invested

                elif action == "SELL" and position is not None:
                    sell_price = float(details.get("sell_price", row['close']))
                    gross_proceeds = sell_price * position.size
                    net_proceeds = gross_proceeds * (1 - BINANCE_FEE)
                    pnl = net_proceeds - position.invested
                    capital += net_proceeds
                    trades.append({'type': 'SELL', 'price': sell_price, 'pnl': pnl})
                    position = None

            if position is not None:
                last_close = float(df.iloc[-1]['close'])
                gross_proceeds = last_close * position.size
                net_proceeds = gross_proceeds * (1 - BINANCE_FEE)
                pnl = net_proceeds - position.invested
                trades.append({'type': 'SELL', 'price': last_close, 'pnl': pnl})

            # Emit final results
            invested_amount = self.initial_capital
            total_pnl = float(sum(t.get('pnl', 0.0) for t in trades))
            self.update_result.emit(self.symbol, total_pnl, invested_amount)

            # Calculate and emit detailed stats
            num_trades = len(trades)
            num_buys = sum(1 for t in trades if t['type'] == 'BUY')
            num_sells = sum(1 for t in trades if t['type'] == 'SELL')
            win_trades = sum(1 for t in trades if t.get('pnl', 0.0) > 0)
            win_pct = (win_trades / num_sells * 100) if num_sells else 0.0
            roi = (total_pnl / invested_amount * 100) if invested_amount else 0.0

            self.update_status.emit(
                f"[BT] Done {self.symbol}: "
                f"Invested {invested_amount:.2f}, "
                f"PnL {total_pnl:.2f}, "
                f"Trades {num_trades}, "
                f"Buys {num_buys}, "
                f"Sells {num_sells}, "
                f"Win% {win_pct:.2f}%, "
                f"ROI {roi:.2f}%"
            )

        except (ccxt.AuthenticationError, ccxt.NetworkError) as e:
            self.update_status.emit(f"ERROR: {e.__class__.__name__} for {self.symbol}. Check keys/connection.")
        except Exception as e:
            tb = traceback.format_exc()
            self.update_status.emit(f"[BT] Worker error {self.symbol}: {e}\n{tb}")
        finally:
            self.finished_pair.emit(self.symbol)


# ========================================================================
# UI Components
# ========================================================================
class PairRow(QWidget):
    start_clicked = pyqtSignal(int, str, float)
    stop_clicked = pyqtSignal(int)

    def __init__(self, pairs: List[str], index: int, parent=None):
        super().__init__(parent)
        self.index = index
        layout = QHBoxLayout(self)

        self.combo = QComboBox()
        self.combo.addItems(pairs)

        self.amount = QDoubleSpinBox()
        self.amount.setDecimals(2)
        self.amount.setRange(0.0, 1_000_000.0)
        self.amount.setValue(100.0)

        self.btn = QPushButton("Start")
        self.btn.clicked.connect(self._toggle)

        layout.addWidget(QLabel(f"Slot {index+1}"))
        layout.addWidget(self.combo, 2)
        layout.addWidget(QLabel("USDT:"))
        layout.addWidget(self.amount)
        layout.addWidget(self.btn)
        self.setLayout(layout)
        self.running = False

    def _toggle(self):
        if not self.running:
            symbol = self.combo.currentText().strip()
            capital = float(self.amount.value())
            if not symbol or capital <= 0:
                QMessageBox.warning(self, "Invalid", "Please choose a symbol and amount > 0")
                return
            self.running = True
            self.btn.setText("Stop")
            self.set_controls_enabled(False)
            self.start_clicked.emit(self.index, symbol, capital)
        else:
            self.running = False
            self.btn.setText("Start")
            self.set_controls_enabled(True)
            self.stop_clicked.emit(self.index)

    def reset(self):
        self.running = False
        self.btn.setText("Start")
        self.set_controls_enabled(True)

    def set_controls_enabled(self, enabled: bool):
        self.combo.setEnabled(enabled)
        self.amount.setEnabled(enabled)


class BaseTab(QWidget):
    log_signal = pyqtSignal(str)

    def __init__(self, pairs: List[str]):
        super().__init__()
        self.pairs = pairs
        self.workers: Dict[int, QThread] = {}
        self.symbol_rows: Dict[str, int] = {}
        self.rows: List[PairRow] = []

    def _log(self, msg: str):
        logger.info(msg)
        self.log_signal.emit(msg)

    def _start_all(self):
        for row in self.rows:
            if not row.running and row.amount.value() > 0:
                row._toggle()

    def _stop_all(self):
        for w in list(self.workers.values()):
            w.stop()
        for row in self.rows:
            if row.running:
                row.reset()

    def start_row(self, idx: int, symbol: str, capital: float):
        for worker in self.workers.values():
            if worker.symbol == symbol:
                QMessageBox.warning(self, "Duplicate Symbol", f"A trade for {symbol} is already active.")
                self.rows[idx].reset()
                return
        if idx in self.workers:
            QMessageBox.warning(self, "Running", f"Slot {idx+1} already running")
            return
        # Implemented by subclasses
        pass

    def stop_row(self, idx: int):
        w = self.workers.get(idx)
        if w: w.stop()

    def _finished_pair(self, symbol: str):
        idx = self.symbol_rows.pop(symbol, None)
        if idx is not None:
            self.workers.pop(idx, None)
            self.rows[idx].reset()


class RealTimeTab(BaseTab):
    def __init__(self, pairs: List[str]):
        super().__init__(pairs)
        layout = QVBoxLayout(self)

        top_layout = QHBoxLayout()
        self.paper_chk = QCheckBox("Paper Trade (no live orders)")
        self.paper_chk.setChecked(True)
        self.start_all_btn = QPushButton("Start All")
        self.stop_all_btn = QPushButton("Stop All")
        top_layout.addWidget(self.paper_chk)
        top_layout.addStretch(1)
        top_layout.addWidget(self.start_all_btn)
        top_layout.addWidget(self.stop_all_btn)
        layout.addLayout(top_layout)

        stats_layout = QHBoxLayout()
        self.pnl_lbl = QLabel("Total PnL: 0.00")
        self.inv_lbl = QLabel("Total Invested: 0.00")
        stats_layout.addWidget(self.pnl_lbl)
        stats_layout.addSpacing(20)
        stats_layout.addWidget(self.inv_lbl)
        stats_layout.addStretch(1)
        layout.addLayout(stats_layout)

        for i in range(10):
            row = PairRow(self.pairs, i)
            row.start_clicked.connect(self.start_row)
            row.stop_clicked.connect(self.stop_row)
            layout.addWidget(row)
            self.rows.append(row)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Symbol", "PnL", "Invested"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(QLabel("Active Trades:"))
        layout.addWidget(self.table)

        self.start_all_btn.clicked.connect(self._start_all)
        self.stop_all_btn.clicked.connect(self._stop_all)

    def start_row(self, idx: int, symbol: str, capital: float):
        super().start_row(idx, symbol, capital)
        if self.rows[idx].running is False: return # Stopped by duplicate check

        wrapper = ExchangeWrapper(paper_trade=self.paper_chk.isChecked())
        worker = RealtimeWorker(symbol, capital, wrapper)
        worker.update_status.connect(self._log)
        worker.update_live.connect(self._update_live)
        worker.finished_pair.connect(self._finished_pair)
        self.workers[idx] = worker
        self.symbol_rows[symbol] = idx
        self._ensure_table_row(symbol)
        worker.start()

    def _ensure_table_row(self, symbol: str) -> int:
        for r in range(self.table.rowCount()):
            if self.table.item(r, 0) and self.table.item(r, 0).text() == symbol:
                return r
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(symbol))
        self.table.setItem(r, 1, QTableWidgetItem("0.00"))
        self.table.setItem(r, 2, QTableWidgetItem("0.00"))
        return r

    def _remove_table_row(self, symbol: str):
        for r in range(self.table.rowCount()):
            if self.table.item(r, 0) and self.table.item(r, 0).text() == symbol:
                self.table.removeRow(r)
                break

    def _recompute_totals(self):
        pnl, inv = 0.0, 0.0
        for r in range(self.table.rowCount()):
            pnl += float(self.table.item(r, 1).text())
            inv += float(self.table.item(r, 2).text())
        self.pnl_lbl.setText(f"Total PnL: {pnl:.2f}")
        self.inv_lbl.setText(f"Total Invested: {inv:.2f}")

    def _update_live(self, symbol: str, pnl: float, invested: float):
        r = self._ensure_table_row(symbol)
        self.table.setItem(r, 1, QTableWidgetItem(f"{pnl:.2f}"))
        self.table.setItem(r, 2, QTableWidgetItem(f"{invested:.2f}"))
        self._recompute_totals()

    def _finished_pair(self, symbol: str):
        self._remove_table_row(symbol)
        self._recompute_totals()
        super()._finished_pair(symbol)


class BacktestTab(BaseTab):
    def __init__(self, pairs: List[str]):
        super().__init__(pairs)
        layout = QVBoxLayout(self)

        settings_layout = QHBoxLayout()
        settings_layout.addWidget(QLabel("Backtesting Duration (Days):"))
        self.days = QSpinBox()
        self.days.setRange(1, 3650)
        self.days.setValue(30)
        settings_layout.addWidget(self.days)
        settings_layout.addStretch(1)
        layout.addLayout(settings_layout)

        top_layout = QHBoxLayout()
        self.start_all_btn = QPushButton("Start All")
        self.stop_all_btn = QPushButton("Stop All")
        top_layout.addStretch(1)
        top_layout.addWidget(self.start_all_btn)
        top_layout.addWidget(self.stop_all_btn)
        layout.addLayout(top_layout)

        stats_layout = QHBoxLayout()
        self.pnl_lbl = QLabel("Total PnL: 0.00")
        self.inv_lbl = QLabel("Total Invested: 0.00")
        stats_layout.addWidget(self.pnl_lbl)
        stats_layout.addSpacing(20)
        stats_layout.addWidget(self.inv_lbl)
        stats_layout.addStretch(1)
        layout.addLayout(stats_layout)

        for i in range(10):
            row = PairRow(self.pairs, i)
            row.start_clicked.connect(self.start_row)
            row.stop_clicked.connect(self.stop_row)
            layout.addWidget(row)
            self.rows.append(row)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Symbol", "PnL", "Invested"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(QLabel("Backtest Results:"))
        layout.addWidget(self.table)

        self.start_all_btn.clicked.connect(self._start_all)
        self.stop_all_btn.clicked.connect(self._stop_all)

    def start_row(self, idx: int, symbol: str, capital: float):
        super().start_row(idx, symbol, capital)
        if self.rows[idx].running is False: return

        ex = ExchangeWrapper(paper_trade=True)
        worker = BacktestWorker(symbol, capital, ex, self.days.value())
        worker.update_status.connect(self._log)
        worker.finished_pair.connect(self._finished_pair)
        worker.update_result.connect(self._update_result)
        self.workers[idx] = worker
        self.symbol_rows[symbol] = idx
        worker.start()

    def _append_result_row(self, symbol: str, pnl: float, invested: float):
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(symbol))
        self.table.setItem(r, 1, QTableWidgetItem(f"{pnl:.2f}"))
        self.table.setItem(r, 2, QTableWidgetItem(f"{invested:.2f}"))
        self._recompute_totals()

    def _recompute_totals(self):
        pnl, inv = 0.0, 0.0
        for r in range(self.table.rowCount()):
            pnl += float(self.table.item(r, 1).text())
            inv += float(self.table.item(r, 2).text())
        self.pnl_lbl.setText(f"Total PnL: {pnl:.2f}")
        self.inv_lbl.setText(f"Total Invested: {inv:.2f}")

    def _update_result(self, symbol: str, pnl: float, invested: float):
        self._append_result_row(symbol, pnl, invested)


# ========================================================================
# Main Window
# ========================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crypto Trader — Realtime & Backtesting (5m)")
        self.resize(1200, 800)

        pairs_path = os.path.join(os.path.dirname(__file__), 'pairs.txt')
        pairs = load_pairs(pairs_path)
        if not pairs:
            logger.warning("pairs.txt not found or empty, using default list.")
            pairs = [ "BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT","ADA/USDT", "XRP/USDT","DOGE/USDT","LINK/USDT","MATIC/USDT","DOT/USDT" ]

        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        splitter = QSplitter(Qt.Vertical)
        
        tabs = QTabWidget()
        self.rt_tab = RealTimeTab(pairs)
        self.bt_tab = BacktestTab(pairs)
        tabs.addTab(self.rt_tab, "Real-time Trading")
        tabs.addTab(self.bt_tab, "Backtesting")

        self.log = QTextEdit()
        self.log.setReadOnly(True)

        splitter.addWidget(tabs)
        splitter.addWidget(self.log)
        splitter.setSizes([600, 200])

        main_layout.addWidget(splitter)
        self.setCentralWidget(main_widget)

        self.rt_tab.log_signal.connect(self.append_log)
        self.bt_tab.log_signal.connect(self.append_log)

    def append_log(self, msg: str):
        self.log.append(msg)


# ========================================================================
# Entrypoint
# ========================================================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())