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
9) **Strategy Filters**: Added Volume, ADX (Trend Strength).
10) **Trailing Take Profit**: Replaced fixed TP with a client-side trailing stop that activates on target.

Requirements (pip):
  PyQt5, ccxt, pandas, numpy, python-dotenv, pandas_ta
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

# --- Third-party libs ---
try:
    import ccxt
    import pandas_ta as ta
except Exception as e:
    print(f"Error: A required library is not installed. {e}")
    print("Please run: pip install ccxt pandas_ta")
    raise


from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow, QTabWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QDoubleSpinBox, QSpinBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QCheckBox, QTextEdit, QSplitter
)

# ===================== Strategy Parameters (Default) =====================
TIMEFRAME = "5m"
RISK_PER_TRADE = 0.10
STOP_LOSS_PCT = 0.02
SMA_FAST = 50
SMA_SLOW = 200
CONSEC_BULL = 2
RSI_PERIOD = 14
RSI_THRESHOLD = 60
TRAILING_STOP_PCT = 0.03
BINANCE_FEE = 0.001
ADX_THRESHOLD = 25
VOLUME_MULTIPLIER = 1.5
RISK_REWARD_RATIO = 2.0 # Target for activating the tight trail
TIGHT_TRAILING_STOP_PCT = 0.015 # The new, tighter trail (e.g., 1.5%)

# ========================================================================
# Logging setup
# ========================================================================
logger = logging.getLogger("crypto_trader")
logger.setLevel(logging.INFO)
if not logger.handlers:
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
    now_s = now_ms / 1000.0
    next_boundary = math.ceil(now_s / 300.0) * 300.0
    delay = max(0.0, next_boundary - now_s)
    return delay + 1.0


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
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df.ta.adx(append=True)
    return df


@dataclass
class Position:
    symbol: str
    size: float
    entry_price: float
    stop_loss: float
    trailing_stop: float
    invested: float
    tp_activation_price: float
    is_tight_trail_active: bool = False


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
            'options': {'defaultType': 'spot'}
        })

    @property
    def rate_limit_s(self) -> float:
        return float(self.client.rateLimit) / 1000.0 if self.client.rateLimit else 0.2

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
# Unified Strategy
# ========================================================================
class Strategy:
    def _get_running_bullish_count(self, df: pd.DataFrame, current_index: int) -> int:
        count = 0
        for i in range(current_index, -1, -1):
            if i == 0: break
            row = df.iloc[i]
            if row['close'] > row['open']:
                count += 1
            else:
                break
        return count

    def decide(self, row: pd.Series, df: pd.DataFrame, position: Optional[Position]) -> Tuple[str, Dict]:
        i = row.name
        is_bearish = row['close'] < row['open']

        if position is None:
            bullish_count = self._get_running_bullish_count(df, i)
            strong_trend = row['sma_fast'] > row['sma_slow']
            strong_momentum = row['rsi'] > RSI_THRESHOLD
            strong_candle = row['close'] > row['open'] + 0.75 * (row['high'] - row['low'])
            strong_volume = row['volume'] > row['volume_sma'] * VOLUME_MULTIPLIER
            trending_market = row[f'ADX_{RSI_PERIOD}'] > ADX_THRESHOLD

            if (bullish_count >= CONSEC_BULL and row['close'] > row['sma_fast'] and
                strong_trend and strong_momentum and strong_candle and strong_volume and trending_market):
                stop_loss_price = float(row['close']) * (1 - STOP_LOSS_PCT)
                return "BUY", {"stop_loss": stop_loss_price}
            return "HOLD", {}

        # Exit logic
        if row['low'] <= position.stop_loss:
            return "SELL", {"sell_price": position.stop_loss, "reason": "stop_loss"}
        if row['low'] <= position.trailing_stop:
            return "SELL", {"sell_price": position.trailing_stop, "reason": "trailing_stop"}
        if is_bearish and row['close'] < row['sma_fast']:
            return "SELL", {"sell_price": float(row['close']), "reason": "bearish_exit"}
        return "HOLD", {}


# ========================================================================
# Worker Base
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
            self.update_status.emit(f"[RT] Starting {self.symbol} with {self.initial_capital:.2f} USDT")
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, TIMEFRAME, limit=max(1000, SMA_SLOW+50))
            df = compute_indicators(ohlcv_to_df(ohlcv))

            while self._running:
                now_ms = int(self.exchange.client.milliseconds())
                delay = ms_to_min5_boundary_delay(now_ms)
                slept = 0
                while self._running and slept < delay:
                    time.sleep(1)
                    slept += 1
                if not self._running: break

                try:
                    latest_candles = self.exchange.fetch_ohlcv(self.symbol, TIMEFRAME, limit=2)
                    new_df = ohlcv_to_df(latest_candles)
                    last_ts = df['timestamp'].iloc[-1]
                    to_append = new_df[new_df['timestamp'] > last_ts]
                    if not to_append.empty:
                        df = pd.concat([df, to_append], ignore_index=True)
                        df = compute_indicators(df)

                    row = df.iloc[-1]
                    current_price = self.exchange.fetch_ticker_price(self.symbol)

                    if self.position:
                        if not self.position.is_tight_trail_active and current_price >= self.position.tp_activation_price:
                            self.position.is_tight_trail_active = True
                            self.update_status.emit(f"[RT] {self.symbol} TP target hit. Activating tight {TIGHT_TRAILING_STOP_PCT*100:.1f}% trail.")
                        trail_pct = TIGHT_TRAILING_STOP_PCT if self.position.is_tight_trail_active else TRAILING_STOP_PCT
                        self.position.trailing_stop = max(self.position.trailing_stop, current_price * (1 - trail_pct))
                    
                    action, details = self.strategy.decide(row, df, self.position)

                    if action == "BUY" and self.position is None:
                        risk_amount = self.capital * RISK_PER_TRADE
                        stop_loss_price = float(details["stop_loss"])
                        if row['close'] <= stop_loss_price: continue
                        
                        raw_size = risk_amount / (row['close'] - stop_loss_price)
                        size = raw_size
                        if size <= 0: continue
                        
                        entry_price = current_price
                        invested = entry_price * size
                        price_risk = entry_price - stop_loss_price
                        tp_activation_price = entry_price + (price_risk * RISK_REWARD_RATIO)
                        
                        try:
                            self.exchange.market_buy(self.symbol, size)
                        except Exception as e:
                            self.update_status.emit(f"ERROR buying {self.symbol}: {e}")
                            continue
                        
                        self.position = Position(self.symbol, size, entry_price, stop_loss_price, entry_price * (1-TRAILING_STOP_PCT), invested, tp_activation_price)
                        self.capital -= invested
                        self.update_status.emit(f"[RT] BUY {self.symbol} size={size:.6f} @ {entry_price:.6f} TP-Activation={tp_activation_price:.6f}")

                    elif action == "SELL" and self.position:
                        sell_price = float(details.get("sell_price", row['close']))
                        reason = details.get("reason", "exit")
                        size = self.position.size
                        try:
                            self.exchange.market_sell(self.symbol, size)
                        except Exception as e:
                            self.update_status.emit(f"ERROR selling {self.symbol}: {e}")
                            continue

                        pnl = (sell_price - self.position.entry_price) * size
                        self.capital += sell_price * size
                        self.trades.append({'symbol': self.symbol, 'pnl': pnl, 'type': reason})
                        self.update_status.emit(f"[RT] SELL {self.symbol} ({reason}) PnL={pnl:.4f}")
                        self.position = None

                    self._emit_live()

                except ccxt.NetworkError as e:
                    self.update_status.emit(f"[RT] Network error {self.symbol}: {e}")
                    time.sleep(self.exchange.rate_limit_s)
                except Exception as e:
                    self.update_status.emit(f"[RT] Loop error {self.symbol}: {traceback.format_exc()}")
                    time.sleep(self.exchange.rate_limit_s)

            if self.position:
                try:
                    price = self.exchange.fetch_ticker_price(self.symbol)
                    self.exchange.market_sell(self.symbol, self.position.size)
                    pnl = (price - self.position.entry_price) * self.position.size
                    self.update_status.emit(f"[RT] FORCE-SELL {self.symbol} @ {price:.6f} PnL={pnl:.4f}")
                except Exception as e:
                    self.update_status.emit(f"[RT] Error force-closing {self.symbol}: {e}")
                finally:
                    self.position = None
                    self._emit_live()

        except Exception as e:
            self.update_status.emit(f"[RT] Worker error {self.symbol}: {traceback.format_exc()}")
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
                current_price = float(row['close'])

                if position is not None:
                    if not position.is_tight_trail_active and current_price >= position.tp_activation_price:
                        position.is_tight_trail_active = True
                    trail_pct = TIGHT_TRAILING_STOP_PCT if position.is_tight_trail_active else TRAILING_STOP_PCT
                    position.trailing_stop = max(position.trailing_stop, current_price * (1 - trail_pct))

                action, details = self.strategy.decide(row, df, position)
                if action == "BUY" and position is None:
                    risk_amount = capital * RISK_PER_TRADE
                    stop_loss_price = float(details["stop_loss"])
                    if row['close'] <= stop_loss_price: continue
                    
                    size = risk_amount / (row['close'] - stop_loss_price)
                    invested = float(row['close']) * size
                    size *= (1 - BINANCE_FEE)
                    
                    entry_price = float(row['close'])
                    price_risk = entry_price - stop_loss_price
                    tp_activation_price = entry_price + (price_risk * RISK_REWARD_RATIO)
                    
                    trades.append({'type': 'BUY', 'price': entry_price, 'size': size})
                    position = Position(self.symbol, size, entry_price, stop_loss_price, entry_price*(1-TRAILING_STOP_PCT), invested, tp_activation_price)
                    capital -= invested

                elif action == "SELL" and position is not None:
                    sell_price = float(details.get("sell_price", row['close']))
                    net_proceeds = (sell_price * position.size) * (1 - BINANCE_FEE)
                    pnl = net_proceeds - position.invested
                    capital += net_proceeds
                    trades.append({'type': 'SELL', 'price': sell_price, 'pnl': pnl})
                    position = None

            if position is not None:
                last_close = float(df.iloc[-1]['close'])
                net_proceeds = (last_close * position.size) * (1 - BINANCE_FEE)
                pnl = net_proceeds - position.invested
                trades.append({'type': 'SELL', 'price': last_close, 'pnl': pnl})

            invested_amount = self.initial_capital
            total_pnl = float(sum(t.get('pnl', 0.0) for t in trades))
            self.update_result.emit(self.symbol, total_pnl, invested_amount)

            num_trades = len(trades)
            num_buys = sum(1 for t in trades if t['type'] == 'BUY')
            num_sells = sum(1 for t in trades if t['type'] == 'SELL')
            win_trades = sum(1 for t in trades if t.get('pnl', 0.0) > 0)
            win_pct = (win_trades / num_sells * 100) if num_sells else 0.0
            roi = (total_pnl / invested_amount * 100) if invested_amount else 0.0

            self.update_status.emit(
                f"[BT] Done {self.symbol}: "
                f"Invested {invested_amount:.2f}, PnL {total_pnl:.2f}, "
                f"Trades {num_trades}, Buys {num_buys}, Sells {num_sells}, "
                f"Win% {win_pct:.2f}%, ROI {roi:.2f}%"
            )

        except (ccxt.AuthenticationError, ccxt.NetworkError) as e:
            self.update_status.emit(f"ERROR: {e.__class__.__name__} for {self.symbol}. Check keys/connection.")
        except Exception as e:
            self.update_status.emit(f"[BT] Worker error {self.symbol}: {traceback.format_exc()}")
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
        if self.rows[idx].running is False: return

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
            pairs = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT", "XRP/USDT"]

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