#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyQt5 Cryptocurrency Trading App (Binance via CCXT)
Strategy: Buy on Green, Sell on Red (Pyramiding)

** WARNING **
This strategy is extremely high-risk. It does not use a stop-loss and aggressively
increases position size (pyramiding) into a trend. A sharp reversal can lead to
catastrophic losses. Use the backtester extensively to understand the risks.
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
except Exception as e:
    print(f"Error: A required library is not installed. {e}")
    print("Please run: pip install ccxt")
    raise

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow, QTabWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QDoubleSpinBox, QSpinBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QCheckBox, QTextEdit, QSplitter
)

# ===================== Strategy Parameters (Default) =====================
TIMEFRAME = "5m"
# The amount in USDT to buy on each green candle signal.
# The total invested amount will be a multiple of this value.
BUY_AMOUNT_PER_TRADE = 100.0

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

def ohlcv_to_df(ohlcv: List[List[float]]) -> pd.DataFrame:
    df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

@dataclass
class Position:
    """Represents an aggregated position for the pyramiding strategy."""
    symbol: str
    total_size: float = 0.0
    average_entry_price: float = 0.0
    total_invested: float = 0.0

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
            'apiKey': api_key, 'secret': api_secret, 'enableRateLimit': True,
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

    def market_buy(self, symbol: str, amount_in_quote: float) -> Dict:
        # Note: For this strategy, we buy with a certain amount of USDT, not a size of the base currency.
        # CCXT's create_market_buy_order_with_cost is ideal, but for simplicity we simulate.
        price = self.fetch_ticker_price(symbol)
        amount_in_base = amount_in_quote / price
        if self.paper:
            return {"info": {"paper": True}, "symbol": symbol, "cost": amount_in_quote, "amount": amount_in_base}
        return self.client.create_market_buy_order(symbol, amount_in_base)

    def market_sell(self, symbol: str, amount_in_base: float) -> Dict:
        if self.paper:
            return {"info": {"paper": True}, "symbol": symbol, "amount": amount_in_base}
        return self.client.create_market_sell_order(symbol, amount_in_base)

# ========================================================================
# Simplified Strategy
# ========================================================================
class Strategy:
    def decide(self, row: pd.Series, df: pd.DataFrame, position: Optional[Position]) -> Tuple[str, Dict]:
        i = row.name
        if i < 1: return "HOLD", {}

        is_bullish = row['close'] > row['open']
        is_bearish = row['close'] < row['open']
        
        prev_row = df.iloc[i - 1]
        prev_is_bullish = prev_row['close'] > prev_row['open']

        if position:
            if is_bearish:
                return "SELL_ALL", {"reason": "First Red Candle"}
            elif is_bullish:
                return "BUY_MORE", {}
        else: # No position is open
            if is_bullish and prev_is_bullish:
                return "BUY_NEW", {}
        
        return "HOLD", {}

# ========================================================================
# Worker Base
# ========================================================================
class BaseWorker(QThread):
    update_status = pyqtSignal(str)
    update_live = pyqtSignal(str, float, float) # symbol, pnl, total_invested
    finished_pair = pyqtSignal(str)

    def __init__(self, symbol: str, initial_capital: float, ex: ExchangeWrapper):
        super().__init__()
        self.symbol = symbol
        self.exchange = ex
        self._running = True
        self.position: Optional[Position] = None
        self.strategy = Strategy()

    def stop(self):
        self._running = False

# ========================================================================
# Real-time Trading Worker
# ========================================================================
class RealtimeWorker(BaseWorker):
    def run(self):
        try:
            self.update_status.emit(f"[RT] Starting {self.symbol} with Buy/Green Sell/Red strategy.")
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, TIMEFRAME, limit=200)
            df = ohlcv_to_df(ohlcv)

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

                    row = df.iloc[-1]
                    action, details = self.strategy.decide(row, df, self.position)
                    
                    current_price = self.exchange.fetch_ticker_price(self.symbol)
                    
                    if action in ["BUY_NEW", "BUY_MORE"]:
                        if action == "BUY_NEW":
                            self.position = Position(symbol=self.symbol)
                        
                        size_to_buy = BUY_AMOUNT_PER_TRADE / current_price
                        
                        self.exchange.market_buy(self.symbol, BUY_AMOUNT_PER_TRADE)

                        self.position.total_invested += BUY_AMOUNT_PER_TRADE
                        self.position.total_size += size_to_buy
                        self.position.average_entry_price = self.position.total_invested / self.position.total_size
                        self.update_status.emit(f"[RT] {action} {self.symbol} with {BUY_AMOUNT_PER_TRADE:.2f} USDT @ {current_price:.4f}")

                    elif action == "SELL_ALL" and self.position:
                        size_to_sell = self.position.total_size
                        self.exchange.market_sell(self.symbol, size_to_sell)
                        
                        proceeds = size_to_sell * current_price
                        pnl = proceeds - self.position.total_invested
                        
                        self.update_status.emit(f"[RT] SELL ALL {self.symbol} @ {current_price:.4f}. PnL: {pnl:.2f} USDT")
                        self.position = None

                    # Emit live PnL
                    pnl = 0.0
                    invested = 0.0
                    if self.position:
                        invested = self.position.total_invested
                        pnl = (current_price * self.position.total_size) - invested
                    self.update_live.emit(self.symbol, pnl, invested)

                except Exception as e:
                    self.update_status.emit(f"[RT] Loop Error: {e}")
                    time.sleep(5)
            
            if self.position:
                self.update_status.emit(f"[RT] Force selling {self.position.total_size:.6f} {self.symbol} on stop.")
                self.exchange.market_sell(self.symbol, self.position.total_size)
                self.position = None

        except Exception as e:
            self.update_status.emit(f"[RT] Worker Error: {traceback.format_exc()}")
        finally:
            self.finished_pair.emit(self.symbol)


# ========================================================================
# Backtesting Worker
# ========================================================================
class BacktestWorker(BaseWorker):
    update_result = pyqtSignal(str, float, float) # symbol, pnl, invested

    def __init__(self, symbol: str, initial_capital: float, ex: ExchangeWrapper, lookback_days: int):
        super().__init__(symbol, initial_capital, ex)
        self.lookback_days = int(lookback_days)

    def run(self):
        try:
            self.update_status.emit(f"[BT] Fetching data for {self.symbol}...")
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

            df = ohlcv_to_df(all_ohlcv)
            position: Optional[Position] = None
            trades_pnl = []

            for i in range(1, len(df)):
                if not self._running: break
                row = df.iloc[i]
                action, details = self.strategy.decide(row, df, position)

                if action == "BUY_NEW":
                    position = Position(symbol=self.symbol)
                    entry_price = float(row['close'])
                    size = BUY_AMOUNT_PER_TRADE / entry_price
                    position.total_invested += BUY_AMOUNT_PER_TRADE
                    position.total_size += size
                    position.average_entry_price = position.total_invested / position.total_size

                elif action == "BUY_MORE" and position:
                    entry_price = float(row['close'])
                    size = BUY_AMOUNT_PER_TRADE / entry_price
                    position.total_invested += BUY_AMOUNT_PER_TRADE
                    position.total_size += size
                    position.average_entry_price = position.total_invested / position.total_size

                elif action == "SELL_ALL" and position:
                    sell_price = float(row['close'])
                    proceeds = position.total_size * sell_price
                    pnl = proceeds - position.total_invested
                    trades_pnl.append(pnl)
                    position = None

            total_pnl = sum(trades_pnl)
            num_cycles = len(trades_pnl)
            win_cycles = sum(1 for pnl in trades_pnl if pnl > 0)
            win_pct = (win_cycles / num_cycles * 100) if num_cycles > 0 else 0
            
            self.update_result.emit(self.symbol, total_pnl, BUY_AMOUNT_PER_TRADE)
            self.update_status.emit(
                f"[BT] Done {self.symbol}: "
                f"Total PnL: {total_pnl:.2f}, "
                f"Trade Cycles: {num_cycles}, "
                f"Win Rate: {win_pct:.2f}%"
            )

        except Exception as e:
            self.update_status.emit(f"[BT] Worker Error: {traceback.format_exc()}")
        finally:
            self.finished_pair.emit(self.symbol)

# ========================================================================
# UI (largely unchanged)
# ========================================================================
class PairRow(QWidget):
    # This class is unchanged
    start_clicked = pyqtSignal(int, str, float)
    stop_clicked = pyqtSignal(int)
    def __init__(self, pairs: List[str], index: int, parent=None):
        super().__init__(parent)
        self.index = index
        layout = QHBoxLayout(self)
        self.combo = QComboBox(); self.combo.addItems(pairs)
        self.amount = QDoubleSpinBox(); self.amount.setDecimals(2); self.amount.setRange(0.0, 1_000_000.0); self.amount.setValue(100.0)
        self.btn = QPushButton("Start"); self.btn.clicked.connect(self._toggle)
        layout.addWidget(QLabel(f"Slot {index+1}"))
        layout.addWidget(self.combo, 2); layout.addWidget(QLabel("USDT:"))
        layout.addWidget(self.amount); layout.addWidget(self.btn)
        self.setLayout(layout); self.running = False
    def _toggle(self):
        if not self.running:
            symbol = self.combo.currentText().strip()
            capital = float(self.amount.value())
            if not symbol: QMessageBox.warning(self, "Invalid", "Please choose a symbol."); return
            self.running = True; self.btn.setText("Stop"); self.set_controls_enabled(False)
            self.start_clicked.emit(self.index, symbol, capital)
        else:
            self.running = False; self.btn.setText("Start"); self.set_controls_enabled(True)
            self.stop_clicked.emit(self.index)
    def reset(self):
        self.running = False; self.btn.setText("Start"); self.set_controls_enabled(True)
    def set_controls_enabled(self, enabled: bool):
        self.combo.setEnabled(enabled); self.amount.setEnabled(enabled)

class BaseTab(QWidget):
    # This class is unchanged
    log_signal = pyqtSignal(str)
    def __init__(self, pairs: List[str]):
        super().__init__(); self.pairs = pairs; self.workers: Dict[int, QThread] = {}
        self.symbol_rows: Dict[str, int] = {}; self.rows: List[PairRow] = []
    def _log(self, msg: str): logger.info(msg); self.log_signal.emit(msg)
    def _start_all(self):
        for row in self.rows:
            if not row.running: row._toggle()
    def _stop_all(self):
        for w in list(self.workers.values()): w.stop()
        for row in self.rows:
            if row.running: row.reset()
    def start_row(self, idx: int, symbol: str, capital: float):
        for worker in self.workers.values():
            if worker.symbol == symbol:
                QMessageBox.warning(self, "Duplicate Symbol", f"A trade for {symbol} is already active.")
                self.rows[idx].reset(); return
        if idx in self.workers: QMessageBox.warning(self, "Running", f"Slot {idx+1} already running"); return
        pass
    def stop_row(self, idx: int):
        w = self.workers.get(idx)
        if w: w.stop()
    def _finished_pair(self, symbol: str):
        idx = self.symbol_rows.pop(symbol, None)
        if idx is not None: self.workers.pop(idx, None); self.rows[idx].reset()

class RealTimeTab(BaseTab):
    # Unchanged except for connecting to the new RealtimeWorker
    def __init__(self, pairs: List[str]):
        super().__init__(pairs); layout = QVBoxLayout(self)
        top_layout = QHBoxLayout(); self.paper_chk = QCheckBox("Paper Trade (no live orders)"); self.paper_chk.setChecked(True)
        self.start_all_btn = QPushButton("Start All"); self.stop_all_btn = QPushButton("Stop All")
        top_layout.addWidget(self.paper_chk); top_layout.addStretch(1); top_layout.addWidget(self.start_all_btn); top_layout.addWidget(self.stop_all_btn); layout.addLayout(top_layout)
        stats_layout = QHBoxLayout(); self.pnl_lbl = QLabel("Total PnL: 0.00"); self.inv_lbl = QLabel("Total Invested: 0.00")
        stats_layout.addWidget(self.pnl_lbl); stats_layout.addSpacing(20); stats_layout.addWidget(self.inv_lbl); stats_layout.addStretch(1); layout.addLayout(stats_layout)
        for i in range(10):
            row = PairRow(self.pairs, i); row.start_clicked.connect(self.start_row); row.stop_clicked.connect(self.stop_row)
            layout.addWidget(row); self.rows.append(row)
        self.table = QTableWidget(0, 3); self.table.setHorizontalHeaderLabels(["Symbol", "PnL", "Invested"]); self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(QLabel("Active Trades:")); layout.addWidget(self.table)
        self.start_all_btn.clicked.connect(self._start_all); self.stop_all_btn.clicked.connect(self._stop_all)
    def start_row(self, idx: int, symbol: str, capital: float):
        super().start_row(idx, symbol, capital);
        if not self.rows[idx].running: return
        wrapper = ExchangeWrapper(paper_trade=self.paper_chk.isChecked()); worker = RealtimeWorker(symbol, capital, wrapper)
        worker.update_status.connect(self._log); worker.update_live.connect(self._update_live); worker.finished_pair.connect(self._finished_pair)
        self.workers[idx] = worker; self.symbol_rows[symbol] = idx; self._ensure_table_row(symbol); worker.start()
    def _ensure_table_row(self, symbol: str) -> int:
        for r in range(self.table.rowCount()):
            if self.table.item(r, 0) and self.table.item(r, 0).text() == symbol: return r
        r = self.table.rowCount(); self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(symbol)); self.table.setItem(r, 1, QTableWidgetItem("0.00")); self.table.setItem(r, 2, QTableWidgetItem("0.00"))
        return r
    def _remove_table_row(self, symbol: str):
        for r in range(self.table.rowCount()):
            if self.table.item(r, 0) and self.table.item(r, 0).text() == symbol: self.table.removeRow(r); break
    def _recompute_totals(self):
        pnl, inv = 0.0, 0.0
        for r in range(self.table.rowCount()): pnl += float(self.table.item(r, 1).text()); inv += float(self.table.item(r, 2).text())
        self.pnl_lbl.setText(f"Total PnL: {pnl:.2f}"); self.inv_lbl.setText(f"Total Invested: {inv:.2f}")
    def _update_live(self, symbol: str, pnl: float, invested: float):
        r = self._ensure_table_row(symbol); self.table.setItem(r, 1, QTableWidgetItem(f"{pnl:.2f}")); self.table.setItem(r, 2, QTableWidgetItem(f"{invested:.2f}")); self._recompute_totals()
    def _finished_pair(self, symbol: str): self._remove_table_row(symbol); self._recompute_totals(); super()._finished_pair(symbol)

class BacktestTab(BaseTab):
    # Unchanged except for connecting to the new BacktestWorker
    def __init__(self, pairs: List[str]):
        super().__init__(pairs); layout = QVBoxLayout(self); settings_layout = QHBoxLayout()
        settings_layout.addWidget(QLabel("Backtesting Duration (Days):")); self.days = QSpinBox(); self.days.setRange(1, 3650); self.days.setValue(30)
        settings_layout.addWidget(self.days); settings_layout.addStretch(1); layout.addLayout(settings_layout)
        top_layout = QHBoxLayout(); self.start_all_btn = QPushButton("Start All"); self.stop_all_btn = QPushButton("Stop All")
        top_layout.addStretch(1); top_layout.addWidget(self.start_all_btn); top_layout.addWidget(self.stop_all_btn); layout.addLayout(top_layout)
        stats_layout = QHBoxLayout(); self.pnl_lbl = QLabel("Total PnL: 0.00"); self.inv_lbl = QLabel("Total Invested: 0.00")
        stats_layout.addWidget(self.pnl_lbl); stats_layout.addSpacing(20); stats_layout.addWidget(self.inv_lbl); stats_layout.addStretch(1); layout.addLayout(stats_layout)
        for i in range(10):
            row = PairRow(self.pairs, i); row.start_clicked.connect(self.start_row); row.stop_clicked.connect(self.stop_row)
            layout.addWidget(row); self.rows.append(row)
        self.table = QTableWidget(0, 3); self.table.setHorizontalHeaderLabels(["Symbol", "PnL", "Cycles Won"]); self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(QLabel("Backtest Results:")); layout.addWidget(self.table)
        self.start_all_btn.clicked.connect(self._start_all); self.stop_all_btn.clicked.connect(self._stop_all)
    def start_row(self, idx: int, symbol: str, capital: float):
        super().start_row(idx, symbol, capital)
        if not self.rows[idx].running: return
        ex = ExchangeWrapper(paper_trade=True); worker = BacktestWorker(symbol, capital, ex, self.days.value())
        worker.update_status.connect(self._log); worker.finished_pair.connect(self._finished_pair); worker.update_result.connect(self._update_result)
        self.workers[idx] = worker; self.symbol_rows[symbol] = idx; worker.start()
    def _append_result_row(self, symbol: str, pnl: float, invested: float):
        r = self.table.rowCount(); self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(symbol)); self.table.setItem(r, 1, QTableWidgetItem(f"{pnl:.2f}"))
        self.table.setItem(r, 2, QTableWidgetItem(f"{invested:.0f}")); self._recompute_totals() # Using 'invested' to pass win_cycles
    def _recompute_totals(self):
        pnl, inv = 0.0, 0.0
        for r in range(self.table.rowCount()): pnl += float(self.table.item(r, 1).text())
        self.pnl_lbl.setText(f"Total PnL: {pnl:.2f}"); self.inv_lbl.setText("")
    def _update_result(self, symbol: str, pnl: float, win_cycles: float): self._append_result_row(symbol, pnl, win_cycles)

class MainWindow(QMainWindow):
    # Unchanged
    def __init__(self):
        super().__init__(); self.setWindowTitle("Crypto Trader — Buy Green/Sell Red"); self.resize(1200, 800)
        pairs_path = os.path.join(os.path.dirname(__file__), 'pairs.txt')
        pairs = load_pairs(pairs_path)
        if not pairs: logger.warning("pairs.txt not found, using default list."); pairs = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT"]
        main_widget = QWidget(); main_layout = QVBoxLayout(main_widget); splitter = QSplitter(Qt.Vertical)
        tabs = QTabWidget(); self.rt_tab = RealTimeTab(pairs); self.bt_tab = BacktestTab(pairs)
        tabs.addTab(self.rt_tab, "Real-time Trading"); tabs.addTab(self.bt_tab, "Backtesting")
        self.log = QTextEdit(); self.log.setReadOnly(True)
        splitter.addWidget(tabs); splitter.addWidget(self.log); splitter.setSizes([600, 200])
        main_layout.addWidget(splitter); self.setCentralWidget(main_widget)
        self.rt_tab.log_signal.connect(self.append_log); self.bt_tab.log_signal.connect(self.append_log)
    def append_log(self, msg: str): self.log.append(msg)
    def closeEvent(self, event):
        """
        Overrides the default close event to ensure all threads are stopped
        gracefully before the application exits.
        """
        self.append_log("Closing application, stopping all active threads...")
        
        # Stop all threads in both tabs
        self.rt_tab._stop_all()
        self.bt_tab._stop_all()
        
        # Wait for all threads to finish
        active_workers = list(self.rt_tab.workers.values()) + list(self.bt_tab.workers.values())
        for worker in active_workers:
            worker.wait() # This blocks until the thread's run() method has returned

        self.append_log("All threads stopped. Exiting.")
        event.accept() # Now it's safe to close

# ========================================================================
# Entrypoint
# ========================================================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())