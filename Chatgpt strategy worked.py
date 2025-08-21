#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyQt5 Cryptocurrency Trading App (Binance via CCXT)
Strategy: Momentum Trend Strategy

- Timeframe: 1-hour (H1).
- Entry Signal: A bullish MACD crossover while the price is above the
  50-period SMA and the RSI is not overbought (< 70).
- Exit Signal: RSI becomes overbought (> 70) and the MACD crosses bearishly.
- Risk Management: A dynamic ATR-based stop-loss is used.

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
from typing import Optional, Dict, List

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
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSpinBox, QDoubleSpinBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QTextEdit, QSplitter, QProgressBar
)

# ===================== Strategy Parameters =====================
TRADE_TIMEFRAME = "1h"
BINANCE_FEE = 0.001
# -- Risk Management --
EQUITY_PER_TRADE = 0.10
ATR_PERIOD = 14
ATR_SL_MULTIPLIER = 2.0
# -- Indicators --
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
SMA_PERIOD = 50

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
# Custom Indicator Functions (from user)
# ========================================================================
def calculate_ema(data, window):
    return data.ewm(span=window, adjust=False).mean()

def calculate_macd(data, fast=12, slow=26, signal=9):
    macd = calculate_ema(data, fast) - calculate_ema(data, slow)
    signal_line = calculate_ema(macd, signal)
    return macd, signal_line

def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9) # Added epsilon for stability
    return 100 - (100 / (1 + rs))

# ========================================================================
# Helpers
# ========================================================================
def load_pairs(file_path: str) -> List[str]:
    if not os.path.exists(file_path): return []
    with open(file_path, 'r') as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith('#')]

def ohlcv_to_df(ohlcv: List[List[float]]) -> pd.DataFrame:
    df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Use custom functions from user
    df['RSI'] = calculate_rsi(df['close'], period=RSI_PERIOD)
    df['MACD'], df['MACD_Signal'] = calculate_macd(df['close'], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    df['SMA_50'] = df['close'].rolling(window=SMA_PERIOD).mean()
    # Add ATR for stop-loss calculation
    df.ta.atr(length=ATR_PERIOD, append=True)
    return df

@dataclass
class Position:
    symbol: str
    size: float
    entry_price: float
    stop_loss_price: float

# ========================================================================
# Exchange Wrapper
# ========================================================================
class ExchangeWrapper:
    def __init__(self):
        load_dotenv(); api_key = os.getenv('BINANCE_API_KEY', ''); api_secret = os.getenv('BINANCE_API_SECRET', '')
        self.client = ccxt.binance({
            'apiKey': api_key, 'secret': api_secret, 'enableRateLimit': True,
            'options': {'defaultType': 'spot'}, 'timeout': 30000
        })
        self.client.options['adjustForTimeDifference'] = True
    @property
    def rate_limit_s(self) -> float: return float(self.client.rateLimit) / 1000.0 if self.client.rateLimit else 0.2
    def fetch_ohlcv(self, symbol: str, timeframe: str, since: Optional[int] = None, limit: int = 1000) -> List[List[float]]: return self.client.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)

# ========================================================================
# Backtesting Worker
# ========================================================================
class BacktestWorker(QThread):
    update_status = pyqtSignal(str)
    finished_pair = pyqtSignal(str)
    update_result = pyqtSignal(str, float, float, int, float, float, float)

    def __init__(self, symbol: str, capital: float, ex: ExchangeWrapper, lookback_days: int):
        super().__init__()
        self.symbol = symbol; self.capital = capital; self.exchange = ex
        self.lookback_days = int(lookback_days); self._running = True

    def stop(self): self._running = False
    
    def run(self):
        try:
            self.update_status.emit(f"[BT] Fetching data for {self.symbol}...")
            now_ms = int(time.time() * 1000)
            since_ms = now_ms - (self.lookback_days + 10) * 24 * 60 * 60 * 1000
            
            all_ohlcv: List[List[float]] = []; since = since_ms
            while self._running:
                batch = self.exchange.fetch_ohlcv(self.symbol, TRADE_TIMEFRAME, since=since, limit=1000)
                if not batch: break
                all_ohlcv.extend(batch); since = batch[-1][0] + 1; time.sleep(self.exchange.rate_limit_s)
            
            if not all_ohlcv: self.update_status.emit(f"[BT] No data for {self.symbol}"); return
            
            df = compute_indicators(ohlcv_to_df(all_ohlcv))
            df.dropna(inplace=True); df.reset_index(drop=True, inplace=True)
            if df.empty: self.update_status.emit(f"[BT] Not enough data for {self.symbol}"); return

            position: Optional[Position] = None; trades = []; capital = self.capital
            total_fees = 0.0; total_equity_used = 0.0

            atr_col = f'ATRr_{ATR_PERIOD}'

            for i in range(1, len(df)):
                if not self._running: break
                row = df.iloc[i]; prev_row = df.iloc[i-1]
                
                # --- Exit Logic ---
                if position:
                    exit_price = None; reason = ""
                    if row['low'] <= position.stop_loss_price:
                        exit_price = position.stop_loss_price; reason = "Stop-Loss"
                    else:
                        sell_cond_rsi = row['RSI'] > RSI_OVERBOUGHT
                        sell_cond_macd = prev_row['MACD'] >= prev_row['MACD_Signal'] and row['MACD'] < row['MACD_Signal']
                        if sell_cond_rsi and sell_cond_macd:
                            exit_price = row['close']; reason = "Sell Signal"

                    if exit_price:
                        entry_cost = position.size * position.entry_price
                        exit_value = position.size * exit_price
                        fee = (entry_cost + exit_value) * BINANCE_FEE
                        pnl = exit_value - entry_cost - fee
                        total_fees += fee; trades.append({'pnl': pnl}); capital += pnl
                        self.update_status.emit(f"[BT] SELL {self.symbol} @ {exit_price:.4f} ({reason}). PnL: ${pnl:.2f}")
                        position = None
                        continue

                # --- Entry Logic ---
                if position is None:
                    buy_cond_rsi = row['RSI'] < RSI_OVERBOUGHT
                    buy_cond_macd = prev_row['MACD'] <= prev_row['MACD_Signal'] and row['MACD'] > row['MACD_Signal']
                    buy_cond_sma = row['close'] > row['SMA_50']
                    
                    if buy_cond_rsi and buy_cond_macd and buy_cond_sma:
                        entry_price = row['close']
                        atr_value = row[atr_col]
                        stop_loss_price = entry_price - (atr_value * ATR_SL_MULTIPLIER)
                        
                        amount_to_invest = capital * EQUITY_PER_TRADE
                        size = amount_to_invest / entry_price
                        
                        position = Position(self.symbol, size, entry_price, stop_loss_price)
                        total_equity_used += amount_to_invest
                        self.update_status.emit(f"[BT] BUY {self.symbol} @ {entry_price:.4f} (Momentum Signal)")

            if position:
                sell_price = float(df.iloc[-1]['close']); entry_cost = position.size * position.entry_price
                exit_value = position.size * sell_price; fees = (entry_cost + exit_value) * BINANCE_FEE
                pnl = exit_value - entry_cost - fees; total_fees += fees
                trades.append({'pnl': pnl}); capital += pnl
            
            total_pnl = capital - self.capital; num_trades = len(trades)
            win_trades = sum(1 for t in trades if t['pnl'] > 0)
            win_rate = (win_trades / num_trades * 100) if num_trades > 0 else 0.0
            roi = (total_pnl / self.capital * 100) if self.capital > 0 else 0.0
            
            self.update_result.emit(self.symbol, total_pnl, roi, num_trades, win_rate, total_equity_used, total_fees)
            self.update_status.emit(f"[BT] Done {self.symbol}: PnL: ${total_pnl:.2f}, Trades: {num_trades}, WinRate: {win_rate:.2f}%, ROI: {roi:.2f}%")

        except ccxt.BadSymbol: self.update_status.emit(f"[BT] Error: Symbol '{self.symbol}' not found. Skipping.")
        except ccxt.RequestTimeout: self.update_status.emit(f"[BT] Error: Network timeout for '{self.symbol}'. Skipping.")
        except Exception: self.update_status.emit(f"[BT] Worker Error on {self.symbol}: {traceback.format_exc()}")
        finally: self.finished_pair.emit(self.symbol)

# ========================================================================
# UI Classes
# ========================================================================
class BacktestTab(QWidget):
    log_signal = pyqtSignal(str)
    def __init__(self, pairs: List[str]):
        super().__init__(); self.pairs = pairs; self.workers: List[BacktestWorker] = []; self.completed_workers = 0
        self._reset_cumulative_stats()
        layout = QVBoxLayout(self); top_hbox = QHBoxLayout(); settings_group = QHBoxLayout()
        settings_group.addWidget(QLabel("Backtesting Duration (Days):")); self.days = QSpinBox(); self.days.setRange(30, 5000); self.days.setValue(365)
        settings_group.addWidget(self.days); settings_group.addWidget(QLabel("Initial Capital per Coin (USDT):"))
        self.capital_input = QDoubleSpinBox(); self.capital_input.setDecimals(2); self.capital_input.setRange(10.0, 1_000_000.0); self.capital_input.setValue(10000.0)
        settings_group.addWidget(self.capital_input); top_hbox.addLayout(settings_group); top_hbox.addStretch(1)
        self.start_all_btn = QPushButton("Start Full Backtest"); self.stop_all_btn = QPushButton("Stop All"); self.clear_btn = QPushButton("Clear Results")
        top_hbox.addWidget(self.start_all_btn); top_hbox.addWidget(self.stop_all_btn); top_hbox.addWidget(self.clear_btn); layout.addLayout(top_hbox)
        progress_hbox = QHBoxLayout(); self.progress_bar = QProgressBar(self); progress_hbox.addWidget(self.progress_bar)
        self.cumulative_pnl_lbl = QLabel("Cumulative Session PnL: $0.00"); progress_hbox.addWidget(self.cumulative_pnl_lbl); layout.addLayout(progress_hbox)
        self.table = QTableWidget(0, 7); self.table.setHorizontalHeaderLabels(["Symbol", "PnL ($)", "ROI (%)", "Trades", "Wins", "Win Rate (%)", "Fees ($)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch); layout.addWidget(QLabel(f"Backtest will run for all {len(pairs)} pairs found in pairs.txt:")); layout.addWidget(self.table)
        self.start_all_btn.clicked.connect(self.start_full_backtest); self.stop_all_btn.clicked.connect(self.stop_all_backtests); self.clear_btn.clicked.connect(self.clear_results)

    def _log(self, msg: str): self.log_signal.emit(msg)
    def _reset_cumulative_stats(self):
        self.cumulative_pnl = 0.0; self.cumulative_trades = 0; self.cumulative_wins = 0
        self.cumulative_equity_used = 0.0; self.cumulative_fees = 0.0
        
    def start_full_backtest(self):
        if self.workers: QMessageBox.warning(self, "In Progress", "A backtest is already running."); return
        self.clear_results(); self.start_all_btn.setEnabled(False); self.completed_workers = 0
        capital = self.capital_input.value(); days = self.days.value()
        self.progress_bar.setRange(0, len(self.pairs)); self.progress_bar.setValue(0)
        self._log(f"--- Starting full backtest for {len(self.pairs)} pairs ---")
        for pair in self.pairs:
            ex = ExchangeWrapper(); worker = BacktestWorker(pair, capital, ex, days)
            worker.update_status.connect(self._log); worker.finished_pair.connect(self.on_worker_finished); worker.update_result.connect(self.update_result_table)
            self.workers.append(worker); worker.start()

    def stop_all_backtests(self):
        for worker in self.workers: worker.stop()
        self._log("--- Stop signal sent to all active backtests ---")

    def on_worker_finished(self, symbol: str):
        self.completed_workers += 1; self.progress_bar.setValue(self.completed_workers)
        self.workers = [w for w in self.workers if w.symbol != symbol or not w.isFinished()]
        if not self.workers:
            self.start_all_btn.setEnabled(True); self._log("--- Full backtest complete ---"); self._add_cumulative_row()

    def _add_cumulative_row(self):
        r = self.table.rowCount(); self.table.insertRow(r)
        bold_font = QFont(); bold_font.setBold(True)
        cumulative_win_rate = (self.cumulative_wins / self.cumulative_trades * 100) if self.cumulative_trades > 0 else 0.0
        initial_capital_total = self.capital_input.value() * self.completed_workers if self.completed_workers > 0 else 0
        cumulative_roi = (self.cumulative_pnl / initial_capital_total * 100) if initial_capital_total > 0 else 0.0
        
        items = ["CUMULATIVE TOTAL", f"${self.cumulative_pnl:.2f}", f"{cumulative_roi:.2f}%", str(self.cumulative_trades), str(self.cumulative_wins), f"{cumulative_win_rate:.2f}%", f"${self.cumulative_fees:.2f}"]
        for i, text in enumerate(items):
            item = QTableWidgetItem(text); item.setFont(bold_font); self.table.setItem(r, i, item)

    def update_result_table(self, symbol: str, pnl: float, roi: float, trades: int, win_rate: float, total_equity_used: float, total_fees: float):
        self.cumulative_pnl += pnl; self.cumulative_trades += trades
        self.cumulative_equity_used += total_equity_used; self.cumulative_fees += total_fees
        wins = int(round(trades * (win_rate / 100.0))); self.cumulative_wins += wins
        self.cumulative_pnl_lbl.setText(f"Cumulative Session PnL: ${self.cumulative_pnl:.2f}")
        
        r = self.table.rowCount(); self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(symbol)); self.table.setItem(r, 1, QTableWidgetItem(f"{pnl:.2f}"))
        self.table.setItem(r, 2, QTableWidgetItem(f"{roi:.2f}%"))
        self.table.setItem(r, 3, QTableWidgetItem(str(trades))); self.table.setItem(r, 4, QTableWidgetItem(str(wins)))
        self.table.setItem(r, 5, QTableWidgetItem(f"{win_rate:.2f}%")); self.table.setItem(r, 6, QTableWidgetItem(f"{total_fees:.2f}"))
        
    def clear_results(self):
        self.table.setRowCount(0); self._reset_cumulative_stats()
        self.cumulative_pnl_lbl.setText("Cumulative Session PnL: $0.00"); self._log("Cleared backtest results.")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("Crypto Trader — Momentum Trend Strategy"); self.resize(1200, 800)
        main_widget = QWidget(); main_layout = QVBoxLayout(main_widget); splitter = QSplitter(Qt.Vertical)
        pairs_path = os.path.join(os.path.dirname(__file__), 'pairs.txt')
        pairs = load_pairs(pairs_path)
        if not pairs: logger.warning("pairs.txt not found, using default list."); pairs = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
        self.bt_tab = BacktestTab(pairs)
        self.log = QTextEdit(); self.log.setReadOnly(True)
        splitter.addWidget(self.bt_tab); splitter.addWidget(self.log); splitter.setSizes([600, 200])
        main_layout.addWidget(splitter); self.setCentralWidget(main_widget)
        self.bt_tab.log_signal.connect(self.append_log)
    def append_log(self, msg: str): self.log.append(msg)
    def closeEvent(self, event):
        self.append_log("Closing application, stopping all active threads...")
        self.bt_tab.stop_all_backtests()
        for worker in self.bt_tab.workers: worker.wait()
        self.append_log("All threads stopped. Exiting.")
        event.accept()

# ========================================================================
# Entrypoint
# ========================================================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())