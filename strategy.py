#!/usr/binbin/env python3
# -*- coding: utf-8 -*-
"""
Stable PyQt5 backtest runner with improved SMC strategy:
- CHOCH + OB/FVG rebound entry
- RSI momentum filter
- Automatic resistance detection from swing highs (two targets)
- Partial TP at TP1 (50%), move SL to breakeven, final TP at TP2
- Same UI/structure; only the strategy internals are upgraded

Notes:
 - Limits concurrent CCXT workers (default 4) to avoid blasting Binance.
 - Robust error handling and logging (debug_backtest.log).
"""

import os
import sys
import time
import traceback
import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple

import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Third-party libs
try:
    import ccxt
    from smartmoneyconcepts import smc
except Exception as e:
    print(f"Missing libraries: {e}")
    print("Run: pip install ccxt pandas numpy smartmoneyconcepts python-dotenv PyQt5")
    raise

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSpinBox, QDoubleSpinBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QTextEdit, QSplitter, QProgressBar
)

# -------------------------
# Config
# -------------------------
TRADE_TIMEFRAME = "1h"
EXCHANGE_FEE = 0.001
EQUITY_PER_TRADE = 1            # 1 = full capital per coin (paper)
SL_PERCENTAGE = 0.01            # fallback SL pad below zone bottom
RESISTANCE_LOOKFORWARD = 200
CHOCH_MAX_LOOKBACK = 60
REBOUND_LOOKFORWARD = 200

RSI_LEN = 14
RSI_OVERSOLD = 40
RSI_OVERBOUGHT = 60
RSI_SL_PERCENTAGE = 0.10 # 10% stop loss
TAKE_PROFIT_RR = 1.5 # Reward:Risk ratio for TP

# Concurrency: reduce to avoid many simultaneous CCXT clients
MAX_CONCURRENT_WORKERS = 4

# Logging to file for easier debugging
LOG_FILE = "debug_backtest.log"
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler(LOG_FILE, encoding='utf-8')])
logger = logging.getLogger("crypto_trader")

# -------------------------
# Helpers
# -------------------------
def load_pairs(file_path: str) -> List[str]:
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r') as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith('#')]

def rsi_series(close: pd.Series, length: int = 14) -> pd.Series:
    # Wilder's RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # RSI
    try:
        df['RSI'] = rsi_series(df['close'].astype(float), RSI_LEN)
    except Exception:
        df['RSI'] = 50.0

    # EMA for trend filter
    try:
        df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()
    except Exception:
        df['EMA200'] = df['close']

    df[['open','high','low','close']] = df[['open','high','low','close']].astype(float)
    return df


@dataclass
class Position:
    symbol: str
    size: float
    entry_price: float
    stop_loss_price: float
    tp1_price: float
    tp2_price: float
    half_closed: bool = False

# -------------------------
# Exchange wrapper (safe)
# -------------------------
class ExchangeWrapper:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv('EXCHANGE_API_KEY', '')
        api_secret = os.getenv('EXCHANGE_API_SECRET', '')
        options = {'defaultType': 'spot', 'fetchCurrencies': False}
        self.client = ccxt.kucoin({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': options,
            'timeout': 30000
        })
        try:
            self.client.options['adjustForTimeDifference'] = True
        except Exception:
            pass
        try:
            # try to preload markets once
            self.client.load_markets(False)
        except Exception as e:
            logger.warning(f"load_markets warning: {e}")

    @property
    def rate_limit_s(self) -> float:
        return float(getattr(self.client, "rateLimit", 200)) / 1000.0

    def fetch_ohlcv(self, symbol: str, timeframe: str, since: Optional[int] = None, limit: int = 1000) -> List[List[float]]:
        attempts = 0
        while attempts < 3:
            try:
                return self.client.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
            except ccxt.BadSymbol:
                raise
            except ccxt.InvalidNonce as e:
                attempts += 1
                logger.warning(f"InvalidNonce retry {attempts} for {symbol}: {e}")
                time.sleep(0.5 + attempts*0.2)
                continue
            except ccxt.NetworkError as e:
                attempts += 1
                logger.warning(f"NetworkError retry {attempts} for {symbol}: {e}")
                time.sleep(0.5 + attempts*0.5)
                continue
            except Exception as e:
                logger.exception(f"Unexpected fetch_ohlcv error for {symbol}: {e}")
                return []
        try:
            return self.client.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        except Exception as e:
            logger.exception(f"Final fetch_ohlcv failed for {symbol}: {e}")
            return []

# -------------------------
# Worker
# -------------------------
class BacktestWorker(QThread):
    update_status = pyqtSignal(str)
    finished_pair = pyqtSignal(str)
    update_result = pyqtSignal(str, float, float, int, float, float, float)

    def __init__(self, symbol: str, capital: float, exchange: ExchangeWrapper, lookback_days: int, fee_rate: float, gui_mode: bool = True):
        super().__init__()
        self.symbol = symbol
        self.capital = capital
        self.exchange = exchange
        self.lookback_days = int(lookback_days)
        self.fee_rate = fee_rate
        self._running = True
        self.gui_mode = gui_mode

    def stop(self):
        self._running = False

    def log_emit(self, msg: str):
        if self.gui_mode:
            try:
                self.update_status.emit(msg)
            except Exception:
                logger.info(msg)
        else:
            logger.info(msg)

    def run(self):
        try:
            self.log_emit(f"[BT] Loading data for {self.symbol} from local file...")

            pair_name = self.symbol.replace('/', '')
            file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'For research', f'{pair_name}.csv')

            if not os.path.exists(file_path):
                self.log_emit(f"[BT] Error: Data file not found at {file_path}. Skipping.")
                if self.gui_mode:
                    self.finished_pair.emit(self.symbol)
                return

            df = pd.read_csv(file_path, sep='\\t', engine='python')

            df.rename(columns={
                'Open_time': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }, inplace=True)

            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%m/%d/%Y %H:%M')

            df = compute_indicators(df)
            df.reset_index(drop=True, inplace=True)
            if df.empty:
                self.log_emit(f"[BT] Not enough data for {self.symbol}")
                self.finished_pair.emit(self.symbol)
                return

            position: Optional[Position] = None
            trades = []
            capital = self.capital
            total_fees = 0.0
            total_equity_used = 0.0

            for i in range(1, len(df)):
                if not self._running:
                    break

                row = df.iloc[i]
                rsi_now = df['RSI'].iat[i]
                rsi_prev = df['RSI'].iat[i-1]

                # --- MANAGE OPEN POSITION ---
                if position:
                    # Stop-Loss
                    if row['low'] <= position.stop_loss_price:
                        exit_price = position.stop_loss_price
                        entry_cost = position.size * position.entry_price
                        exit_value = position.size * exit_price
                        fee = (entry_cost + exit_value) * self.fee_rate
                        pnl = exit_value - entry_cost - fee
                        trades.append({'pnl': pnl})
                        total_fees += fee
                        capital += pnl
                        self.log_emit(f"[BT] SELL {self.symbol} @ {exit_price:.4f} (Stop-Loss). PnL: ${pnl:.2f}")
                        position = None
                        continue

                    # Check for Take Profit
                    if row['high'] >= position.tp1_price:
                        exit_price = position.tp1_price
                        entry_cost = position.size * position.entry_price
                        exit_value = position.size * exit_price
                        fee = (entry_cost + exit_value) * self.fee_rate
                        pnl = exit_value - entry_cost - fee
                        trades.append({'pnl': pnl})
                        total_fees += fee
                        capital += pnl
                        self.log_emit(f"[BT] SELL {self.symbol} @ {exit_price:.4f} (Take-Profit). PnL: ${pnl:.2f}")
                        position = None
                        continue

                    # Exit Signal (RSI crosses above OVERBOUGHT)
                    if rsi_prev < RSI_OVERBOUGHT and rsi_now >= RSI_OVERBOUGHT:
                        exit_price = row['close']
                        entry_cost = position.size * position.entry_price
                        exit_value = position.size * exit_price
                        fee = (entry_cost + exit_value) * self.fee_rate
                        pnl = exit_value - entry_cost - fee
                        trades.append({'pnl': pnl})
                        total_fees += fee
                        capital += pnl
                        self.log_emit(f"[BT] SELL {self.symbol} @ {exit_price:.4f} (RSI Overbought). PnL: ${pnl:.2f}")
                        position = None
                        continue

                # --- ENTRY LOGIC ---
                if position is None:
                    # Entry Signal (RSI crosses below OVERSOLD & price > EMA200)
                    if rsi_prev > RSI_OVERSOLD and rsi_now <= RSI_OVERSOLD and row['close'] > df['EMA200'].iat[i]:
                        entry_price = row['close']
                        stop_loss_price = entry_price * (1 - RSI_SL_PERCENTAGE)

                        # Calculate Take Profit based on Risk:Reward
                        risk = entry_price - stop_loss_price
                        take_profit_price = entry_price + (risk * TAKE_PROFIT_RR)

                        amount_to_invest = capital * EQUITY_PER_TRADE
                        size = amount_to_invest / entry_price if entry_price > 0 else 0.0

                        position = Position(
                            symbol=self.symbol,
                            size=size,
                            entry_price=entry_price,
                            stop_loss_price=stop_loss_price,
                            tp1_price=take_profit_price, # Use tp1_price for our TP
                            tp2_price=0, # Not used
                            half_closed=False
                        )
                        total_equity_used += amount_to_invest
                        self.log_emit(f"[BT] BUY {self.symbol} @ {entry_price:.4f} (RSI Oversold) SL {stop_loss_price:.4f} TP {take_profit_price:.4f}")
                        continue

            # close open position at EOD
            if position:
                sell_price = float(df.iloc[-1]['close'])
                entry_cost = position.size * position.entry_price
                exit_value = position.size * sell_price
                fees = (entry_cost + exit_value) * self.fee_rate
                pnl = exit_value - entry_cost - fees
                total_fees += fees
                trades.append({'pnl': pnl})
                capital += pnl
                self.log_emit(f"[BT] EOD SELL {self.symbol} @ {sell_price:.4f}. PnL: ${pnl:.2f}")

            total_pnl = capital - self.capital
            num_trades = len(trades)
            win_trades = sum(1 for t in trades if t['pnl'] > 0)
            win_rate = (win_trades / num_trades * 100) if num_trades > 0 else 0.0
            roi = (total_pnl / self.capital * 100) if self.capital > 0 else 0.0

            if self.gui_mode:
                self.update_result.emit(self.symbol, total_pnl, roi, num_trades, win_rate, total_equity_used, total_fees)
            self.log_emit(f"[BT] Done {self.symbol}: PnL ${total_pnl:.2f}, Trades {num_trades}, WinRate {win_rate:.2f}%, ROI {roi:.2f}%")

        except Exception as e:
            logger.exception(f"Worker fatal error for {self.symbol}: {e}")
            self.log_emit(f"[BT] Worker Error on {self.symbol}: {e}")
        finally:
            if self.gui_mode:
                self.finished_pair.emit(self.symbol)

# -------------------------
# UI and queueing
# -------------------------
class BacktestTab(QWidget):
    log_signal = pyqtSignal(str)

    def __init__(self, pairs: List[str]):
        super().__init__()
        self.pairs = list(pairs)
        self.pending = list(self.pairs)  # queue
        self.workers: List[BacktestWorker] = []
        self.active_workers = {}
        self.completed_workers = 0
        self._reset_cumulative_stats()

        # UI
        layout = QVBoxLayout(self)
        top_hbox = QHBoxLayout()

        settings_group = QHBoxLayout()
        settings_group.addWidget(QLabel("Backtesting Duration (Days):"))
        self.days = QSpinBox(); self.days.setRange(30, 5000); self.days.setValue(365)
        settings_group.addWidget(self.days)

        settings_group.addWidget(QLabel("Initial Capital per Coin (USDT):"))
        self.capital_input = QDoubleSpinBox(); self.capital_input.setDecimals(2); self.capital_input.setRange(10.0, 1_000_000.0); self.capital_input.setValue(10000.0)
        settings_group.addWidget(self.capital_input)

        top_hbox.addLayout(settings_group); top_hbox.addStretch(1)

        self.start_all_btn = QPushButton("Start Full Backtest")
        self.stop_all_btn = QPushButton("Stop All")
        self.clear_btn = QPushButton("Clear Results")
        top_hbox.addWidget(self.start_all_btn); top_hbox.addWidget(self.stop_all_btn); top_hbox.addWidget(self.clear_btn)
        layout.addLayout(top_hbox)

        progress_hbox = QHBoxLayout()
        self.progress_bar = QProgressBar(self)
        progress_hbox.addWidget(self.progress_bar)
        self.cumulative_pnl_lbl = QLabel("Cumulative Session PnL: $0.00")
        progress_hbox.addWidget(self.cumulative_pnl_lbl)
        layout.addLayout(progress_hbox)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["Symbol", "PnL ($)", "ROI (%)", "Trades", "Wins", "Win Rate (%)", "Fees ($)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        layout.addWidget(QLabel(f"Backtest will run for all {len(self.pairs)} pairs found in pairs.txt:"))
        layout.addWidget(self.table)

        self.start_all_btn.clicked.connect(self.start_full_backtest)
        self.stop_all_btn.clicked.connect(self.stop_all_backtests)
        self.clear_btn.clicked.connect(self.clear_results)

    def _log(self, msg: str):
        self.log_signal.emit(msg); logger.info(msg)

    def _reset_cumulative_stats(self):
        self.cumulative_pnl = 0.0
        self.cumulative_trades = 0
        self.cumulative_wins = 0
        self.cumulative_equity_used = 0.0
        self.cumulative_fees = 0.0
        self.completed_workers = 0

    def start_full_backtest(self):
        if self.active_workers:
            QMessageBox.warning(self, "In Progress", "A backtest is already running.")
            return
        self.clear_results()
        self.start_all_btn.setEnabled(False)
        self.pending = list(self.pairs)
        self.progress_bar.setRange(0, len(self.pairs)); self.progress_bar.setValue(0)
        self._log(f"--- Starting full backtest for {len(self.pairs)} pairs (max concurrent {MAX_CONCURRENT_WORKERS}) ---")
        # start up to MAX_CONCURRENT_WORKERS
        for _ in range(min(MAX_CONCURRENT_WORKERS, len(self.pending))):
            self._start_next_worker()

    def _start_next_worker(self):
        if not self.pending:
            return
        pair = self.pending.pop(0)
        ex = ExchangeWrapper()
        worker = BacktestWorker(pair, self.capital_input.value(), ex, self.days.value(), fee_rate=EXCHANGE_FEE)
        worker.update_status.connect(self._log)
        worker.finished_pair.connect(self.on_worker_finished)
        worker.update_result.connect(self.update_result_table)
        self.active_workers[pair] = worker
        worker.start()
        self._log(f"[BT] Worker started for {pair}. Active workers: {len(self.active_workers)}")

    def stop_all_backtests(self):
        for w in list(self.active_workers.values()):
            w.stop()
        self.pending = []
        self._log("--- Stop signal sent to all active backtests ---")

    def on_worker_finished(self, symbol: str):
        try:
            if symbol in self.active_workers:
                del self.active_workers[symbol]
            self.completed_workers += 1
            self.progress_bar.setValue(self.completed_workers)
            self._log(f"[BT] Worker finished for {symbol}. Completed {self.completed_workers}/{len(self.pairs)}")
            if self.pending:
                QTimer.singleShot(200, self._start_next_worker)
            elif not self.active_workers:
                self.start_all_btn.setEnabled(True)
                self._log("--- Full backtest complete ---")
                self._add_cumulative_row()
        except Exception:
            logger.exception("Error in on_worker_finished")

    def _add_cumulative_row(self):
        r = self.table.rowCount(); self.table.insertRow(r)
        bold_font = QFont(); bold_font.setBold(True)
        cumulative_win_rate = (self.cumulative_wins / self.cumulative_trades * 100) if self.cumulative_trades > 0 else 0.0
        initial_capital_total = self.capital_input.value() * self.completed_workers if self.completed_workers > 0 else 0
        cumulative_roi = (self.cumulative_pnl / initial_capital_total * 100) if initial_capital_total > 0 else 0.0
        items = [
            "CUMULATIVE TOTAL",
            f"${self.cumulative_pnl:.2f}",
            f"{cumulative_roi:.2f}%",
            str(self.cumulative_trades),
            str(self.cumulative_wins),
            f"{cumulative_win_rate:.2f}%",
            f"${self.cumulative_fees:.2f}"
        ]
        for i, text in enumerate(items):
            item = QTableWidgetItem(text); item.setFont(bold_font); self.table.setItem(r, i, item)

    def update_result_table(self, symbol: str, pnl: float, roi: float, trades: int, win_rate: float, total_equity_used: float, total_fees: float):
        self.cumulative_pnl += pnl
        self.cumulative_trades += trades
        self.cumulative_equity_used += total_equity_used
        self.cumulative_fees += total_fees
        wins = int(round(trades * (win_rate / 100.0)))
        self.cumulative_wins += wins
        self.cumulative_pnl_lbl.setText(f"Cumulative Session PnL: ${self.cumulative_pnl:.2f}")

        r = self.table.rowCount(); self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(symbol))
        self.table.setItem(r, 1, QTableWidgetItem(f"{pnl:.2f}"))
        self.table.setItem(r, 2, QTableWidgetItem(f"{roi:.2f}%"))
        self.table.setItem(r, 3, QTableWidgetItem(str(trades)))
        self.table.setItem(r, 4, QTableWidgetItem(str(wins)))
        self.table.setItem(r, 5, QTableWidgetItem(f"{win_rate:.2f}%"))
        self.table.setItem(r, 6, QTableWidgetItem(f"{total_fees:.2f}"))

    def clear_results(self):
        self.table.setRowCount(0)
        self._reset_cumulative_stats()
        self.cumulative_pnl_lbl.setText("Cumulative Session PnL: $0.00")
        self._log("Cleared backtest results.")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crypto Trader — SMC OB/FVG Backtester (Improved)")
        self.resize(1200, 800)
        main_widget = QWidget(); main_layout = QVBoxLayout(main_widget)
        splitter = QSplitter(Qt.Vertical)

        pairs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pairs.txt')
        pairs = load_pairs(pairs_path)
        if not pairs:
            logger.warning("pairs.txt not found, using default list.")
            pairs = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]

        self.bt_tab = BacktestTab(pairs)
        self.log = QTextEdit(); self.log.setReadOnly(True)
        splitter.addWidget(self.bt_tab); splitter.addWidget(self.log); splitter.setSizes([600, 200])
        main_layout.addWidget(splitter); self.setCentralWidget(main_widget)
        self.bt_tab.log_signal.connect(self.append_log)

    def append_log(self, msg: str):
        self.log.append(msg); logger.info(msg)

    def closeEvent(self, event):
        self.append_log("Closing application, stopping all active threads...")
        self.bt_tab.stop_all_backtests()
        time.sleep(0.5)
        for w in list(self.bt_tab.active_workers.values()):
            w.wait(1000)
        self.append_log("All threads stopped. Exiting.")
        event.accept()

# Entrypoint
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     w = MainWindow()
#     w.show()
#     sys.exit(app.exec_())

# -------------------------
# Quick console test (uncomment to run single pair in terminal for debugging)
# -------------------------
if __name__ == '__main__':
    # For now, we are only testing with the provided ADA/USDT data
    pairs = ["ADA/USDT"]

    logger.info(f"--- Running backtest for {len(pairs)} pairs ---")
    for pair in pairs:
        # The ExchangeWrapper is not strictly needed for local backtests,
        # but the worker expects it. We can pass None.
        wkr = BacktestWorker(pair, 10000.0, None, 90, fee_rate=EXCHANGE_FEE, gui_mode=False)
        wkr.run()
    logger.info("--- Backtest for all pairs complete ---")
