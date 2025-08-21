#!/usr/bin/env python3
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
BINANCE_FEE = 0.001
EQUITY_PER_TRADE = 1            # 1 = full capital per coin (paper)
SL_PERCENTAGE = 0.01            # fallback SL pad below zone bottom
RESISTANCE_LOOKFORWARD = 200
CHOCH_MAX_LOOKBACK = 60
REBOUND_LOOKFORWARD = 200

RSI_LEN = 14
RSI_MIN = 45                    # momentum filter
RSI_SLOPE_MIN = 0.0             # require RSI rising vs previous bar

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

def ohlcv_to_df(ohlcv: List[List[float]]) -> pd.DataFrame:
    df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

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
    try:
        swing_highs_lows = smc.swing_highs_lows(df)
    except Exception:
        swing_highs_lows = None

    # CHOCH/BOS
    try:
        choch = smc.bos_choch(df, swing_highs_lows=swing_highs_lows)
        df['CHOCH'] = choch['CHOCH'] if 'CHOCH' in choch.columns else 0
    except Exception:
        df['CHOCH'] = 0

    # OB
    try:
        order_blocks = smc.ob(df, swing_highs_lows=swing_highs_lows)
        df['OB'] = order_blocks['OB'] if 'OB' in order_blocks.columns else 0
        df['OB_bottom'] = order_blocks['Bottom'] if 'Bottom' in order_blocks.columns else pd.NA
        df['OB_top'] = order_blocks['Top'] if 'Top' in order_blocks.columns else pd.NA
    except Exception:
        df['OB'] = 0
        df['OB_bottom'] = pd.NA
        df['OB_top'] = pd.NA

    # FVG
    try:
        fvgs = smc.fvg(df)
        df['FVG'] = fvgs['FVG'] if 'FVG' in fvgs.columns else 0
        df['FVG_bottom'] = fvgs['Bottom'] if 'Bottom' in fvgs.columns else pd.NA
        df['FVG_top'] = fvgs['Top'] if 'Top' in fvgs.columns else pd.NA
    except Exception:
        df['FVG'] = 0
        df['FVG_bottom'] = pd.NA
        df['FVG_top'] = pd.NA

    # RSI
    try:
        df['RSI'] = rsi_series(df['close'].astype(float), RSI_LEN)
    except Exception:
        df['RSI'] = 50.0

    # numeric safety
    for col in ['CHOCH','OB','FVG']:
        if col not in df.columns:
            df[col] = 0
    df[['open','high','low','close']] = df[['open','high','low','close']].astype(float)
    return df

def is_one_flag(val) -> bool:
    try:
        if val is None: return False
        if isinstance(val, (bool, np.bool_)): return bool(val)
        if isinstance(val, (int, np.integer)): return int(val) == 1
        if isinstance(val, (float, np.floating)):
            if np.isnan(val): return False
            return int(val) == 1
        return int(val) == 1
    except Exception:
        return False

def find_last_ob_before(df: pd.DataFrame, idx: int, max_lookback: int = CHOCH_MAX_LOOKBACK) -> Optional[int]:
    start = max(0, idx - max_lookback)
    for j in range(idx-1, start-1, -1):
        try:
            if is_one_flag(df.at[j, 'OB']): return j
        except Exception:
            continue
    return None

def find_last_fvg_before(df: pd.DataFrame, idx: int, max_lookback: int = CHOCH_MAX_LOOKBACK) -> Optional[int]:
    start = max(0, idx - max_lookback)
    for j in range(idx-1, start-1, -1):
        try:
            if is_one_flag(df.at[j, 'FVG']): return j
        except Exception:
            continue
    # fallback 3-bar gap
    for j in range(idx-3, start-1, -1):
        if j < 0: break
        try:
            if df['low'].iat[j+2] > df['high'].iat[j]:
                return j
        except Exception:
            continue
    return None

def is_bullish_rejection(df: pd.DataFrame, bar_idx: int, zone_top: float, zone_bot: float) -> bool:
    if zone_top is None or zone_bot is None: return False
    if bar_idx <= 0 or bar_idx >= len(df): return False
    o = df['open'].iat[bar_idx]; c = df['close'].iat[bar_idx]; h = df['high'].iat[bar_idx]; l = df['low'].iat[bar_idx]
    prev_c = df['close'].iat[bar_idx-1]
    # Touch zone and close strong
    touches = ((l <= zone_top and h >= zone_bot) or (l <= zone_bot and h >= zone_top) or (zone_bot <= l <= zone_top) or (zone_bot <= h <= zone_top))
    # bullish candle or engulfing
    body_up = (c > o) and (c > prev_c)
    engulf = (c > df['open'].iat[bar_idx-1]) and (o <= df['close'].iat[bar_idx-1])
    hammer_like = (c > o) and ((o - l) > (h - c)) and ((o - l) > (c - o))
    return bool(touches and (body_up or engulf or hammer_like))

def _pivot_highs(series: pd.Series, left: int = 2, right: int = 2) -> pd.Series:
    """Return boolean series where a pivot high is True."""
    highs = series.values
    n = len(highs)
    piv = np.zeros(n, dtype=bool)
    for i in range(left, n - right):
        window_left = highs[i - left:i]
        window_right = highs[i+1:i+1+right]
        if np.all(highs[i] > window_left) and np.all(highs[i] >= window_right):
            piv[i] = True
    return pd.Series(piv, index=series.index)

def find_forward_resistances(df: pd.DataFrame, entry_idx: int, lookforward: int = RESISTANCE_LOOKFORWARD, max_levels: int = 2) -> List[Tuple[int, float]]:
    """
    Auto resistance detection:
      - scan forward up to lookforward bars
      - collect pivot highs (left/right=2)
      - return up to max_levels (closest then next)
    """
    end = min(len(df)-1, entry_idx + lookforward)
    if entry_idx + 3 > end:
        return []
    ph = _pivot_highs(df['high'], left=2, right=2)
    levels: List[Tuple[int, float]] = []
    for j in range(entry_idx+1, end+1):
        if bool(ph.iat[j]):
            levels.append((j, float(df['high'].iat[j])))
            if len(levels) >= max_levels:
                break
    # Fallbacks if not enough pivots found
    if not levels:
        # use the maximum high ahead as TP
        seg = df['high'].iloc[entry_idx+1:end+1]
        if not seg.empty:
            idx = int(seg.idxmax())
            levels.append((idx, float(seg.max())))
    return levels[:max_levels]

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
        api_key = os.getenv('BINANCE_API_KEY', '')
        api_secret = os.getenv('BINANCE_API_SECRET', '')
        options = {'defaultType': 'spot', 'fetchCurrencies': False}
        self.client = ccxt.binance({
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

    def __init__(self, symbol: str, capital: float, exchange: ExchangeWrapper, lookback_days: int):
        super().__init__()
        self.symbol = symbol
        self.capital = capital
        self.exchange = exchange
        self.lookback_days = int(lookback_days)
        self._running = True

    def stop(self):
        self._running = False

    def log_emit(self, msg: str):
        try:
            self.update_status.emit(msg)
        except Exception:
            logger.info(msg)

    def run(self):
        try:
            self.log_emit(f"[BT] Fetching data for {self.symbol}...")
            now_ms = int(time.time() * 1000)
            since_ms = now_ms - (self.lookback_days + 10) * 24 * 60 * 60 * 1000

            all_ohlcv = []
            since = since_ms
            while self._running:
                try:
                    batch = self.exchange.fetch_ohlcv(self.symbol, TRADE_TIMEFRAME, since=since, limit=1000)
                except ccxt.BadSymbol:
                    self.log_emit(f"[BT] Error: Symbol '{self.symbol}' not found. Skipping.")
                    return
                except Exception as e:
                    self.log_emit(f"[BT] Warning fetching {self.symbol}: {e}")
                    logger.exception(f"fetch batch error for {self.symbol}: {e}")
                    break

                if not batch:
                    break
                all_ohlcv.extend(batch)
                since = batch[-1][0] + 1
                time.sleep(self.exchange.rate_limit_s)

            if not all_ohlcv:
                self.log_emit(f"[BT] No data for {self.symbol}")
                self.finished_pair.emit(self.symbol)
                return

            df = compute_indicators(ohlcv_to_df(all_ohlcv))
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
            choch_detected_index = -1

            for i in range(1, len(df)):
                if not self._running:
                    break
                row = df.iloc[i]

                # ---------- MANAGE OPEN POSITION ----------
                if position:
                    exit_price = None
                    exit_reason = ""

                    # Stop first (conservative)
                    try:
                        if row['low'] <= position.stop_loss_price:
                            exit_price = position.stop_loss_price
                            exit_reason = "Stop-Loss"
                            # If half remained (after partial), close remaining
                            entry_cost = position.size * position.entry_price
                            exit_value = position.size * exit_price
                            fee = (entry_cost + exit_value) * BINANCE_FEE
                            pnl = exit_value - entry_cost - fee
                            trades.append({'pnl': pnl})
                            total_fees += fee
                            capital += pnl
                            self.log_emit(f"[BT] SELL {self.symbol} @ {exit_price:.4f} ({exit_reason}). PnL: ${pnl:.2f}")
                            position = None
                            choch_detected_index = -1
                            continue
                    except Exception:
                        logger.exception("Error SL check")

                    # TP1 partial exit (50%)
                    try:
                        if (not position.half_closed) and row['high'] >= position.tp1_price:
                            half_size = position.size * 0.5
                            entry_cost = half_size * position.entry_price
                            exit_value = half_size * position.tp1_price
                            fee = (entry_cost + exit_value) * BINANCE_FEE
                            pnl = exit_value - entry_cost - fee
                            total_fees += fee
                            capital += pnl
                            trades.append({'pnl': pnl})
                            position.size = position.size - half_size
                            position.half_closed = True
                            # Move SL to breakeven after TP1
                            position.stop_loss_price = position.entry_price
                            self.log_emit(f"[BT] TP1 {self.symbol} @ {position.tp1_price:.4f} (50% close, BE SL). PnL: ${pnl:.2f}")
                            # do not continue; allow TP2 in same bar if high >= tp2
                    except Exception:
                        logger.exception("Error TP1 check")

                    # TP2 final exit
                    try:
                        if row['high'] >= position.tp2_price:
                            entry_cost = position.size * position.entry_price
                            exit_value = position.size * position.tp2_price
                            fee = (entry_cost + exit_value) * BINANCE_FEE
                            pnl = exit_value - entry_cost - fee
                            total_fees += fee
                            capital += pnl
                            trades.append({'pnl': pnl})
                            self.log_emit(f"[BT] TP2 {self.symbol} @ {position.tp2_price:.4f} (final). PnL: ${pnl:.2f}")
                            position = None
                            choch_detected_index = -1
                            continue
                    except Exception:
                        logger.exception("Error TP2 check")

                # ---------- ENTRY LOGIC ----------
                if position is None:
                    choch_flag = row.get('CHOCH', 0)
                    # If we get a bullish CHOCH, arm the entry window
                    if is_one_flag(choch_flag):
                        choch_detected_index = i
                        self.log_emit(f"[BT] Bullish CHOCH detected for {self.symbol} idx {i} price {row['close']:.4f}")
                        continue

                    # After CHOCH, wait for rebound into OB or FVG + bullish signal + momentum
                    if choch_detected_index != -1 and i > choch_detected_index:
                        ob_idx = find_last_ob_before(df, choch_detected_index, max_lookback=CHOCH_MAX_LOOKBACK)
                        fvg_idx = find_last_fvg_before(df, choch_detected_index, max_lookback=CHOCH_MAX_LOOKBACK)

                        entry_price = None
                        stop_loss_price = None
                        zone_top = None
                        zone_bot = None
                        entry_reason = None

                        # OB setup
                        if ob_idx is not None:
                            zone_top = (df['OB_top'].iat[ob_idx]
                                        if pd.notna(df['OB_top'].iat[ob_idx]) else df['high'].iat[ob_idx])
                            zone_bot = (df['OB_bottom'].iat[ob_idx]
                                        if pd.notna(df['OB_bottom'].iat[ob_idx]) else df['low'].iat[ob_idx])
                            if is_bullish_rejection(df, i, zone_top, zone_bot):
                                entry_price = float(df['close'].iat[i])
                                stop_loss_price = float(zone_bot) * (1 - SL_PERCENTAGE)
                                entry_reason = f"OB Rebound (idx {ob_idx})"

                        # FVG setup if no OB entry
                        if entry_price is None and fvg_idx is not None:
                            if pd.notna(df['FVG_top'].iat[fvg_idx]) and pd.notna(df['FVG_bottom'].iat[fvg_idx]):
                                zone_top = float(df['FVG_top'].iat[fvg_idx])
                                zone_bot = float(df['FVG_bottom'].iat[fvg_idx])
                            else:
                                if fvg_idx + 2 < len(df):
                                    zone_top = float(df['high'].iat[fvg_idx])
                                    zone_bot = float(df['low'].iat[fvg_idx+2])
                            if zone_top is not None and zone_bot is not None and is_bullish_rejection(df, i, zone_top, zone_bot):
                                entry_price = float(df['close'].iat[i])
                                stop_loss_price = float(zone_bot) * (1 - SL_PERCENTAGE)
                                entry_reason = f"FVG Rebound (idx {fvg_idx})"

                        # Momentum filter (RSI rising & above RSI_MIN)
                        if entry_price is not None and stop_loss_price is not None:
                            rsi_now = float(df['RSI'].iat[i])
                            rsi_prev = float(df['RSI'].iat[i-1]) if i > 0 else rsi_now
                            if (rsi_now >= RSI_MIN) and ((rsi_now - rsi_prev) >= RSI_SLOPE_MIN):
                                # Auto resistance — two levels ahead
                                levels = find_forward_resistances(df, i, lookforward=RESISTANCE_LOOKFORWARD, max_levels=2)
                                if len(levels) == 0:
                                    # fallback: +10% TP
                                    tp1_price = entry_price * 1.05
                                    tp2_price = entry_price * 1.10
                                elif len(levels) == 1:
                                    tp1_price = max(levels[0][1], entry_price * 1.02)
                                    # tp2 as extension (1.5x distance from SL or +7%)
                                    rr = abs(entry_price - stop_loss_price)
                                    tp2_price = max(tp1_price + 1.5 * rr, entry_price * 1.07)
                                else:
                                    tp1_price = max(levels[0][1], entry_price * 1.02)
                                    tp2_price = max(levels[1][1], tp1_price * 1.02)

                                amount_to_invest = capital * EQUITY_PER_TRADE
                                size = amount_to_invest / entry_price if entry_price > 0 else 0.0

                                position = Position(
                                    symbol=self.symbol,
                                    size=size,
                                    entry_price=entry_price,
                                    stop_loss_price=stop_loss_price,
                                    tp1_price=tp1_price,
                                    tp2_price=tp2_price,
                                    half_closed=False
                                )
                                total_equity_used += amount_to_invest
                                self.log_emit(f"[BT] BUY {self.symbol} @ {entry_price:.4f} ({entry_reason}) "
                                              f"SL {stop_loss_price:.4f} TP1 {tp1_price:.4f} TP2 {tp2_price:.4f}")
                                choch_detected_index = -1
                                continue

                        # Time out the CHOCH if no rebound seen
                        if i >= choch_detected_index + REBOUND_LOOKFORWARD:
                            choch_detected_index = -1

            # close open position at EOD (market close of backtest)
            if position:
                sell_price = float(df.iloc[-1]['close'])
                # If half was closed, current size is the remainder already
                entry_cost = position.size * position.entry_price
                exit_value = position.size * sell_price
                fees = (entry_cost + exit_value) * BINANCE_FEE
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

            self.update_result.emit(self.symbol, total_pnl, roi, num_trades, win_rate, total_equity_used, total_fees)
            self.log_emit(f"[BT] Done {self.symbol}: PnL ${total_pnl:.2f}, Trades {num_trades}, WinRate {win_rate:.2f}%, ROI {roi:.2f}%")

        except Exception as e:
            logger.exception(f"Worker fatal error for {self.symbol}: {e}")
            self.log_emit(f"[BT] Worker Error on {self.symbol}: {e}")
        finally:
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
        worker = BacktestWorker(pair, self.capital_input.value(), ex, self.days.value())
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
if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

# -------------------------
# Quick console test (uncomment to run single pair in terminal for debugging)
# -------------------------
# if False:
#     ex = ExchangeWrapper()
#     wkr = BacktestWorker("BTC/USDT", 10000.0, ex, 365)
#     wkr.run()
