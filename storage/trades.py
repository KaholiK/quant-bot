"""
SQLite-based storage for trades, orders, equity, and logs.
Provides CSV/Parquet export functionality with optional Postgres support.
"""

import sqlite3
import os
import json
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from contextlib import contextmanager
from loguru import logger


class TradeStorage:
    """SQLite-based storage for trading data with export capabilities."""
    
    def __init__(self, db_path: str = "data/trading.db"):
        """
        Initialize trade storage.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.use_postgres = bool(os.getenv("POSTGRES_URL"))
        
        if self.use_postgres:
            logger.info("Postgres URL detected, will use PostgreSQL for storage")
            self.use_postgres = False
            logger.warning("PostgreSQL support not yet implemented, falling back to SQLite")
        
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.init_db()
    
    @contextmanager
    def get_connection(self):
        """Get database connection with proper cleanup."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()
    
    def init_db(self) -> None:
        """Initialize database tables if they don't exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    time TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    qty REAL NOT NULL,
                    avg_price REAL NOT NULL,
                    fees REAL DEFAULT 0.0,
                    slippage_bps REAL DEFAULT 0.0,
                    pnl REAL DEFAULT 0.0,
                    meta TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Orders table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT UNIQUE,
                    submitted TEXT NOT NULL,
                    filled TEXT,
                    status TEXT NOT NULL,
                    type TEXT NOT NULL,
                    price REAL,
                    qty REAL NOT NULL,
                    symbol TEXT NOT NULL,
                    meta TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Equity curve table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS equity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    equity REAL NOT NULL,
                    drawdown REAL DEFAULT 0.0,
                    realized_vol REAL DEFAULT 0.0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    level TEXT NOT NULL,
                    msg TEXT NOT NULL,
                    meta TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indices for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_time ON trades(time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_equity_ts ON equity(ts)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_ts ON logs(ts)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_level ON logs(level)")
            
            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")
    
    def record_order(self, order_event: Dict[str, Any]) -> None:
        """Record order event to database."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Extract order data
                order_id = order_event.get("order_id", "")
                submitted = order_event.get("submitted", datetime.utcnow().isoformat())
                filled = order_event.get("filled")
                status = order_event.get("status", "submitted")
                order_type = order_event.get("type", "market")
                price = order_event.get("price")
                qty = order_event.get("qty", 0.0)
                symbol = order_event.get("symbol", "")
                
                # Store metadata as JSON
                meta = {k: v for k, v in order_event.items() 
                       if k not in ["order_id", "submitted", "filled", "status", "type", "price", "qty", "symbol"]}
                meta_json = json.dumps(meta) if meta else None
                
                # Insert or update order
                cursor.execute("""
                    INSERT OR REPLACE INTO orders 
                    (order_id, submitted, filled, status, type, price, qty, symbol, meta)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (order_id, submitted, filled, status, order_type, price, qty, symbol, meta_json))
                
                conn.commit()
                logger.debug(f"Recorded order: {order_id} {symbol} {qty}@{price}")
                
        except Exception as e:
            logger.error(f"Failed to record order: {e}")
    
    def record_fill(self, fill_event: Dict[str, Any]) -> None:
        """Record trade fill to database."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Extract trade data
                time = fill_event.get("time", datetime.utcnow().isoformat())
                symbol = fill_event.get("symbol", "")
                side = fill_event.get("side", "")
                qty = fill_event.get("qty", 0.0)
                avg_price = fill_event.get("avg_price", 0.0)
                fees = fill_event.get("fees", 0.0)
                slippage_bps = fill_event.get("slippage_bps", 0.0)
                pnl = fill_event.get("pnl", 0.0)
                
                # Store metadata as JSON
                meta = {k: v for k, v in fill_event.items() 
                       if k not in ["time", "symbol", "side", "qty", "avg_price", "fees", "slippage_bps", "pnl"]}
                meta_json = json.dumps(meta) if meta else None
                
                cursor.execute("""
                    INSERT INTO trades (time, symbol, side, qty, avg_price, fees, slippage_bps, pnl, meta)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (time, symbol, side, qty, avg_price, fees, slippage_bps, pnl, meta_json))
                
                conn.commit()
                logger.info(f"Recorded fill: {symbol} {side} {qty}@{avg_price} (PnL: {pnl:.2f})")
                
        except Exception as e:
            logger.error(f"Failed to record fill: {e}")
    
    def record_equity(self, ts: str, equity: float, drawdown: float = 0.0, realized_vol: float = 0.0) -> None:
        """Record equity snapshot."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO equity (ts, equity, drawdown, realized_vol)
                    VALUES (?, ?, ?, ?)
                """, (ts, equity, drawdown, realized_vol))
                conn.commit()
                logger.debug(f"Recorded equity: {equity:.2f} (DD: {drawdown:.2%})")
                
        except Exception as e:
            logger.error(f"Failed to record equity: {e}")
    
    def record_log(self, level: str, msg: str, meta: Optional[Dict[str, Any]] = None) -> None:
        """Record log entry."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                ts = datetime.utcnow().isoformat()
                meta_json = json.dumps(meta) if meta else None
                
                cursor.execute("""
                    INSERT INTO logs (ts, level, msg, meta)
                    VALUES (?, ?, ?, ?)
                """, (ts, level, msg, meta_json))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to record log: {e}")
    
    def list_trades(self, 
                   symbol: Optional[str] = None,
                   start_time: Optional[str] = None,
                   end_time: Optional[str] = None,
                   limit: int = 1000,
                   offset: int = 0) -> List[Dict[str, Any]]:
        """List trades with optional filters."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Build query with filters
                query = "SELECT * FROM trades WHERE 1=1"
                params = []
                
                if symbol:
                    query += " AND symbol = ?"
                    params.append(symbol)
                
                if start_time:
                    query += " AND time >= ?"
                    params.append(start_time)
                
                if end_time:
                    query += " AND time <= ?"
                    params.append(end_time)
                
                query += " ORDER BY time DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # Convert to dictionaries
                trades = []
                for row in rows:
                    trade = dict(row)
                    # Parse metadata JSON
                    if trade["meta"]:
                        try:
                            trade["meta"] = json.loads(trade["meta"])
                        except json.JSONDecodeError:
                            trade["meta"] = {}
                    trades.append(trade)
                
                return trades
                
        except Exception as e:
            logger.error(f"Failed to list trades: {e}")
            return []
    
    def get_equity_curve(self, 
                        start_time: Optional[str] = None,
                        end_time: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get equity curve data."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                query = "SELECT ts, equity, drawdown, realized_vol FROM equity WHERE 1=1"
                params = []
                
                if start_time:
                    query += " AND ts >= ?"
                    params.append(start_time)
                
                if end_time:
                    query += " AND ts <= ?"
                    params.append(end_time)
                
                query += " ORDER BY ts ASC"
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get equity curve: {e}")
            return []
    
    def export_trades_csv(self, output_path: str, **filters) -> bool:
        """Export trades to CSV file."""
        try:
            trades = self.list_trades(**filters)
            if not trades:
                logger.warning("No trades to export")
                return False
            
            # Convert to DataFrame
            df = pd.DataFrame(trades)
            
            # Flatten metadata column if present
            if "meta" in df.columns:
                df = df.drop("meta", axis=1)  # Skip complex JSON for CSV
            
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            df.to_csv(output_path, index=False)
            logger.info(f"Exported {len(trades)} trades to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export trades to CSV: {e}")
            return False
    
    def export_trades_parquet(self, output_path: str, **filters) -> bool:
        """Export trades to Parquet file."""
        try:
            trades = self.list_trades(**filters)
            if not trades:
                logger.warning("No trades to export")
                return False
            
            # Convert to DataFrame
            df = pd.DataFrame(trades)
            
            # Handle metadata column for Parquet
            if "meta" in df.columns:
                # Convert meta dict to JSON string for Parquet compatibility
                df["meta"] = df["meta"].apply(lambda x: json.dumps(x) if x else None)
            
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            df.to_parquet(output_path, index=False)
            logger.info(f"Exported {len(trades)} trades to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export trades to Parquet: {e}")
            return False
    
    def get_trade_stats(self) -> Dict[str, Any]:
        """Get basic trade statistics."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get basic counts
                cursor.execute("SELECT COUNT(*) as total_trades FROM trades")
                total_trades = cursor.fetchone()["total_trades"]
                
                cursor.execute("SELECT COUNT(DISTINCT symbol) as unique_symbols FROM trades")
                unique_symbols = cursor.fetchone()["unique_symbols"]
                
                cursor.execute("SELECT SUM(pnl) as total_pnl FROM trades")
                total_pnl = cursor.fetchone()["total_pnl"] or 0.0
                
                cursor.execute("SELECT COUNT(*) as winning_trades FROM trades WHERE pnl > 0")
                winning_trades = cursor.fetchone()["winning_trades"]
                
                # Calculate win rate
                win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
                
                # Get latest equity
                cursor.execute("SELECT equity, drawdown FROM equity ORDER BY ts DESC LIMIT 1")
                equity_row = cursor.fetchone()
                latest_equity = equity_row["equity"] if equity_row else 0.0
                latest_drawdown = equity_row["drawdown"] if equity_row else 0.0
                
                return {
                    "total_trades": total_trades,
                    "unique_symbols": unique_symbols,
                    "total_pnl": total_pnl,
                    "winning_trades": winning_trades,
                    "win_rate": win_rate,
                    "latest_equity": latest_equity,
                    "latest_drawdown": latest_drawdown,
                }
                
        except Exception as e:
            logger.error(f"Failed to get trade stats: {e}")
            return {}


# Global instance
_trade_storage: Optional[TradeStorage] = None


def get_trade_storage(db_path: str = "data/trading.db") -> TradeStorage:
    """Get global trade storage instance."""
    global _trade_storage
    if _trade_storage is None:
        _trade_storage = TradeStorage(db_path)
    return _trade_storage