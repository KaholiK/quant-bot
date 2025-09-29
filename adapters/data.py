"""
Data adapter interface for portability across different data sources.
Provides unified interface for historical and live market data.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from datetime import datetime, timedelta


class DataAdapter(ABC):
    """
    Abstract data adapter interface.
    
    Provides unified access to market data from different sources.
    """
    
    @abstractmethod
    def get_historical_data(self, 
                          symbols: List[str],
                          start_date: datetime,
                          end_date: datetime,
                          timeframe: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Get historical OHLCV data.
        
        Args:
            symbols: List of symbols to fetch
            start_date: Start date
            end_date: End date
            timeframe: Timeframe ("1m", "5m", "1h", "1d", etc.)
            
        Returns:
            Dict of {symbol: DataFrame} with OHLCV data
        """
        pass
    
    @abstractmethod
    def get_live_price(self, symbol: str) -> float:
        """Get current live price for symbol."""
        pass


class YFinanceAdapter(DataAdapter):
    """Yahoo Finance data adapter for training scripts."""
    
    def __init__(self):
        try:
            import yfinance as yf
            self.yf = yf
        except ImportError:
            raise ImportError("yfinance not installed. Run: pip install yfinance")
    
    def get_historical_data(self, 
                          symbols: List[str],
                          start_date: datetime,
                          end_date: datetime,
                          timeframe: str = "1d") -> Dict[str, pd.DataFrame]:
        """Get historical data from Yahoo Finance."""
        data = {}
        
        for symbol in symbols:
            try:
                ticker = self.yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, interval=timeframe)
                
                if not df.empty:
                    # Standardize column names
                    df = df.rename(columns={
                        'Open': 'open', 'High': 'high', 'Low': 'low', 
                        'Close': 'close', 'Volume': 'volume'
                    })
                    data[symbol] = df
                    
            except Exception as e:
                print(f"Failed to fetch {symbol}: {e}")
                continue
        
        return data
    
    def get_live_price(self, symbol: str) -> float:
        """Get current price from Yahoo Finance."""
        try:
            ticker = self.yf.Ticker(symbol)
            return float(ticker.info.get('regularMarketPrice', 0.0))
        except:
            return 0.0


class BinanceAdapter(DataAdapter):
    """Binance data adapter for crypto data."""
    
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
    
    def get_historical_data(self, 
                          symbols: List[str],
                          start_date: datetime,
                          end_date: datetime,
                          timeframe: str = "1d") -> Dict[str, pd.DataFrame]:
        """Get historical data from Binance public API."""
        import requests
        
        data = {}
        
        # Convert timeframe
        interval_map = {
            "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1h", "4h": "4h", "1d": "1d"
        }
        
        interval = interval_map.get(timeframe, "1d")
        
        for symbol in symbols:
            try:
                # Convert symbol format (BTCUSD -> BTCUSDT for Binance)
                binance_symbol = symbol.replace("USD", "USDT")
                
                params = {
                    "symbol": binance_symbol,
                    "interval": interval,
                    "startTime": int(start_date.timestamp() * 1000),
                    "endTime": int(end_date.timestamp() * 1000),
                    "limit": 1000
                }
                
                response = requests.get(f"{self.base_url}/klines", params=params)
                response.raise_for_status()
                
                klines = response.json()
                
                if klines:
                    df = pd.DataFrame(klines, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                    ])
                    
                    # Convert types and set index
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col])
                    
                    df = df.set_index('timestamp')
                    df = df[['open', 'high', 'low', 'close', 'volume']]
                    
                    data[symbol] = df
                    
            except Exception as e:
                print(f"Failed to fetch {symbol} from Binance: {e}")
                continue
        
        return data
    
    def get_live_price(self, symbol: str) -> float:
        """Get current price from Binance."""
        import requests
        
        try:
            binance_symbol = symbol.replace("USD", "USDT")
            response = requests.get(f"{self.base_url}/ticker/price", 
                                  params={"symbol": binance_symbol})
            response.raise_for_status()
            return float(response.json()["price"])
        except:
            return 0.0


def create_data_adapter(source: str) -> DataAdapter:
    """
    Factory function to create data adapters.
    
    Args:
        source: Data source ("yfinance", "binance", etc.)
        
    Returns:
        DataAdapter instance
    """
    if source.lower() == "yfinance":
        return YFinanceAdapter()
    elif source.lower() == "binance":
        return BinanceAdapter()
    else:
        raise ValueError(f"Unsupported data source: {source}")