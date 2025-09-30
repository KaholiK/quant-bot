"""
Data adapter interface for portability across different data sources.  
Provides unified interface for historical and live market data with retry and rate limiting.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from datetime import datetime, timedelta
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging

logger = logging.getLogger(__name__)


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
    
    def normalize_kline_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize kline data to standard OHLCV format."""
        # Ensure standard column names
        column_map = {
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'close'  # Use adjusted close if available
        }
        
        # Rename columns
        for old_name, new_name in column_map.items():
            if old_name in data.columns:
                data = data.rename(columns={old_name: new_name})
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                logger.warning(f"Missing required column: {col}")
                data[col] = 0.0
        
        # Add volume if missing
        if 'volume' not in data.columns:
            data['volume'] = 0.0
        
        # Sort by index (timestamp)
        data = data.sort_index()
        
        return data[['open', 'high', 'low', 'close', 'volume']]


class YFinanceAdapter(DataAdapter):
    """Yahoo Finance data adapter with retry logic and rate limiting."""
    
    def __init__(self, rate_limit_delay: float = 0.1):
        """Initialize YFinance adapter with rate limiting."""
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0.0
        
        try:
            import yfinance as yf
            self.yf = yf
            
            # Setup session with retry strategy
            self.session = requests.Session()
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)
            
            logger.info("YFinance adapter initialized")
        except ImportError:
            logger.error("yfinance package not installed")
            self.yf = None
    
    def _rate_limit(self):
        """Apply rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
    
    def get_historical_data(self, 
                          symbols: List[str],
                          start_date: datetime,
                          end_date: datetime,
                          timeframe: str = "1d") -> Dict[str, pd.DataFrame]:
        """Get historical data from Yahoo Finance with retry logic."""
        if self.yf is None:
            logger.error("YFinance not available")
            return {}
        
        data_dict = {}
        
        for symbol in symbols:
            try:
                self._rate_limit()
                
                # Convert symbol for yfinance
                yf_symbol = self._convert_symbol_to_yf(symbol)
                
                # Create ticker object
                ticker = self.yf.Ticker(yf_symbol, session=self.session)
                
                # Download data with retry
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        data = ticker.history(
                            start=start_date,
                            end=end_date,
                            interval=self._convert_timeframe(timeframe)
                        )
                        
                        if not data.empty:
                            # Normalize the data
                            normalized_data = self.normalize_kline_data(data)
                            data_dict[symbol] = normalized_data
                            logger.debug(f"Downloaded {len(normalized_data)} bars for {symbol}")
                            break
                        else:
                            logger.warning(f"No data returned for {symbol}")
                            
                    except Exception as e:
                        if attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 2  # Exponential backoff
                            logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {e}. Retrying in {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            logger.error(f"Failed to download data for {symbol} after {max_retries} attempts: {e}")
                            
            except Exception as e:
                logger.error(f"Error downloading {symbol}: {e}")
                
        return data_dict
    
    def get_live_price(self, symbol: str) -> float:
        """Get current live price from Yahoo Finance."""
        if self.yf is None:
            return 0.0
            
        try:
            self._rate_limit()
            
            yf_symbol = self._convert_symbol_to_yf(symbol)
            ticker = self.yf.Ticker(yf_symbol, session=self.session)
            
            # Get current price
            info = ticker.info
            price = info.get('regularMarketPrice') or info.get('previousClose', 0.0)
            
            return float(price)
            
        except Exception as e:
            logger.error(f"Error getting live price for {symbol}: {e}")
            return 0.0
    
    def _convert_symbol_to_yf(self, symbol: str) -> str:
        """Convert internal symbol to Yahoo Finance format."""
        # Handle crypto symbols
        if symbol.upper() in ['BTCUSD', 'BTC-USD']:
            return 'BTC-USD'
        elif symbol.upper() in ['ETHUSD', 'ETH-USD']:
            return 'ETH-USD'
        
        # Handle equity symbols (most are direct)
        return symbol.upper()
    
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert timeframe to Yahoo Finance format."""
        timeframe_map = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '1d': '1d',
            '1w': '1wk',
            '1M': '1mo'
        }
        
        return timeframe_map.get(timeframe, '1d')


class BinanceAdapter(DataAdapter):
    """Binance data adapter for crypto data with rate limiting."""
    
    def __init__(self, rate_limit_delay: float = 0.1):
        """Initialize Binance adapter."""
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0.0
        self.base_url = "https://api.binance.com/api/v3"
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        logger.info("Binance adapter initialized")
    
    def _rate_limit(self):
        """Apply rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
    
    def get_historical_data(self, 
                          symbols: List[str],
                          start_date: datetime,
                          end_date: datetime,
                          timeframe: str = "1d") -> Dict[str, pd.DataFrame]:
        """Get historical kline data from Binance."""
        data_dict = {}
        
        for symbol in symbols:
            try:
                self._rate_limit()
                
                # Convert symbol for Binance
                binance_symbol = self._convert_symbol_to_binance(symbol)
                
                # Skip non-crypto symbols
                if not binance_symbol:
                    continue
                
                # Get kline data
                klines = self._get_klines(binance_symbol, timeframe, start_date, end_date)
                
                if klines:
                    # Convert to DataFrame
                    df = pd.DataFrame(klines, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'trades_count', 
                        'taker_buy_base', 'taker_buy_quote', 'ignore'
                    ])
                    
                    # Process the data
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df = df.set_index('timestamp')
                    
                    # Convert price/volume columns to float
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Normalize the data
                    normalized_data = self.normalize_kline_data(df)
                    data_dict[symbol] = normalized_data
                    
                    logger.debug(f"Downloaded {len(normalized_data)} klines for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error downloading {symbol} from Binance: {e}")
                
        return data_dict
    
    def get_live_price(self, symbol: str) -> float:
        """Get current live price from Binance."""
        try:
            self._rate_limit()
            
            binance_symbol = self._convert_symbol_to_binance(symbol)
            if not binance_symbol:
                return 0.0
            
            url = f"{self.base_url}/ticker/price"
            params = {'symbol': binance_symbol}
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return float(data['price'])
            
        except Exception as e:
            logger.error(f"Error getting live price for {symbol} from Binance: {e}")
            return 0.0
    
    def _get_klines(self, symbol: str, interval: str, start_date: datetime, end_date: datetime) -> List:
        """Get kline data from Binance API."""
        url = f"{self.base_url}/klines"
        
        # Convert interval
        binance_interval = self._convert_timeframe(interval)
        
        # Convert dates to milliseconds
        start_ms = int(start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)
        
        params = {
            'symbol': symbol,
            'interval': binance_interval,
            'startTime': start_ms,
            'endTime': end_ms,
            'limit': 1000  # Max limit per request
        }
        
        response = self.session.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        return response.json()
    
    def _convert_symbol_to_binance(self, symbol: str) -> Optional[str]:
        """Convert internal symbol to Binance format."""
        symbol = symbol.upper()
        
        # Handle crypto pairs
        if symbol in ['BTCUSD', 'BTC-USD']:
            return 'BTCUSDT'
        elif symbol in ['ETHUSD', 'ETH-USD']:
            return 'ETHUSDT'
        elif symbol.endswith('USD'):
            # Convert other crypto symbols
            base = symbol[:-3]
            return f"{base}USDT"
        
        # Skip non-crypto symbols
        return None
    
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert timeframe to Binance format."""
        timeframe_map = {
            '1m': '1m',
            '5m': '5m', 
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '1d': '1d',
            '1w': '1w',
            '1M': '1M'
        }
        
        return timeframe_map.get(timeframe, '1d')
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