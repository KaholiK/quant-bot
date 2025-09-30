"""
Broker adapter interface for portability across different platforms.
Provides a unified interface for order management and market data access.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled" 
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Order data structure."""
    order_id: str
    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: float
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class BrokerAdapter(ABC):
    """
    Abstract broker adapter interface.
    
    This provides a unified interface for different brokers/platforms.
    Implementations should handle platform-specific details.
    """
    
    @abstractmethod
    def place_order(self, 
                   symbol: str,
                   side: str,
                   quantity: float,
                   order_type: OrderType,
                   limit_price: Optional[float] = None,
                   stop_price: Optional[float] = None,
                   time_in_force: str = "GTC") -> Order:
        """Place a new order."""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        pass
    
    @abstractmethod
    def flatten_all(self) -> List[Order]:
        """Flatten all positions with market orders."""
        pass
    
    @abstractmethod
    def get_price(self, symbol: str) -> float:
        """Get current market price for symbol."""
        pass
    
    @abstractmethod
    def get_spread(self, symbol: str) -> Tuple[float, float]:
        """Get current bid/ask spread."""
        pass


class QuantConnectAdapter(BrokerAdapter):
    """QuantConnect LEAN adapter implementation."""
    
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self._orders = {}
    
    def place_order(self, symbol: str, side: str, quantity: float, order_type: OrderType,
                   limit_price: Optional[float] = None, stop_price: Optional[float] = None,
                   time_in_force: str = "GTC") -> Order:
        """Place order via QuantConnect."""
        qc_symbol = self.algorithm.Symbol(symbol)
        qc_quantity = quantity if side == "BUY" else -quantity
        
        if order_type == OrderType.MARKET:
            order_ticket = self.algorithm.MarketOrder(qc_symbol, qc_quantity)
        elif order_type == OrderType.LIMIT:
            order_ticket = self.algorithm.LimitOrder(qc_symbol, qc_quantity, limit_price)
        else:
            raise ValueError(f"Unsupported order type: {order_type}")
        
        order = Order(
            order_id=str(order_ticket.OrderId),
            symbol=symbol,
            side=side,
            quantity=abs(quantity),
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price
        )
        
        self._orders[order.order_id] = order
        return order
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order via QuantConnect."""
        try:
            order_ticket = self.algorithm.Transactions.GetOrderTicket(int(order_id))
            response = order_ticket.Cancel()
            return response.IsSuccess
        except Exception:
            return False
    
    def flatten_all(self) -> List[Order]:
        """Flatten all positions via QuantConnect."""
        orders = []
        for kvp in self.algorithm.Portfolio:
            holding = kvp.Value
            if holding.Quantity != 0:
                symbol = str(kvp.Key.Value)
                side = "SELL" if holding.Quantity > 0 else "BUY"
                quantity = abs(float(holding.Quantity))
                
                order = self.place_order(symbol, side, quantity, OrderType.MARKET)
                orders.append(order)
        return orders
    
    def get_price(self, symbol: str) -> float:
        """Get current price from QuantConnect."""
        qc_symbol = self.algorithm.Symbol(symbol)
        security = self.algorithm.Securities.get(qc_symbol)
        if security is not None:
            return float(security.Price)
        return 0.0
    
    def get_spread(self, symbol: str) -> Tuple[float, float]:
        """Get current bid/ask spread from QuantConnect."""
        qc_symbol = self.algorithm.Symbol(symbol)
        security = self.algorithm.Securities.get(qc_symbol)
        if security is not None:
            bid = float(security.BidPrice) if security.BidPrice > 0 else float(security.Price)
            ask = float(security.AskPrice) if security.AskPrice > 0 else float(security.Price)
            return bid, ask
        return 0.0, 0.0
    
    def normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol for QuantConnect format."""
        return symbol.upper()


class AlpacaAdapter(BrokerAdapter):
    """Alpaca Markets adapter implementation (stub)."""
    
    def __init__(self, api_key: str, secret_key: str, paper_trading: bool = True):
        """Initialize Alpaca adapter."""
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper_trading = paper_trading
        self._orders = {}
        
        # In practice, would initialize alpaca-trade-api client
        self.client = None  # Mock client
        
    def place_order(self, symbol: str, side: str, quantity: float, order_type: OrderType,
                   limit_price: Optional[float] = None, stop_price: Optional[float] = None,
                   time_in_force: str = "GTC") -> Order:
        """Place order via Alpaca API."""
        # Normalize symbol for Alpaca
        alpaca_symbol = self.normalize_symbol(symbol)
        
        # Mock order placement - in practice would use Alpaca API
        order_id = f"alpaca_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        order = Order(
            order_id=order_id,
            symbol=alpaca_symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price
        )
        
        self._orders[order_id] = order
        
        # Log the order details
        print(f"Alpaca order placed: {order_id} {side} {quantity} {alpaca_symbol} @ {limit_price or 'market'}")
        
        return order
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order via Alpaca API."""
        # Mock cancellation - in practice would call Alpaca API
        if order_id in self._orders:
            self._orders[order_id].status = OrderStatus.CANCELLED
            print(f"Alpaca order cancelled: {order_id}")
            return True
        return False
    
    def flatten_all(self) -> List[Order]:
        """Flatten all positions via Alpaca API."""
        # Mock flattening - in practice would get positions from Alpaca
        orders = []
        # Would iterate through actual positions here
        print("Alpaca: Flattening all positions")
        return orders
    
    def get_price(self, symbol: str) -> float:
        """Get current price from Alpaca."""
        # Mock price - in practice would call Alpaca market data API
        alpaca_symbol = self.normalize_symbol(symbol)
        # Would call self.client.get_latest_quote(alpaca_symbol) or similar
        print(f"Alpaca: Getting price for {alpaca_symbol}")
        return 100.0  # Mock price
    
    def get_spread(self, symbol: str) -> Tuple[float, float]:
        """Get current bid/ask spread from Alpaca."""
        # Mock spread - in practice would call Alpaca market data API
        price = self.get_price(symbol)
        spread = price * 0.001  # Mock 10bp spread
        return price - spread/2, price + spread/2
    
    def normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol for Alpaca format."""
        # Remove common suffixes and standardize
        symbol = symbol.upper().replace("USD", "").replace("USDT", "")
        
        # Handle crypto symbols
        if "BTC" in symbol:
            return "BTCUSD"
        elif "ETH" in symbol:
            return "ETHUSD"
        
        return symbol


class IBKRAdapter(BrokerAdapter):
    """Interactive Brokers adapter implementation (stub)."""
    
    def __init__(self, client_id: int = 1, host: str = "127.0.0.1", port: int = 7497):
        """Initialize IBKR adapter."""
        self.client_id = client_id
        self.host = host
        self.port = port
        self._orders = {}
        
        # In practice, would initialize IB API client
        self.client = None  # Mock client
        
    def place_order(self, symbol: str, side: str, quantity: float, order_type: OrderType,
                   limit_price: Optional[float] = None, stop_price: Optional[float] = None,
                   time_in_force: str = "GTC") -> Order:
        """Place order via Interactive Brokers API."""
        # Normalize symbol for IBKR
        ib_symbol, ib_exchange = self.normalize_symbol_and_exchange(symbol)
        
        # Mock order placement - in practice would use IB API
        order_id = f"ib_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        order = Order(
            order_id=order_id,
            symbol=ib_symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price
        )
        
        self._orders[order_id] = order
        
        # Log the order details
        print(f"IBKR order placed: {order_id} {side} {quantity} {ib_symbol}@{ib_exchange} @ {limit_price or 'market'}")
        
        return order
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order via IBKR API."""
        # Mock cancellation
        if order_id in self._orders:
            self._orders[order_id].status = OrderStatus.CANCELLED
            print(f"IBKR order cancelled: {order_id}")
            return True
        return False
    
    def flatten_all(self) -> List[Order]:
        """Flatten all positions via IBKR API."""
        # Mock flattening
        orders = []
        print("IBKR: Flattening all positions")
        return orders
    
    def get_price(self, symbol: str) -> float:
        """Get current price from IBKR."""
        # Mock price
        ib_symbol, ib_exchange = self.normalize_symbol_and_exchange(symbol)
        print(f"IBKR: Getting price for {ib_symbol}@{ib_exchange}")
        return 100.0  # Mock price
    
    def get_spread(self, symbol: str) -> Tuple[float, float]:
        """Get current bid/ask spread from IBKR."""
        # Mock spread
        price = self.get_price(symbol)
        spread = price * 0.0005  # Mock 5bp spread
        return price - spread/2, price + spread/2
    
    def normalize_symbol_and_exchange(self, symbol: str) -> Tuple[str, str]:
        """Normalize symbol and determine exchange for IBKR format."""
        symbol = symbol.upper()
        
        # Default to SMART routing
        exchange = "SMART"
        
        # Handle crypto
        if "BTC" in symbol or "ETH" in symbol:
            exchange = "PAXOS"  # or other crypto exchange
            if "BTCUSD" in symbol:
                return "BTC", exchange
            elif "ETHUSD" in symbol:
                return "ETH", exchange
        
        # Handle equities
        if symbol in ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]:
            exchange = "NASDAQ"  # or NYSE as appropriate
        
        return symbol, exchange
        security = self.algorithm.Securities[qc_symbol]
        return float(security.Price)
    
    def get_spread(self, symbol: str) -> Tuple[float, float]:
        """Get bid/ask spread from QuantConnect."""
        qc_symbol = self.algorithm.Symbol(symbol)
        security = self.algorithm.Securities[qc_symbol]
        
        bid = float(security.BidPrice) if hasattr(security, 'BidPrice') else float(security.Price)
        ask = float(security.AskPrice) if hasattr(security, 'AskPrice') else float(security.Price)
        
        return (bid, ask)
