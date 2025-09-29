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
        security = self.algorithm.Securities[qc_symbol]
        return float(security.Price)
    
    def get_spread(self, symbol: str) -> Tuple[float, float]:
        """Get bid/ask spread from QuantConnect."""
        qc_symbol = self.algorithm.Symbol(symbol)
        security = self.algorithm.Securities[qc_symbol]
        
        bid = float(security.BidPrice) if hasattr(security, 'BidPrice') else float(security.Price)
        ask = float(security.AskPrice) if hasattr(security, 'AskPrice') else float(security.Price)
        
        return (bid, ask)
