"""
Discord alerts system for trading bot notifications.
Sends order, fill, and risk event notifications to Discord webhook.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from urllib.request import Request, urlopen
from urllib.error import URLError

logger = logging.getLogger(__name__)


class DiscordAlerter:
    """Discord webhook notification system."""
    
    def __init__(self):
        """Initialize Discord alerter."""
        self.webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
        self.enabled = bool(self.webhook_url)
        
        if not self.enabled:
            logger.info("Discord alerts disabled: DISCORD_WEBHOOK_URL not set")
        else:
            logger.info("Discord alerts enabled")
    
    def _send_webhook(self, message: str, embed: Optional[Dict[str, Any]] = None) -> bool:
        """
        Send message to Discord webhook.
        
        Args:
            message: Text message to send
            embed: Optional rich embed data
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False
        
        payload = {
            "content": message,
            "username": "Quant Bot",
            "avatar_url": "https://cdn.discordapp.com/emojis/789395846419046420.png"
        }
        
        if embed:
            payload["embeds"] = [embed]
        
        try:
            data = json.dumps(payload).encode('utf-8')
            request = Request(
                self.webhook_url,
                data=data,
                headers={'Content-Type': 'application/json'}
            )
            
            with urlopen(request, timeout=10) as response:
                if response.status == 204:
                    return True
                else:
                    logger.warning(f"Discord webhook returned status {response.status}")
                    return False
                    
        except URLError as e:
            logger.error(f"Failed to send Discord alert: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending Discord alert: {e}")
            return False
    
    def send_order_alert(self, 
                        symbol: str,
                        side: str,
                        quantity: float,
                        price: float,
                        order_type: str = "MARKET") -> bool:
        """
        Send order placement alert.
        
        Args:
            symbol: Trading symbol
            side: BUY or SELL
            quantity: Order quantity
            price: Order price
            order_type: Order type
            
        Returns:
            True if sent successfully
        """
        emoji = "🟢" if side.upper() == "BUY" else "🔴"
        message = f"{emoji} **ORDER PLACED**"
        
        embed = {
            "title": f"{side.upper()} {symbol}",
            "color": 0x00ff00 if side.upper() == "BUY" else 0xff0000,
            "fields": [
                {"name": "Symbol", "value": symbol, "inline": True},
                {"name": "Side", "value": side.upper(), "inline": True},
                {"name": "Quantity", "value": f"{quantity:.4f}", "inline": True},
                {"name": "Price", "value": f"${price:.4f}", "inline": True},
                {"name": "Type", "value": order_type, "inline": True},
                {"name": "Value", "value": f"${abs(quantity * price):.2f}", "inline": True}
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return self._send_webhook(message, embed)
    
    def send_fill_alert(self,
                       symbol: str,
                       side: str,
                       quantity: float,
                       fill_price: float,
                       strategy: str = "Unknown") -> bool:
        """
        Send order fill alert.
        
        Args:
            symbol: Trading symbol
            side: BUY or SELL
            quantity: Filled quantity
            fill_price: Fill price
            strategy: Strategy that generated the trade
            
        Returns:
            True if sent successfully
        """
        emoji = "✅" if side.upper() == "BUY" else "❌"
        message = f"{emoji} **ORDER FILLED**"
        
        embed = {
            "title": f"FILLED: {side.upper()} {symbol}",
            "color": 0x0099ff,
            "fields": [
                {"name": "Symbol", "value": symbol, "inline": True},
                {"name": "Side", "value": side.upper(), "inline": True},
                {"name": "Quantity", "value": f"{quantity:.4f}", "inline": True},
                {"name": "Fill Price", "value": f"${fill_price:.4f}", "inline": True},
                {"name": "Strategy", "value": strategy, "inline": True},
                {"name": "Fill Value", "value": f"${abs(quantity * fill_price):.2f}", "inline": True}
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return self._send_webhook(message, embed)
    
    def send_risk_alert(self,
                       event_type: str,
                       message: str,
                       severity: str = "WARNING",
                       details: Optional[Dict[str, Any]] = None) -> bool:
        """
        Send risk management alert.
        
        Args:
            event_type: Type of risk event (KILL_SWITCH, POSITION_LIMIT, etc.)
            message: Risk event message
            severity: WARNING, ERROR, CRITICAL
            details: Additional event details
            
        Returns:
            True if sent successfully
        """
        severity_colors = {
            "INFO": 0x00ff00,
            "WARNING": 0xffff00,
            "ERROR": 0xff8000,
            "CRITICAL": 0xff0000
        }
        
        severity_emojis = {
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "ERROR": "🚨",
            "CRITICAL": "🔥"
        }
        
        emoji = severity_emojis.get(severity, "⚠️")
        alert_message = f"{emoji} **RISK ALERT: {event_type}**"
        
        embed = {
            "title": f"Risk Event: {event_type}",
            "description": message,
            "color": severity_colors.get(severity, 0xffff00),
            "fields": [
                {"name": "Severity", "value": severity, "inline": True},
                {"name": "Event Type", "value": event_type, "inline": True}
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if details:
            for key, value in details.items():
                embed["fields"].append({
                    "name": key.replace("_", " ").title(),
                    "value": str(value),
                    "inline": True
                })
        
        return self._send_webhook(alert_message, embed)
    
    def send_performance_alert(self,
                             portfolio_value: float,
                             daily_pnl: float,
                             daily_pnl_pct: float,
                             positions_count: int,
                             drawdown: float) -> bool:
        """
        Send daily performance summary.
        
        Args:
            portfolio_value: Current portfolio value
            daily_pnl: Daily P&L in dollars
            daily_pnl_pct: Daily P&L percentage
            positions_count: Number of open positions
            drawdown: Current drawdown percentage
            
        Returns:
            True if sent successfully
        """
        pnl_emoji = "📈" if daily_pnl >= 0 else "📉"
        message = f"{pnl_emoji} **DAILY PERFORMANCE**"
        
        embed = {
            "title": "Daily Performance Summary",
            "color": 0x00ff00 if daily_pnl >= 0 else 0xff0000,
            "fields": [
                {"name": "Portfolio Value", "value": f"${portfolio_value:,.2f}", "inline": True},
                {"name": "Daily P&L", "value": f"${daily_pnl:+,.2f}", "inline": True},
                {"name": "Daily P&L %", "value": f"{daily_pnl_pct:+.2f}%", "inline": True},
                {"name": "Open Positions", "value": str(positions_count), "inline": True},
                {"name": "Drawdown", "value": f"{drawdown:.2f}%", "inline": True},
                {"name": "Status", "value": "🟢 Active" if drawdown < 15 else "🟡 Caution", "inline": True}
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return self._send_webhook(message, embed)
    
    def send_startup_alert(self, version: str = "1.0.0") -> bool:
        """
        Send bot startup notification.
        
        Args:
            version: Bot version
            
        Returns:
            True if sent successfully
        """
        message = "🚀 **QUANT BOT STARTED**"
        
        embed = {
            "title": "Quant Bot Startup",
            "description": "Trading bot has started successfully",
            "color": 0x00ff00,
            "fields": [
                {"name": "Version", "value": version, "inline": True},
                {"name": "Status", "value": "🟢 Online", "inline": True},
                {"name": "Environment", "value": "Production", "inline": True}
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return self._send_webhook(message, embed)


# Global alerter instance
alerter = DiscordAlerter()


def send_order_alert(symbol: str, side: str, quantity: float, price: float, order_type: str = "MARKET") -> bool:
    """Send order placement alert."""
    return alerter.send_order_alert(symbol, side, quantity, price, order_type)


def send_fill_alert(symbol: str, side: str, quantity: float, fill_price: float, strategy: str = "Unknown") -> bool:
    """Send order fill alert."""
    return alerter.send_fill_alert(symbol, side, quantity, fill_price, strategy)


def send_risk_alert(event_type: str, message: str, severity: str = "WARNING", details: Optional[Dict[str, Any]] = None) -> bool:
    """Send risk management alert."""
    return alerter.send_risk_alert(event_type, message, severity, details)


def send_performance_alert(portfolio_value: float, daily_pnl: float, daily_pnl_pct: float, positions_count: int, drawdown: float) -> bool:
    """Send daily performance summary."""
    return alerter.send_performance_alert(portfolio_value, daily_pnl, daily_pnl_pct, positions_count, drawdown)


def send_startup_alert(version: str = "1.0.0") -> bool:
    """Send bot startup notification."""
    return alerter.send_startup_alert(version)