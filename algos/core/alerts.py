"""
Discord alerts system for trading bot notifications.
Sends order, fill, risk events, regime shifts, and model promotion notifications.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import threading
import queue
import time

logger = logging.getLogger(__name__)


class AlertManager:
    """Comprehensive alert management system with Discord integration."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize alert manager."""
        self.config = config or {}
        self.webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
        self.enabled = bool(self.webhook_url)
        
        # Alert categories and colors
        self.alert_colors = {
            'order': 0x3498db,      # Blue
            'fill': 0x2ecc71,       # Green  
            'risk': 0xe74c3c,       # Red
            'kill_switch': 0x8e44ad, # Purple
            'regime': 0xf39c12,     # Orange
            'model': 0x1abc9c,      # Teal
            'system': 0x95a5a6      # Gray
        }
        
        # Setup session with retry strategy
        if self.enabled:
            self.session = requests.Session()
            retry_strategy = Retry(
                total=2,
                backoff_factor=0.5,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)
        
        # Async alert queue
        self.alert_queue: queue.Queue = queue.Queue()
        self.alert_thread = None
        self.stop_thread = False
        
        # Start alert processing thread
        if self.enabled:
            self._start_alert_thread()
            logger.info("Discord alerts enabled with async processing")
        else:
            logger.info("Discord alerts disabled: DISCORD_WEBHOOK_URL not set")
    
    def _start_alert_thread(self):
        """Start background thread for processing alerts."""
        self.alert_thread = threading.Thread(target=self._alert_worker, daemon=True)
        self.alert_thread.start()
    
    def _alert_worker(self):
        """Background worker to process alerts."""
        while not self.stop_thread:
            try:
                # Get alert from queue (timeout to check stop condition)
                alert_data = self.alert_queue.get(timeout=1.0)
                
                # Send the alert
                self._send_webhook_sync(alert_data)
                
                # Mark task as done
                self.alert_queue.task_done()
                
                # Rate limit
                time.sleep(0.1)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in alert worker: {e}")
    
    def _send_webhook_sync(self, alert_data: Dict[str, Any]) -> bool:
        """Send webhook synchronously."""
        try:
            response = self.session.post(
                self.webhook_url,
                json=alert_data,
                timeout=5.0
            )
            response.raise_for_status()
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")
            return False
    
    def _send_alert(self, 
                   category: str,
                   title: str, 
                   description: str,
                   fields: Optional[List[Dict[str, str]]] = None,
                   footer: Optional[str] = None) -> None:
        """Queue an alert for sending."""
        if not self.enabled:
            return
        
        embed = {
            "title": title,
            "description": description,
            "color": self.alert_colors.get(category, 0x95a5a6),
            "timestamp": datetime.utcnow().isoformat(),
            "fields": fields or [],
            "footer": {"text": footer or f"Quant Bot • {category.title()}"}
        }
        
        payload = {
            "username": "Quant Bot",
            "embeds": [embed]
        }
        
        # Add to queue for async processing
        try:
            self.alert_queue.put_nowait(payload)
        except queue.Full:
            logger.warning("Alert queue full, dropping alert")
    
    def send_order_alert(self, 
                        symbol: str,
                        side: str,
                        quantity: float,
                        order_type: str,
                        price: Optional[float] = None,
                        strategy: Optional[str] = None,
                        regime: Optional[str] = None) -> None:
        """Send order placement alert."""
        price_str = f"${price:.2f}" if price else "market"
        title = f"🎯 Order Placed: {side} {symbol}"
        
        description = f"**{quantity:,.0f}** shares at **{price_str}** ({order_type})"
        
        fields = []
        if strategy:
            fields.append({"name": "Strategy", "value": strategy, "inline": True})
        if regime:
            fields.append({"name": "Regime", "value": regime, "inline": True})
        
        self._send_alert("order", title, description, fields)
    
    def send_fill_alert(self,
                       symbol: str,
                       side: str,
                       filled_quantity: float,
                       fill_price: float,
                       remaining_quantity: float = 0.0,
                       total_value: Optional[float] = None) -> None:
        """Send order fill alert."""
        fill_status = "Partial" if remaining_quantity > 0 else "Complete"
        title = f"✅ {fill_status} Fill: {side} {symbol}"
        
        description = f"**{filled_quantity:,.0f}** shares at **${fill_price:.2f}**"
        
        fields = []
        if total_value:
            fields.append({"name": "Total Value", "value": f"${total_value:,.2f}", "inline": True})
        if remaining_quantity > 0:
            fields.append({"name": "Remaining", "value": f"{remaining_quantity:,.0f} shares", "inline": True})
        
        self._send_alert("fill", title, description, fields)
    
    def send_risk_alert(self,
                       alert_type: str,
                       message: str,
                       current_values: Optional[Dict[str, float]] = None,
                       thresholds: Optional[Dict[str, float]] = None) -> None:
        """Send risk management alert."""
        title = f"⚠️ Risk Alert: {alert_type}"
        description = message
        
        fields = []
        if current_values:
            for key, value in current_values.items():
                if isinstance(value, float):
                    value_str = f"{value:.2%}" if 'pct' in key or 'ratio' in key else f"{value:.2f}"
                else:
                    value_str = str(value)
                fields.append({"name": key.replace('_', ' ').title(), "value": value_str, "inline": True})
        
        if thresholds:
            for key, value in thresholds.items():
                threshold_str = f"{value:.2%}" if 'pct' in key or 'ratio' in key else f"{value:.2f}"
                fields.append({"name": f"{key.replace('_', ' ').title()} Limit", "value": threshold_str, "inline": True})
        
        self._send_alert("risk", title, description, fields)
    
    def send_kill_switch_alert(self,
                              reason: str,
                              current_drawdown: float,
                              threshold: float,
                              portfolio_value: float) -> None:
        """Send kill switch activation alert."""
        title = "🛑 KILL SWITCH ACTIVATED"
        description = f"**{reason}**\n\nAll positions will be flattened and new entries blocked."
        
        fields = [
            {"name": "Current Drawdown", "value": f"{current_drawdown:.2%}", "inline": True},
            {"name": "Threshold", "value": f"{threshold:.2%}", "inline": True},
            {"name": "Portfolio Value", "value": f"${portfolio_value:,.2f}", "inline": True}
        ]
        
        self._send_alert("kill_switch", title, description, fields)
    
    def send_regime_shift_alert(self,
                               from_regime: str,
                               to_regime: str,
                               confidence: float,
                               impact: Optional[str] = None) -> None:
        """Send market regime shift alert."""
        title = f"🔄 Regime Shift: {from_regime.title()} → {to_regime.title()}"
        description = f"Market regime changed with **{confidence:.1%}** confidence"
        
        fields = [
            {"name": "From", "value": from_regime.title(), "inline": True},
            {"name": "To", "value": to_regime.title(), "inline": True},
            {"name": "Confidence", "value": f"{confidence:.1%}", "inline": True}
        ]
        
        if impact:
            fields.append({"name": "Impact", "value": impact, "inline": False})
        
        self._send_alert("regime", title, description, fields)
    
    def send_model_promotion_alert(self,
                                  model_type: str,
                                  action: str,  # "promoted" or "rejected"
                                  metrics: Dict[str, float],
                                  gates_passed: Dict[str, bool]) -> None:
        """Send model promotion/rejection alert."""
        status_emoji = "🚀" if action == "promoted" else "❌"
        title = f"{status_emoji} Model {action.title()}: {model_type}"
        
        if action == "promoted":
            description = "New model passed all promotion gates and is now active"
        else:
            description = "Model failed promotion gates and was rejected"
        
        fields = []
        for metric, value in metrics.items():
            passed = gates_passed.get(metric, True)
            emoji = "✅" if passed else "❌"
            
            if 'ratio' in metric or 'factor' in metric:
                value_str = f"{value:.2f}"
            elif 'pct' in metric or 'dd' in metric:
                value_str = f"{value:.2%}"
            else:
                value_str = f"{value:.3f}"
            
            fields.append({
                "name": f"{emoji} {metric.replace('_', ' ').title()}", 
                "value": value_str, 
                "inline": True
            })
        
        self._send_alert("model", title, description, fields)
    
    def send_system_alert(self,
                         event: str,
                         details: str,
                         severity: str = "info") -> None:
        """Send system event alert."""
        emoji_map = {
            "info": "ℹ️",
            "warning": "⚠️", 
            "error": "❌",
            "success": "✅"
        }
        
        emoji = emoji_map.get(severity, "ℹ️")
        title = f"{emoji} System {event.title()}"
        
        self._send_alert("system", title, details)
    
    def send_startup_alert(self, config_summary: Dict[str, Any]) -> None:
        """Send bot startup notification."""
        title = "🚀 Quant Bot Started"
        description = "Trading bot has been initialized and is ready for trading"
        
        fields = []
        for key, value in config_summary.items():
            if isinstance(value, (int, float)):
                if 'pct' in key or 'threshold' in key:
                    value_str = f"{value:.1%}"
                else:
                    value_str = str(value)
            else:
                value_str = str(value)
            
            fields.append({
                "name": key.replace('_', ' ').title(),
                "value": value_str,
                "inline": True
            })
        
        self._send_alert("system", title, description, fields)
    
    def close(self):
        """Close alert manager and cleanup resources."""
        if self.alert_thread and self.alert_thread.is_alive():
            # Stop the worker thread
            self.stop_thread = True
            
            # Wait for queue to be processed
            self.alert_queue.join()
            
            # Wait for thread to finish
            self.alert_thread.join(timeout=5.0)
            
        logger.info("Alert manager closed")


# Global alert manager instance
_alert_manager: Optional[AlertManager] = None


def get_alert_manager(config: Optional[Dict[str, Any]] = None) -> AlertManager:
    """Get or create global alert manager instance."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager(config)
    return _alert_manager


def send_startup_alert(config_summary: Dict[str, Any]) -> None:
    """Convenience function to send startup alert."""
    alert_manager = get_alert_manager()
    alert_manager.send_startup_alert(config_summary)


# Legacy compatibility functions
def send_order_fill_alert(*args, **kwargs):
    """Legacy function for backward compatibility."""
    alert_manager = get_alert_manager()
    alert_manager.send_fill_alert(*args, **kwargs)
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