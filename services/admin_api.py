"""
FastAPI Admin API for quant trading bot control and monitoring.
Provides REST endpoints for control, monitoring, and data export.
"""

import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

# Import our modules
from algos.core.runtime_state import get_runtime_state
from storage.trades import get_trade_storage

# Prometheus metrics
REQUEST_COUNT = Counter("admin_api_requests_total", "Total requests", ["method", "endpoint"])
REQUEST_DURATION = Histogram("admin_api_request_duration_seconds", "Request duration")
ACTIVE_STRATEGIES = Gauge("trading_active_strategies", "Number of active strategies")
EQUITY_VALUE = Gauge("trading_equity_value", "Current equity value")
DRAWDOWN_PCT = Gauge("trading_drawdown_percent", "Current drawdown percentage")


# Request/Response models
class RiskUpdateRequest(BaseModel):
    key: str = Field(..., description="Risk parameter name")
    value: float = Field(..., description="New parameter value")


class StrategyToggleRequest(BaseModel):
    name: str = Field(..., description="Strategy name")
    enabled: bool = Field(..., description="Enable/disable strategy")


class KillSwitchRequest(BaseModel):
    reason: str | None = Field(default="Manual kill switch", description="Reason for kill switch")


class TradeFilters(BaseModel):
    symbol: str | None = None
    start: str | None = None
    end: str | None = None
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=100, ge=1, le=1000)


# Security
security = HTTPBearer(auto_error=False)


def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> bool:
    """Verify admin token for mutating endpoints."""
    if not credentials:
        return False

    admin_token = os.getenv("ADMIN_TOKEN")
    if not admin_token:
        logger.warning("ADMIN_TOKEN not configured, rejecting request")
        return False

    return credentials.credentials == admin_token


def require_auth(credentials: HTTPAuthorizationCredentials = Security(security)) -> bool:
    """Dependency for endpoints requiring authentication."""
    if not verify_token(credentials):
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return True


# FastAPI app
app = FastAPI(
    title="Quant Bot Admin API",
    description="Admin API for quant trading bot control and monitoring",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Static files and templates
static_dir = Path(__file__).parent / "static"
templates_dir = Path(__file__).parent / "templates"

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

if templates_dir.exists():
    templates = Jinja2Templates(directory=str(templates_dir))


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware to collect request metrics."""
    start_time = datetime.utcnow()

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path
    ).inc()

    with REQUEST_DURATION.time():
        response = await call_next(request)

    return response


# Health and status endpoints
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": 0  # TODO: Track actual uptime
    }


@app.get("/status")
async def get_status():
    """Get runtime status summary."""
    try:
        runtime_state = get_runtime_state()
        trade_storage = get_trade_storage()

        # Get trade statistics
        trade_stats = trade_storage.get_trade_stats()

        # Get equity curve (last 30 days)
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=30)
        equity_data = trade_storage.get_equity_curve(
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat()
        )

        # Calculate current metrics
        current_equity = trade_stats.get("latest_equity", 0.0)
        current_drawdown = trade_stats.get("latest_drawdown", 0.0)
        active_strategies = sum(1 for enabled in runtime_state.strategy_enabled.values() if enabled)

        # Update Prometheus metrics
        EQUITY_VALUE.set(current_equity)
        DRAWDOWN_PCT.set(current_drawdown * 100)
        ACTIVE_STRATEGIES.set(active_strategies)

        return {
            "trading_paused": runtime_state.trading_paused,
            "kill_switch_active": runtime_state.kill_switch_active,
            "last_kill_reason": runtime_state.last_kill_reason,
            "current_regime": runtime_state.last_regime,
            "active_strategies": active_strategies,
            "total_strategies": len(runtime_state.strategy_enabled),
            "strategy_status": runtime_state.strategy_enabled,
            "equity": current_equity,
            "drawdown_pct": current_drawdown * 100,
            "total_trades": trade_stats.get("total_trades", 0),
            "win_rate": trade_stats.get("win_rate", 0.0),
            "total_pnl": trade_stats.get("total_pnl", 0.0),
            "last_update": runtime_state.to_dict().get("last_update"),
        }

    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get status")


@app.get("/runtime_state.json")
async def get_runtime_state_json():
    """Get runtime state for QC polling."""
    try:
        runtime_state = get_runtime_state()
        return runtime_state.to_dict()
    except Exception as e:
        logger.error(f"Failed to get runtime state: {e}")
        raise HTTPException(status_code=500, detail="Failed to get runtime state")


# Control endpoints (require authentication)
@app.post("/risk")
async def update_risk(request: RiskUpdateRequest, _: bool = Depends(require_auth)):
    """Update risk parameter."""
    try:
        runtime_state = get_runtime_state()

        patch = {
            "risk_params": {
                request.key: request.value
            }
        }

        success = runtime_state.apply_patch(patch)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to apply risk update")

        return {
            "success": True,
            "message": f"Updated {request.key} to {request.value}",
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update risk: {e}")
        raise HTTPException(status_code=500, detail="Failed to update risk parameter")


@app.post("/toggle_strategy")
async def toggle_strategy(request: StrategyToggleRequest, _: bool = Depends(require_auth)):
    """Toggle strategy on/off."""
    try:
        runtime_state = get_runtime_state()

        patch = {
            "strategy_enabled": {
                request.name: request.enabled
            }
        }

        success = runtime_state.apply_patch(patch)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to toggle strategy")

        status = "enabled" if request.enabled else "disabled"
        return {
            "success": True,
            "message": f"Strategy {request.name} {status}",
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to toggle strategy: {e}")
        raise HTTPException(status_code=500, detail="Failed to toggle strategy")


@app.post("/kill_switch")
async def activate_kill_switch(request: KillSwitchRequest, _: bool = Depends(require_auth)):
    """Activate kill switch."""
    try:
        runtime_state = get_runtime_state()
        runtime_state.mark_kill(request.reason)

        return {
            "success": True,
            "message": f"Kill switch activated: {request.reason}",
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to activate kill switch: {e}")
        raise HTTPException(status_code=500, detail="Failed to activate kill switch")


@app.post("/resume")
async def resume_trading(_: bool = Depends(require_auth)):
    """Resume trading (deactivate kill switch)."""
    try:
        runtime_state = get_runtime_state()
        runtime_state.resume()

        return {
            "success": True,
            "message": "Trading resumed - kill switch deactivated",
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to resume trading: {e}")
        raise HTTPException(status_code=500, detail="Failed to resume trading")


@app.post("/retrain_now")
async def retrain_now(_: bool = Depends(require_auth)):
    """Trigger model retraining."""
    try:
        # Check if we're in a local environment (not QC cloud)
        if os.path.exists("scripts/train_classifier.py"):
            # Spawn training in background
            subprocess.Popen([
                "python", "scripts/train_classifier.py",
                "--config", "config.yaml",
                "--output-dir", "models_new"
            ])

            return {
                "success": True,
                "message": "Model retraining started in background",
                "timestamp": datetime.utcnow().isoformat()
            }
        return {
            "success": False,
            "message": "Use retrain workflow - not available in this environment",
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to start retraining: {e}")
        raise HTTPException(status_code=500, detail="Failed to start retraining")


# Data endpoints
@app.get("/trades")
async def get_trades(
    symbol: str | None = None,
    start: str | None = None,
    end: str | None = None,
    page: int = 1,
    page_size: int = 100
):
    """Get trades with optional filters."""
    try:
        trade_storage = get_trade_storage()

        # Calculate offset for pagination
        offset = (page - 1) * page_size

        trades = trade_storage.list_trades(
            symbol=symbol,
            start_time=start,
            end_time=end,
            limit=page_size,
            offset=offset
        )

        return {
            "trades": trades,
            "page": page,
            "page_size": page_size,
            "has_more": len(trades) == page_size
        }

    except Exception as e:
        logger.error(f"Failed to get trades: {e}")
        raise HTTPException(status_code=500, detail="Failed to get trades")


@app.get("/trades/export.csv")
async def export_trades_csv(
    symbol: str | None = None,
    start: str | None = None,
    end: str | None = None,
    _: bool = Depends(require_auth)
):
    """Export trades to CSV file."""
    try:
        trade_storage = get_trade_storage()

        # Generate filename with timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"trades_{timestamp}.csv"
        output_path = f"exports/{filename}"

        # Export trades
        success = trade_storage.export_trades_csv(
            output_path,
            symbol=symbol,
            start_time=start,
            end_time=end
        )

        if not success:
            raise HTTPException(status_code=404, detail="No trades to export")

        return FileResponse(
            path=output_path,
            filename=filename,
            media_type="text/csv"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export trades CSV: {e}")
        raise HTTPException(status_code=500, detail="Failed to export trades")


@app.get("/trades/export.parquet")
async def export_trades_parquet(
    symbol: str | None = None,
    start: str | None = None,
    end: str | None = None,
    _: bool = Depends(require_auth)
):
    """Export trades to Parquet file."""
    try:
        trade_storage = get_trade_storage()

        # Generate filename with timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"trades_{timestamp}.parquet"
        output_path = f"exports/{filename}"

        # Export trades
        success = trade_storage.export_trades_parquet(
            output_path,
            symbol=symbol,
            start_time=start,
            end_time=end
        )

        if not success:
            raise HTTPException(status_code=404, detail="No trades to export")

        return FileResponse(
            path=output_path,
            filename=filename,
            media_type="application/octet-stream"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export trades Parquet: {e}")
        raise HTTPException(status_code=500, detail="Failed to export trades")


@app.get("/equity_curve.json")
async def get_equity_curve(
    start: str | None = None,
    end: str | None = None
):
    """Get equity curve data."""
    try:
        trade_storage = get_trade_storage()

        equity_data = trade_storage.get_equity_curve(
            start_time=start,
            end_time=end
        )

        return {
            "data": equity_data,
            "count": len(equity_data)
        }

    except Exception as e:
        logger.error(f"Failed to get equity curve: {e}")
        raise HTTPException(status_code=500, detail="Failed to get equity curve")


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    try:
        return generate_latest()
    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate metrics")


# Dashboard endpoints (if templates exist)
@app.get("/")
async def dashboard(request: Request):
    """Main dashboard page."""
    if not templates_dir.exists():
        return {"message": "Dashboard templates not available"}

    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/trades.html")
async def trades_page(request: Request):
    """Trades blotter page."""
    if not templates_dir.exists():
        return {"message": "Dashboard templates not available"}

    return templates.TemplateResponse("trades.html", {"request": request})


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={"error": "Not found", "path": request.url.path}
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn

    # Get configuration from environment
    host = os.getenv("ADMIN_API_HOST", "0.0.0.0")
    port = int(os.getenv("ADMIN_API_PORT", "8080"))

    logger.info(f"Starting Admin API on {host}:{port}")

    uvicorn.run(
        "services.admin_api:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
