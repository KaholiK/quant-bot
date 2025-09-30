"""
QC polling module for fetching runtime state from Admin API.
Handles network errors, timeouts, and jitter for safe polling.
"""

import json
import random
import time
from typing import Any

import requests
from loguru import logger


def fetch_runtime_state(url: str,
                       token: str | None = None,
                       timeout: int = 3,
                       jitter: bool = True) -> dict[str, Any] | None:
    """
    Fetch runtime state from Admin API endpoint.
    
    Args:
        url: Full URL to runtime_state.json endpoint
        token: Optional bearer token for authentication
        timeout: Request timeout in seconds
        jitter: Add random jitter to reduce thundering herd
        
    Returns:
        Runtime state dictionary or None if failed
    """
    if jitter:
        # Add random jitter up to 10% of timeout to prevent thundering herd
        jitter_delay = random.uniform(0, timeout * 0.1)
        time.sleep(jitter_delay)

    try:
        headers = {"User-Agent": "QuantConnect-LEAN/1.0"}

        if token:
            headers["Authorization"] = f"Bearer {token}"

        response = requests.get(
            url,
            headers=headers,
            timeout=timeout,
            allow_redirects=False
        )

        if response.status_code == 200:
            try:
                runtime_state = response.json()
                logger.debug(f"Successfully fetched runtime state from {url}")
                return runtime_state

            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON response from {url}: {e}")
                return None

        elif response.status_code == 401:
            logger.warning(f"Authentication failed for {url} - check token")
            return None

        elif response.status_code == 404:
            logger.warning(f"Runtime state endpoint not found: {url}")
            return None

        else:
            logger.warning(f"HTTP {response.status_code} from {url}")
            return None

    except requests.exceptions.ConnectTimeout:
        logger.debug(f"Connection timeout to {url}")
        return None

    except requests.exceptions.ReadTimeout:
        logger.debug(f"Read timeout from {url}")
        return None

    except requests.exceptions.ConnectionError as e:
        logger.debug(f"Connection error to {url}: {e}")
        return None

    except requests.exceptions.RequestException as e:
        logger.warning(f"Request error to {url}: {e}")
        return None

    except Exception as e:
        logger.error(f"Unexpected error fetching runtime state: {e}")
        return None


def build_runtime_state_url(base_url: str, port: int = 8080) -> str:
    """
    Build runtime state URL from base configuration.
    
    Args:
        base_url: Base URL or IP address
        port: Port number
        
    Returns:
        Full URL to runtime_state.json endpoint
    """
    # Handle various URL formats
    if base_url.startswith("http://") or base_url.startswith("https://"):
        # Full URL provided
        if not base_url.endswith("/"):
            base_url += "/"
        return f"{base_url}runtime_state.json"
    # IP or hostname provided
    return f"http://{base_url}:{port}/runtime_state.json"


class RuntimeStatePoller:
    """
    Polling client for runtime state with error handling and retry logic.
    """

    def __init__(self,
                 base_url: str,
                 port: int = 8080,
                 token: str | None = None,
                 timeout: int = 3,
                 retry_attempts: int = 2):
        """
        Initialize poller.
        
        Args:
            base_url: Base URL or IP address of Admin API
            port: Port number
            token: Optional bearer token
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts on failure
        """
        self.url = build_runtime_state_url(base_url, port)
        self.token = token
        self.timeout = timeout
        self.retry_attempts = retry_attempts

        # Statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.last_successful_fetch = None
        self.consecutive_failures = 0

        logger.info(f"Runtime state poller initialized for {self.url}")

    def fetch(self, jitter: bool = True) -> dict[str, Any] | None:
        """
        Fetch runtime state with retry logic.
        
        Args:
            jitter: Add random jitter to requests
            
        Returns:
            Runtime state dictionary or None if failed
        """
        self.total_requests += 1

        for attempt in range(self.retry_attempts + 1):
            result = fetch_runtime_state(
                self.url,
                self.token,
                self.timeout,
                jitter and attempt == 0  # Only jitter on first attempt
            )

            if result is not None:
                self.successful_requests += 1
                self.last_successful_fetch = time.time()
                self.consecutive_failures = 0
                return result

            # Brief delay before retry (exponential backoff)
            if attempt < self.retry_attempts:
                delay = 0.5 * (2 ** attempt)  # 0.5s, 1s, 2s...
                time.sleep(delay)

        # All attempts failed
        self.consecutive_failures += 1

        if self.consecutive_failures % 10 == 1:  # Log every 10th failure
            logger.warning(f"Runtime state polling failed {self.consecutive_failures} consecutive times")

        return None

    def get_stats(self) -> dict[str, Any]:
        """Get polling statistics."""
        success_rate = self.successful_requests / max(self.total_requests, 1)

        return {
            "url": self.url,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "success_rate": success_rate,
            "consecutive_failures": self.consecutive_failures,
            "last_successful_fetch": self.last_successful_fetch,
        }

    def is_healthy(self, max_consecutive_failures: int = 5) -> bool:
        """
        Check if poller is healthy based on recent success rate.
        
        Args:
            max_consecutive_failures: Maximum consecutive failures before unhealthy
            
        Returns:
            True if poller is healthy
        """
        return self.consecutive_failures < max_consecutive_failures
