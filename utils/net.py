"""
Centralized network utilities with retry logic and rate limiting.

This module provides consistent retry behavior for all network calls,
with awareness of rate limits (429), server errors (5xx), and exponential backoff.
"""

import time
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, cast

import requests
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

T = TypeVar('T')


class NetworkError(Exception):
    """Base exception for network-related errors."""
    pass


class RateLimitError(NetworkError):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, message: str, retry_after: Optional[int] = None):
        super().__init__(message)
        self.retry_after = retry_after


class ServerError(NetworkError):
    """Raised when server returns 5xx error."""
    pass


def create_retry_session(
    max_retries: int = 3,
    backoff_factor: float = 1.0,
    status_forcelist: tuple[int, ...] = (429, 500, 502, 503, 504),
    allowed_methods: tuple[str, ...] = ("HEAD", "GET", "OPTIONS", "POST", "PUT", "PATCH", "DELETE"),
) -> requests.Session:
    """
    Create a requests session with retry logic configured.
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Backoff factor for exponential backoff (delay = backoff_factor * (2 ** retry_number))
        status_forcelist: HTTP status codes to retry on
        allowed_methods: HTTP methods to retry
        
    Returns:
        Configured requests session
        
    Example:
        >>> session = create_retry_session(max_retries=3, backoff_factor=1.0)
        >>> response = session.get("https://api.example.com/data")
    """
    session = requests.Session()
    
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=allowed_methods,
        raise_on_status=False,  # We'll handle status codes ourselves
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential: bool = True,
    on_429: bool = True,
    on_5xx: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to retry a function with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exponential: Use exponential backoff
        on_429: Retry on 429 (rate limit)
        on_5xx: Retry on 5xx (server errors)
        
    Returns:
        Decorated function
        
    Example:
        >>> @retry_with_backoff(max_attempts=3, base_delay=1.0)
        ... def fetch_data():
        ...     response = requests.get("https://api.example.com/data")
        ...     response.raise_for_status()
        ...     return response.json()
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[Exception] = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                    
                except requests.exceptions.HTTPError as e:
                    status_code = e.response.status_code if e.response is not None else None
                    
                    # Check if we should retry this status code
                    should_retry = False
                    if on_429 and status_code == 429:
                        should_retry = True
                        # Check for Retry-After header
                        retry_after = None
                        if e.response is not None:
                            retry_after = e.response.headers.get('Retry-After')
                        
                        if retry_after:
                            try:
                                delay = float(retry_after)
                            except ValueError:
                                # Retry-After might be a date
                                delay = base_delay
                        else:
                            delay = base_delay
                            
                    elif on_5xx and status_code and 500 <= status_code < 600:
                        should_retry = True
                        delay = base_delay
                    
                    if not should_retry or attempt == max_attempts - 1:
                        raise
                    
                    last_exception = e
                    
                except (requests.exceptions.ConnectionError, 
                        requests.exceptions.Timeout,
                        requests.exceptions.RequestException) as e:
                    if attempt == max_attempts - 1:
                        raise
                    
                    last_exception = e
                    delay = base_delay
                
                # Calculate delay with exponential backoff
                if exponential:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                else:
                    delay = base_delay
                
                logger.warning(
                    f"Attempt {attempt + 1}/{max_attempts} failed: {last_exception}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
            
            # If we get here, all attempts failed
            if last_exception:
                raise last_exception
            
            # This should never happen, but satisfy type checker
            raise NetworkError("All retry attempts exhausted")
        
        return wrapper
    return decorator


def rate_limited(calls_per_second: float) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to rate limit function calls.
    
    Args:
        calls_per_second: Maximum number of calls per second
        
    Returns:
        Decorated function
        
    Example:
        >>> @rate_limited(calls_per_second=5.0)
        ... def fetch_ticker(symbol: str):
        ...     return requests.get(f"https://api.example.com/ticker/{symbol}").json()
    """
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]  # Use list for mutability
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            elapsed = time.time() - last_called[0]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
        
        return wrapper
    return decorator


def check_response_ok(response: requests.Response, context: str = "") -> None:
    """
    Check if response is OK and raise appropriate exceptions.
    
    Args:
        response: HTTP response object
        context: Context string for error messages
        
    Raises:
        RateLimitError: If rate limit exceeded (429)
        ServerError: If server error (5xx)
        requests.HTTPError: For other HTTP errors
    """
    if response.status_code == 429:
        retry_after = response.headers.get('Retry-After')
        retry_after_int = int(retry_after) if retry_after else None
        raise RateLimitError(
            f"Rate limit exceeded{' for ' + context if context else ''}. "
            f"Retry after: {retry_after_int}s" if retry_after_int else "Rate limit exceeded",
            retry_after=retry_after_int
        )
    
    if 500 <= response.status_code < 600:
        raise ServerError(
            f"Server error ({response.status_code}){' for ' + context if context else ''}: "
            f"{response.text[:200]}"
        )
    
    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        # Add context to error message
        if context:
            e.args = (f"{context}: {e.args[0]}",) + e.args[1:]
        raise


def get_default_headers(api_key: Optional[str] = None) -> dict[str, str]:
    """
    Get default headers for HTTP requests.
    
    Args:
        api_key: Optional API key to include
        
    Returns:
        Dictionary of headers
    """
    headers = {
        'User-Agent': 'quant-bot/0.1.0',
        'Accept': 'application/json',
    }
    
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    
    return headers


class NetworkClient:
    """
    Base class for network clients with built-in retry and rate limiting.
    
    Example:
        >>> class MyAPIClient(NetworkClient):
        ...     def __init__(self):
        ...         super().__init__(
        ...             base_url="https://api.example.com",
        ...             rate_limit=5.0,
        ...             max_retries=3
        ...         )
        ...
        ...     def get_data(self, symbol: str):
        ...         return self.get(f"/ticker/{symbol}")
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        rate_limit: Optional[float] = None,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        timeout: int = 30,
    ):
        """
        Initialize network client.
        
        Args:
            base_url: Base URL for API
            api_key: Optional API key
            rate_limit: Optional rate limit in calls per second
            max_retries: Maximum number of retries
            backoff_factor: Backoff factor for exponential backoff
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.timeout = timeout
        
        # Create session with retry logic
        self.session = create_retry_session(
            max_retries=max_retries,
            backoff_factor=backoff_factor
        )
        
        # Track last request time for rate limiting
        self.last_request_time = 0.0
    
    def _rate_limit(self) -> None:
        """Apply rate limiting if configured."""
        if self.rate_limit:
            min_interval = 1.0 / self.rate_limit
            elapsed = time.time() - self.last_request_time
            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any
    ) -> requests.Response:
        """
        Make HTTP request with rate limiting and error handling.
        
        Args:
            method: HTTP method
            endpoint: API endpoint (relative to base_url)
            **kwargs: Additional arguments for requests
            
        Returns:
            HTTP response
        """
        self._rate_limit()
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        # Set default headers
        headers = get_default_headers(self.api_key)
        if 'headers' in kwargs:
            headers.update(kwargs['headers'])
        kwargs['headers'] = headers
        
        # Set timeout if not provided
        if 'timeout' not in kwargs:
            kwargs['timeout'] = self.timeout
        
        logger.debug(f"{method} {url}")
        
        response = self.session.request(method, url, **kwargs)
        self.last_request_time = time.time()
        
        check_response_ok(response, context=f"{method} {endpoint}")
        
        return response
    
    def get(self, endpoint: str, **kwargs: Any) -> Any:
        """GET request."""
        response = self._make_request('GET', endpoint, **kwargs)
        return response.json() if response.content else None
    
    def post(self, endpoint: str, **kwargs: Any) -> Any:
        """POST request."""
        response = self._make_request('POST', endpoint, **kwargs)
        return response.json() if response.content else None
    
    def put(self, endpoint: str, **kwargs: Any) -> Any:
        """PUT request."""
        response = self._make_request('PUT', endpoint, **kwargs)
        return response.json() if response.content else None
    
    def delete(self, endpoint: str, **kwargs: Any) -> Any:
        """DELETE request."""
        response = self._make_request('DELETE', endpoint, **kwargs)
        return response.json() if response.content else None
