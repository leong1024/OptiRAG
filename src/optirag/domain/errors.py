class OptiRAGError(Exception):
    """Base error for the package."""


class ConfigError(OptiRAGError):
    """Invalid configuration."""


class RetriableAPIError(OptiRAGError):
    """Transient API failure (retryable)."""
