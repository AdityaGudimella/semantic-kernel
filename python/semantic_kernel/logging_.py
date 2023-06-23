"""Logging API for Semantic Kernel."""

import logging
import typing as t

import pydantic as pdt

LogLevels = t.Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class LoggerSettings(pdt.BaseSettings):
    """Settings to configure logging for SemanticKernel."""

    log_level: LogLevels = pdt.Field(
        default="DEBUG", env="SK_LOG_LEVEL", description="Logging level to use."
    )

    def get_logger(self, name: str) -> logging.Logger:
        """Returns a logger with the specified name.

        Important:
        ----------
            This method should be used instead of `logging.getLogger` to ensure that
            the logger is configured with the correct settings.

        Arguments:
        ----------
        name : str
            Name of the logger to return.

        Returns:
        --------
        logging.Logger
            Fully configured Logger with the specified name
        """
        logger = logging.getLogger(name)
        logger.setLevel(self.log_level)
        logging.captureWarnings(True)
        return logger


class SKLogger(pdt.BaseModel):
    """API for logging."""

    name: str = pdt.Field(
        default="semantic_kernel",
        description="Name of the logger.",
    )
    settings: LoggerSettings = pdt.Field(
        default_factory=LoggerSettings,
        description="Settings to configure logging for SemanticKernel.",
    )
    _logger: logging.Logger = pdt.PrivateAttr(None)

    class Config:
        """Pydantic configuration."""

        allow_mutation = False

    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            self._logger = self.settings.get_logger(self.name)
        return self._logger

    def info(self, *args: t.Any, **kwargs: t.Any) -> None:
        return self.logger.info(*args, **kwargs)

    def debug(self, *args: t.Any, **kwargs: t.Any) -> None:
        return self.logger.debug(*args, **kwargs)

    def warning(self, *args: t.Any, **kwargs: t.Any) -> None:
        return self.logger.warning(*args, **kwargs)

    def error(self, *args: t.Any, **kwargs: t.Any) -> None:
        return self.logger.error(*args, **kwargs)

    def critical(self, *args: t.Any, **kwargs: t.Any) -> None:
        return self.logger.critical(*args, **kwargs)


class NullLogger(SKLogger):
    """A logger that does nothing."""

    def info(self, _: str) -> None:
        pass

    def debug(self, _: str) -> None:
        pass

    def warning(self, _: str) -> None:
        pass

    def error(self, _: str) -> None:
        pass

    def critical(self, _: str) -> None:
        pass
