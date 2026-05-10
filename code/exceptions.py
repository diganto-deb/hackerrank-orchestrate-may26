from __future__ import annotations


class PipelineError(Exception):
    def __init__(self, stage: str, cause: Exception) -> None:
        self.stage = stage
        self.cause = cause
        super().__init__(f"[{stage}] {type(cause).__name__}: {cause}")


class ProviderError(PipelineError):
    pass


class JSONParseError(PipelineError):
    pass


class SchemaValidationError(PipelineError):
    pass


class GuardrailError(PipelineError):
    pass


class StageTimeoutError(PipelineError):
    pass