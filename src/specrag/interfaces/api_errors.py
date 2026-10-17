"""Stable HTTP error envelope and exception mappings for the API boundary."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from specrag.models import ApiErrorResponse


class ApiHTTPException(HTTPException):
    """HTTP error carrying a stable client-facing error code."""

    def __init__(
        self,
        status_code: int,
        detail: str,
        *,
        code: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.error_code = code


def error(status_code: int, detail: str, *, code: str) -> ApiHTTPException:
    return ApiHTTPException(status_code, detail, code=code)


def unauthenticated() -> HTTPException:
    return ApiHTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="authentication required",
        code="authentication_required",
        headers={"WWW-Authenticate": "Bearer"},
    )


def forbidden() -> HTTPException:
    return error(status.HTTP_403_FORBIDDEN, "access denied", code="access_denied")


def run_not_found() -> HTTPException:
    return error(status.HTTP_404_NOT_FOUND, "run not found", code="run_not_found")


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiHTTPException)
    async def api_error_handler(
        request: Request, exc: ApiHTTPException,
    ) -> JSONResponse:
        del request
        payload = ApiErrorResponse(code=exc.error_code, detail=exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content=payload.model_dump(mode="json"),
            headers=exc.headers,
        )

    @app.exception_handler(HTTPException)
    async def generic_http_error_handler(
        request: Request, exc: HTTPException,
    ) -> JSONResponse:
        del request
        code = {
            401: "authentication_required",
            403: "access_denied",
            404: "not_found",
            405: "method_not_allowed",
            409: "conflict",
            422: "validation_error",
        }.get(exc.status_code, "http_error")
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        payload = ApiErrorResponse(code=code, detail=detail)
        return JSONResponse(
            status_code=exc.status_code,
            content=payload.model_dump(mode="json"),
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(
        request: Request, exc: RequestValidationError,
    ) -> JSONResponse:
        del request
        payload = ApiErrorResponse(
            code="validation_error",
            detail=tuple(jsonable_encoder(exc.errors())),
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content=payload.model_dump(mode="json"),
        )


__all__ = [
    "ApiHTTPException",
    "error",
    "forbidden",
    "install_error_handlers",
    "run_not_found",
    "unauthenticated",
]
