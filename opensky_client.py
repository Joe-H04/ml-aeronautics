from __future__ import annotations

import os
import time
from threading import Lock
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv


load_dotenv()


class OpenSkyError(Exception):
    pass


class OpenSkyHTTPError(OpenSkyError):
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class OpenSkyClient:
    """OpenSky REST client with optional OAuth token caching."""

    BASE_URL = "https://opensky-network.org/api"
    TOKEN_URL = (
        "https://auth.opensky-network.org/auth/realms/opensky-network/"
        "protocol/openid-connect/token"
    )
    TOKEN_REFRESH_MARGIN_SECONDS = 30

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        timeout_seconds: int = 20,
    ):
        self.client_id = client_id or os.getenv("OPENSKY_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("OPENSKY_CLIENT_SECRET")
        self.timeout_seconds = timeout_seconds
        self._session = requests.Session()
        self._token: Optional[str] = None
        self._expires_at: float = 0.0
        self._lock = Lock()

    @property
    def auth_mode(self) -> str:
        return "oauth" if self.is_configured else "anonymous"

    @property
    def is_configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def list_current_flights(
        self,
        *,
        lamin: Optional[float] = None,
        lamax: Optional[float] = None,
        lomin: Optional[float] = None,
        lomax: Optional[float] = None,
        limit: int = 25,
        query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if None not in (lamin, lamax, lomin, lomax):
            params.update({
                "lamin": lamin,
                "lamax": lamax,
                "lomin": lomin,
                "lomax": lomax,
            })

        payload = self._request_json("GET", "/states/all", params=params)
        states = payload.get("states") or []
        query_lc = (query or "").strip().lower()
        flights: List[Dict[str, Any]] = []

        for row in states:
            if len(row) < 10:
                continue

            icao24 = (row[0] or "").strip().lower()
            callsign = (row[1] or "").strip()
            origin_country = (row[2] or "").strip()
            time_position = row[3]
            last_contact = row[4]
            longitude = row[5]
            latitude = row[6]
            altitude_m = row[7]
            on_ground = bool(row[8])
            velocity_mps = row[9]

            if not icao24 or latitude is None or longitude is None or last_contact is None:
                continue

            haystack = " ".join([icao24, callsign.lower(), origin_country.lower()])
            if query_lc and query_lc not in haystack:
                continue

            flights.append({
                "icao24": icao24,
                "callsign": callsign,
                "origin_country": origin_country,
                "time_position": int(time_position) if time_position is not None else None,
                "last_contact": int(last_contact),
                "longitude": float(longitude),
                "latitude": float(latitude),
                "altitude_m": float(altitude_m) if altitude_m is not None else None,
                "velocity_mps": float(velocity_mps) if velocity_mps is not None else None,
                "on_ground": on_ground,
            })

        flights.sort(
            key=lambda flight: (
                -flight["last_contact"],
                flight["callsign"] or flight["icao24"],
            )
        )
        return flights[:max(1, min(limit, 100))]

    def get_track(self, icao24: str, *, time_seconds: int = 0) -> Dict[str, Any]:
        params = {"icao24": icao24.strip().lower(), "time": int(time_seconds)}
        last_error: Optional[OpenSkyHTTPError] = None

        for path in ("/tracks/all", "/tracks"):
            try:
                payload = self._request_json("GET", path, params=params)
                if payload:
                    return payload
            except OpenSkyHTTPError as exc:
                last_error = exc
                if exc.status_code not in {404}:
                    break

        if last_error is not None:
            raise last_error
        raise OpenSkyHTTPError("OpenSky returned an empty track response.")

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        response = self._request(method, path, params=params)
        try:
            return response.json()
        except ValueError as exc:
            raise OpenSkyHTTPError(
                f"OpenSky returned invalid JSON for {path}.",
                status_code=response.status_code,
            ) from exc

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
    ) -> requests.Response:
        headers = self._headers()
        response = self._send(method, path, params=params, headers=headers)

        if response.status_code == 401 and self.is_configured:
            headers = self._headers(force_refresh=True)
            response = self._send(method, path, params=params, headers=headers)

        if response.status_code == 404:
            raise OpenSkyHTTPError(
                "No OpenSky data found for that request.",
                status_code=404,
            )

        if not response.ok:
            raise OpenSkyHTTPError(
                self._build_error_message(response, path),
                status_code=response.status_code,
            )

        return response

    def _send(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]],
        headers: Dict[str, str],
    ) -> requests.Response:
        try:
            return self._session.request(
                method=method,
                url=f"{self.BASE_URL}{path}",
                params=params,
                headers=headers,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise OpenSkyHTTPError(f"OpenSky request failed: {exc}") from exc

    def _headers(self, force_refresh: bool = False) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        token = self._get_token(force_refresh=force_refresh)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _get_token(self, force_refresh: bool = False) -> Optional[str]:
        if not self.is_configured:
            return None

        now = time.time()
        with self._lock:
            if (
                not force_refresh
                and self._token
                and now < (self._expires_at - self.TOKEN_REFRESH_MARGIN_SECONDS)
            ):
                return self._token

            try:
                response = self._session.post(
                    self.TOKEN_URL,
                    data={
                        "grant_type": "client_credentials",
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                    },
                    timeout=self.timeout_seconds,
                )
            except requests.RequestException as exc:
                raise OpenSkyHTTPError(f"OpenSky token request failed: {exc}") from exc

            if not response.ok:
                raise OpenSkyHTTPError(
                    self._build_error_message(response, "token"),
                    status_code=response.status_code,
                )

            try:
                payload = response.json()
            except ValueError as exc:
                raise OpenSkyHTTPError(
                    "OpenSky token endpoint returned invalid JSON.",
                    status_code=response.status_code,
                ) from exc

            token = payload.get("access_token")
            if not token:
                raise OpenSkyHTTPError("OpenSky token response did not include access_token.")

            expires_in = int(payload.get("expires_in", 1800))
            self._token = token
            self._expires_at = now + expires_in
            return self._token

    @staticmethod
    def _build_error_message(response: requests.Response, path: str) -> str:
        detail = response.text.strip()
        try:
            payload = response.json()
            if isinstance(payload, dict):
                detail = payload.get("message") or payload.get("detail") or detail
        except ValueError:
            pass

        detail = detail or response.reason or "unknown error"
        return f"OpenSky request to {path} failed with HTTP {response.status_code}: {detail}"
