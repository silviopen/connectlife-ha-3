"""Tests for the local ConnectLife API wrapper."""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
from pathlib import Path
import sys
import types
from typing import Any
import unittest
from unittest.mock import patch


CONNECTLIFE_MODULE = types.ModuleType("connectlife")
CONNECTLIFE_APPLIANCE_MODULE = types.ModuleType("connectlife.appliance")


class StubConnectLifeAppliance:
    """Minimal stand-in for the packaged ConnectLife appliance model."""

    def __init__(self, api: Any, data: dict[str, Any]) -> None:
        self.api = api
        self.data = data
        self.device_id = data.get("deviceId")
        self.status_list = data.get("statusList")


CONNECTLIFE_APPLIANCE_MODULE.ConnectLifeAppliance = StubConnectLifeAppliance
sys.modules.setdefault("connectlife", CONNECTLIFE_MODULE)
sys.modules["connectlife.appliance"] = CONNECTLIFE_APPLIANCE_MODULE


API_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "custom_components" / "connectlife" / "api.py"
)
API_SPEC = importlib.util.spec_from_file_location("connectlife_local_api", API_MODULE_PATH)
if API_SPEC is None or API_SPEC.loader is None:
    raise RuntimeError(f"Unable to load {API_MODULE_PATH}")
API_MODULE = importlib.util.module_from_spec(API_SPEC)
API_SPEC.loader.exec_module(API_MODULE)

ConnectLifeApi = API_MODULE.ConnectLifeApi


class FakeResponse:
    """Minimal aiohttp response stand-in."""

    def __init__(self, status: int, payload: Any, headers: dict[str, str] | None = None) -> None:
        self.status = status
        self.headers = headers or {}
        self._payload = payload

    async def __aenter__(self) -> "FakeResponse":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def text(self) -> str:
        if isinstance(self._payload, str):
            return self._payload
        return json.dumps(self._payload)

    async def json(self) -> Any:
        if isinstance(self._payload, str):
            return json.loads(self._payload)
        return self._payload


class FakeSession:
    """Minimal aiohttp ClientSession stand-in."""

    def __init__(self, requests: list[tuple[str, str, FakeResponse | BaseException]]) -> None:
        self._requests = requests

    async def __aenter__(self) -> "FakeSession":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    def get(self, url: str, **kwargs) -> FakeResponse:
        return self._next("GET", url)

    def post(self, url: str, **kwargs) -> FakeResponse:
        return self._next("POST", url)

    def _next(self, method: str, url: str) -> FakeResponse:
        if not self._requests:
            raise AssertionError(f"Unexpected {method} request to {url}")

        expected_method, expected_url, outcome = self._requests.pop(0)
        if expected_method != method or expected_url != url:
            raise AssertionError(
                f"Expected {expected_method} {expected_url}, got {method} {url}"
            )
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeClientSessionFactory:
    """Factory returning fake sessions that share one scripted request queue."""

    def __init__(self, requests: list[tuple[str, str, FakeResponse | BaseException]]) -> None:
        self._requests = requests

    def __call__(self, *args, **kwargs) -> FakeSession:
        return FakeSession(self._requests)


class ConnectLifeApiTests(unittest.IsolatedAsyncioTestCase):
    """Tests for auth and request recovery paths."""

    async def test_refresh_failure_falls_back_to_full_login(self) -> None:
        api = ConnectLifeApi("user@example.com", "secret")
        api._access_token = "expired-token"
        api._refresh_token = "expired-refresh"
        api._expires = dt.datetime.now() - dt.timedelta(seconds=1)

        requests = [
            (
                "POST",
                api.oauth2_token,
                FakeResponse(500, {"error": "temporary failure"}),
            ),
            (
                "POST",
                api.login_url,
                FakeResponse(
                    200,
                    {"UID": "uid-1", "sessionInfo": {"cookieValue": "login-token"}},
                ),
            ),
            ("POST", api.jwt_url, FakeResponse(200, {"id_token": "jwt-token"})),
            ("POST", api.oauth2_authorize, FakeResponse(200, {"code": "auth-code"})),
            (
                "POST",
                api.oauth2_token,
                FakeResponse(
                    200,
                    {
                        "access_token": "new-access-token",
                        "expires_in": 3600,
                        "refresh_token": "new-refresh-token",
                        "refreshTokenExpiredTime": 4_102_444_800_000,
                    },
                ),
            ),
        ]

        with patch.object(
            API_MODULE.aiohttp,
            "ClientSession",
            new=FakeClientSessionFactory(requests),
        ):
            await api._fetch_access_token()

        self.assertEqual(api._access_token, "new-access-token")
        self.assertEqual(api._refresh_token, "new-refresh-token")
        self.assertGreater(api._expires, dt.datetime.now())
        self.assertFalse(requests)

    async def test_refresh_transport_error_falls_back_to_full_login(self) -> None:
        api = ConnectLifeApi("user@example.com", "secret")
        api._access_token = "expired-token"
        api._refresh_token = "expired-refresh"
        api._expires = dt.datetime.now() - dt.timedelta(seconds=1)

        requests = [
            (
                "POST",
                api.oauth2_token,
                API_MODULE.aiohttp.ClientConnectionError("connection reset"),
            ),
            (
                "POST",
                api.login_url,
                FakeResponse(
                    200,
                    {"UID": "uid-1", "sessionInfo": {"cookieValue": "login-token"}},
                ),
            ),
            ("POST", api.jwt_url, FakeResponse(200, {"id_token": "jwt-token"})),
            ("POST", api.oauth2_authorize, FakeResponse(200, {"code": "auth-code"})),
            (
                "POST",
                api.oauth2_token,
                FakeResponse(
                    200,
                    {
                        "access_token": "new-access-token",
                        "expires_in": 3600,
                        "refresh_token": "new-refresh-token",
                    },
                ),
            ),
        ]

        with patch.object(
            API_MODULE.aiohttp,
            "ClientSession",
            new=FakeClientSessionFactory(requests),
        ):
            await api._fetch_access_token()

        self.assertEqual(api._access_token, "new-access-token")
        self.assertEqual(api._refresh_token, "new-refresh-token")
        self.assertFalse(requests)

    async def test_initial_login_retries_after_transient_auth_error(self) -> None:
        api = ConnectLifeApi("user@example.com", "secret")

        requests = [
            (
                "POST",
                api.login_url,
                FakeResponse(500, {"error": "upstream login error"}),
            ),
            (
                "POST",
                api.login_url,
                FakeResponse(
                    200,
                    {"UID": "uid-1", "sessionInfo": {"cookieValue": "login-token"}},
                ),
            ),
            ("POST", api.jwt_url, FakeResponse(200, {"id_token": "jwt-token"})),
            ("POST", api.oauth2_authorize, FakeResponse(200, {"code": "auth-code"})),
            (
                "POST",
                api.oauth2_token,
                FakeResponse(
                    200,
                    {
                        "access_token": "new-access-token",
                        "expires_in": 3600,
                        "refresh_token": "new-refresh-token",
                    },
                ),
            ),
        ]

        with patch.object(
            API_MODULE.aiohttp,
            "ClientSession",
            new=FakeClientSessionFactory(requests),
        ):
            with patch.object(API_MODULE.asyncio, "sleep", return_value=None):
                await api.login()

        self.assertEqual(api._access_token, "new-access-token")
        self.assertFalse(requests)

    async def test_appliances_request_reauths_after_transient_server_error(self) -> None:
        api = ConnectLifeApi("user@example.com", "secret")
        api._allow_gateway_fallback = False
        api._access_token = "cached-access-token"
        api._expires = dt.datetime.now() + dt.timedelta(minutes=5)

        requests = [
            (
                "GET",
                api.appliances_url,
                FakeResponse(500, {"error": "backend unavailable"}),
            ),
            (
                "POST",
                api.login_url,
                FakeResponse(
                    200,
                    {"UID": "uid-1", "sessionInfo": {"cookieValue": "login-token"}},
                ),
            ),
            ("POST", api.jwt_url, FakeResponse(200, {"id_token": "jwt-token"})),
            ("POST", api.oauth2_authorize, FakeResponse(200, {"code": "auth-code"})),
            (
                "POST",
                api.oauth2_token,
                FakeResponse(
                    200,
                    {
                        "access_token": "replacement-access-token",
                        "expires_in": 3600,
                        "refresh_token": "replacement-refresh-token",
                    },
                ),
            ),
            (
                "GET",
                api.appliances_url,
                FakeResponse(200, [{"deviceId": "device-1"}]),
            ),
        ]

        with patch.object(
            API_MODULE.aiohttp,
            "ClientSession",
            new=FakeClientSessionFactory(requests),
        ):
            result = await api.get_appliances_json()

        self.assertEqual(result, [{"deviceId": "device-1"}])
        self.assertEqual(api._access_token, "replacement-access-token")
        self.assertFalse(requests)

    async def test_update_appliance_uses_gateway_before_bapi(self) -> None:
        api = ConnectLifeApi("user@example.com", "secret")
        api._access_token = "cached-access-token"
        api._expires = dt.datetime.now() + dt.timedelta(minutes=5)

        requests = [
            (
                "POST",
                API_MODULE.GATEWAY_UPDATE_URL,
                FakeResponse(
                    200,
                    {"response": {"resultCode": 1, "errorCode": 555, "errorDesc": "gateway unavailable"}},
                ),
            ),
            (
                "POST",
                api.appliances_url,
                FakeResponse(200, {"ok": True}),
            ),
        ]

        with patch.object(
            API_MODULE.aiohttp,
            "ClientSession",
            new=FakeClientSessionFactory(requests),
        ):
            await api.update_appliance("puid-1", {"t_temp": "22"})

        self.assertFalse(requests)

    async def test_update_appliance_retries_gateway_randstr_failure_before_bapi(self) -> None:
        api = ConnectLifeApi("user@example.com", "secret")
        api._access_token = "cached-access-token"
        api._expires = dt.datetime.now() + dt.timedelta(minutes=5)

        requests = [
            (
                "POST",
                API_MODULE.GATEWAY_UPDATE_URL,
                FakeResponse(
                    200,
                    {"response": {"resultCode": 1, "errorCode": 101005, "errorDesc": "randStr check fail."}},
                ),
            ),
            (
                "POST",
                API_MODULE.GATEWAY_UPDATE_URL,
                FakeResponse(200, {"response": {"resultCode": 0}}),
            ),
        ]

        with patch.object(
            API_MODULE.aiohttp,
            "ClientSession",
            new=FakeClientSessionFactory(requests),
        ):
            await api.update_appliance("puid-1", {"t_temp": "22"})

        self.assertFalse(requests)

    async def test_update_appliance_retries_bapi_after_transient_status(self) -> None:
        api = ConnectLifeApi("user@example.com", "secret")
        api._access_token = "cached-access-token"
        api._expires = dt.datetime.now() + dt.timedelta(minutes=5)

        requests = [
            (
                "POST",
                API_MODULE.GATEWAY_UPDATE_URL,
                FakeResponse(
                    200,
                    {"response": {"resultCode": 1, "errorCode": 101005, "errorDesc": "randStr check fail."}},
                ),
            ),
            (
                "POST",
                API_MODULE.GATEWAY_UPDATE_URL,
                FakeResponse(
                    200,
                    {"response": {"resultCode": 1, "errorCode": 555, "errorDesc": "gateway unavailable"}},
                ),
            ),
            (
                "POST",
                api.appliances_url,
                FakeResponse(502, "<title>bapi.ovh | 502: Bad gateway</title>"),
            ),
            (
                "POST",
                api.login_url,
                FakeResponse(
                    200,
                    {"UID": "uid-1", "sessionInfo": {"cookieValue": "login-token"}},
                ),
            ),
            ("POST", api.jwt_url, FakeResponse(200, {"id_token": "jwt-token"})),
            ("POST", api.oauth2_authorize, FakeResponse(200, {"code": "auth-code"})),
            (
                "POST",
                api.oauth2_token,
                FakeResponse(
                    200,
                    {
                        "access_token": "replacement-access-token",
                        "expires_in": 3600,
                        "refresh_token": "replacement-refresh-token",
                    },
                ),
            ),
            (
                "POST",
                api.appliances_url,
                FakeResponse(200, {"ok": True}),
            ),
        ]

        with patch.object(
            API_MODULE.aiohttp,
            "ClientSession",
            new=FakeClientSessionFactory(requests),
        ):
            await api.update_appliance("puid-1", {"t_temp": "22"})

        self.assertEqual(api._access_token, "replacement-access-token")
        self.assertFalse(requests)

    async def test_update_appliance_retries_bapi_after_transport_error(self) -> None:
        api = ConnectLifeApi("user@example.com", "secret")
        api._access_token = "cached-access-token"
        api._expires = dt.datetime.now() + dt.timedelta(minutes=5)

        requests = [
            (
                "POST",
                API_MODULE.GATEWAY_UPDATE_URL,
                FakeResponse(
                    200,
                    {"response": {"resultCode": 1, "errorCode": 555, "errorDesc": "gateway unavailable"}},
                ),
            ),
            (
                "POST",
                api.appliances_url,
                API_MODULE.aiohttp.ClientConnectionError("connection reset"),
            ),
            (
                "POST",
                api.login_url,
                FakeResponse(
                    200,
                    {"UID": "uid-1", "sessionInfo": {"cookieValue": "login-token"}},
                ),
            ),
            ("POST", api.jwt_url, FakeResponse(200, {"id_token": "jwt-token"})),
            ("POST", api.oauth2_authorize, FakeResponse(200, {"code": "auth-code"})),
            (
                "POST",
                api.oauth2_token,
                FakeResponse(
                    200,
                    {
                        "access_token": "replacement-access-token",
                        "expires_in": 3600,
                        "refresh_token": "replacement-refresh-token",
                    },
                ),
            ),
            (
                "POST",
                api.appliances_url,
                FakeResponse(200, {"ok": True}),
            ),
        ]

        with patch.object(
            API_MODULE.aiohttp,
            "ClientSession",
            new=FakeClientSessionFactory(requests),
        ):
            await api.update_appliance("puid-1", {"t_temp": "22"})

        self.assertEqual(api._access_token, "replacement-access-token")
        self.assertFalse(requests)

    async def test_get_appliances_json_raises_lifeconnect_error_for_html_body(self) -> None:
        api = ConnectLifeApi("user@example.com", "secret")
        api._access_token = "cached-access-token"
        api._expires = dt.datetime.now() + dt.timedelta(minutes=5)

        requests = [
            (
                "GET",
                api.appliances_url,
                FakeResponse(200, "<title>bapi.ovh | 502: Bad gateway</title>"),
            ),
        ]

        with patch.object(
            API_MODULE.aiohttp,
            "ClientSession",
            new=FakeClientSessionFactory(requests),
        ):
            with self.assertRaises(API_MODULE.LifeConnectError) as context:
                await api.get_appliances_json()

        self.assertEqual(
            str(context.exception),
            f"Unexpected non-JSON response from {api.appliances_url}",
        )
        self.assertFalse(requests)

    async def test_get_appliances_preserves_cached_status_list_when_missing(self) -> None:
        api = ConnectLifeApi("user@example.com", "secret")
        api._access_token = "cached-access-token"
        api._expires = dt.datetime.now() + dt.timedelta(minutes=5)
        api.appliances = [
            StubConnectLifeAppliance(
                api,
                {
                    "deviceId": "device-1",
                    "statusList": {"t_temp": 22, "power": 1},
                },
            )
        ]

        requests = [
            (
                "GET",
                api.appliances_url,
                FakeResponse(200, [{"deviceId": "device-1", "nickName": "Office"}]),
            ),
        ]

        with patch.object(
            API_MODULE.aiohttp,
            "ClientSession",
            new=FakeClientSessionFactory(requests),
        ):
            appliances = await api.get_appliances()

        self.assertEqual(len(appliances), 1)
        self.assertEqual(appliances[0].status_list, {"t_temp": 22, "power": 1})
        self.assertEqual(appliances[0].data["statusList"], {"t_temp": 22, "power": 1})
        self.assertFalse(requests)

    async def test_get_appliances_drops_incomplete_payload_on_cold_start(self) -> None:
        api = ConnectLifeApi("user@example.com", "secret")
        api._access_token = "cached-access-token"
        api._expires = dt.datetime.now() + dt.timedelta(minutes=5)

        requests = [
            (
                "GET",
                api.appliances_url,
                FakeResponse(200, [{"deviceId": "device-1", "nickName": "Office"}]),
            ),
        ]

        with patch.object(
            API_MODULE.aiohttp,
            "ClientSession",
            new=FakeClientSessionFactory(requests),
        ):
            appliances = await api.get_appliances()

        self.assertEqual(appliances, [])
        self.assertFalse(requests)

    async def test_get_appliances_keeps_fresh_status_list_when_present(self) -> None:
        api = ConnectLifeApi("user@example.com", "secret")
        api._access_token = "cached-access-token"
        api._expires = dt.datetime.now() + dt.timedelta(minutes=5)
        api.appliances = [
            StubConnectLifeAppliance(
                api,
                {
                    "deviceId": "device-1",
                    "statusList": {"t_temp": 22, "power": 1},
                },
            )
        ]

        requests = [
            (
                "GET",
                api.appliances_url,
                FakeResponse(
                    200,
                    [{"deviceId": "device-1", "statusList": {"t_temp": 19, "power": 0}}],
                ),
            ),
        ]

        with patch.object(
            API_MODULE.aiohttp,
            "ClientSession",
            new=FakeClientSessionFactory(requests),
        ):
            appliances = await api.get_appliances()

        self.assertEqual(len(appliances), 1)
        self.assertEqual(appliances[0].status_list, {"t_temp": 19, "power": 0})
        self.assertFalse(requests)

    async def test_get_appliances_drops_new_incomplete_payload_after_merge(self) -> None:
        api = ConnectLifeApi("user@example.com", "secret")
        api._access_token = "cached-access-token"
        api._expires = dt.datetime.now() + dt.timedelta(minutes=5)
        api.appliances = [
            StubConnectLifeAppliance(
                api,
                {
                    "deviceId": "device-1",
                    "statusList": {"t_temp": 22, "power": 1},
                },
            )
        ]

        requests = [
            (
                "GET",
                api.appliances_url,
                FakeResponse(
                    200,
                    [
                        {"deviceId": "device-1", "nickName": "Office"},
                        {"deviceId": "device-2", "nickName": "Bedroom"},
                    ],
                ),
            ),
        ]

        with patch.object(
            API_MODULE.aiohttp,
            "ClientSession",
            new=FakeClientSessionFactory(requests),
        ):
            appliances = await api.get_appliances()

        self.assertEqual(len(appliances), 1)
        self.assertEqual(appliances[0].device_id, "device-1")
        self.assertEqual(appliances[0].status_list, {"t_temp": 22, "power": 1})
        self.assertFalse(requests)


if __name__ == "__main__":
    unittest.main()
