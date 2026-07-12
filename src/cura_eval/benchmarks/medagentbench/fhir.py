"""FHIR HTTP client, ported from MedAgentBench ``src/server/tasks/medagentbench/utils.py``.

Read-only: the harness never POSTs to the server (POSTs are record-only and graded by
payload structure). The sync ``requests`` call is wrapped by callers via
``asyncio.to_thread`` so it never blocks the event loop.
"""

from __future__ import annotations

import requests


def send_get_request(url: str, params=None, headers=None) -> dict:
    """Send a GET; return {"status_code", "data"} on success or {"error"} on failure."""
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        return {
            "status_code": response.status_code,
            "data": response.json()
            if response.headers.get("Content-Type") == "application/json"
            else response.text,
        }
    except Exception as e:
        return {"error": str(e)}


def verify_fhir_server(fhir_api_base: str) -> bool:
    """True iff the FHIR server's metadata endpoint returns 200."""
    res = send_get_request(f"{fhir_api_base}metadata")
    return res.get("status_code", 0) == 200
