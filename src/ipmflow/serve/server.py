"""HTTP server for dashboard and LLM proxy."""

import json
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

from ipmflow.inference.predict import predict as local_predict
from ipmflow.llm.providers import GROQ_API_KEY, GROQ_MODEL, OPENAI_API_KEY, OPENAI_MODEL, call_groq, call_openai
from ipmflow.paths import ARTIFACTS_DIR, DASHBOARD_DIR, PROJECT_ROOT
from ipmflow.serve.benchmark_api import load_benchmark_payload

DEFAULT_PORT = 8765
PORT = int(os.environ.get("PORT", str(DEFAULT_PORT)))

PROVIDERS = {"groq": call_groq, "openai": call_openai}


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PROJECT_ROOT), **kwargs)

    def do_GET(self):
        if self.path in ("/", "/dashboard", "/dashboard/"):
            self.path = "/dashboard/index.html"
        elif self.path == "/api/providers":
            self._json_response(
                200,
                {
                    "groq": {"configured": bool(GROQ_API_KEY), "model": GROQ_MODEL},
                    "openai": {"configured": bool(OPENAI_API_KEY), "model": OPENAI_MODEL},
                },
            )
            return
        elif self.path == "/api/benchmark":
            self._json_response(200, load_benchmark_payload())
            return
        super().do_GET()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        text = body.get("text", "").strip()

        if self.path == "/api/classify":
            provider = body.get("provider", "groq").lower()
            rules_context = body.get("rules_context")
            horizon_context = body.get("horizon_context")
            if not text:
                self._json_response(400, {"error": "Missing pitch text"})
                return
            if provider not in PROVIDERS:
                self._json_response(400, {"error": f"Unknown provider: {provider}"})
                return
            try:
                content = PROVIDERS[provider](text, rules_context, horizon_context)
                self._json_response(200, {"content": content, "provider": provider})
            except Exception as exc:
                self._json_response(500, {"error": str(exc), "provider": provider})
            return

        if self.path == "/api/predict":
            model = body.get("model", "c").lower()
            if not text:
                self._json_response(400, {"error": "Missing pitch text"})
                return
            if model not in ("c", "d"):
                self._json_response(400, {"error": "model must be c or d"})
                return
            try:
                result = local_predict(text, model=model)
                self._json_response(200, result)
            except Exception as exc:
                self._json_response(500, {"error": str(exc), "model": model})
            return

        self.send_error(404, "Not found")

    def _json_response(self, status: int, payload: dict):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {format % args}")


def run():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    if not GROQ_API_KEY:
        print("Warning: GROQ_API_KEY is not set.")
    if not OPENAI_API_KEY:
        print("Warning: OPENAI_API_KEY is not set.")
    print(f"Dashboard: http://localhost:{PORT}/dashboard/index.html")
    print(f"Benchmark API: http://localhost:{PORT}/api/benchmark")
    try:
        HTTPServer(("localhost", PORT), DashboardHandler).serve_forever()
    except PermissionError as exc:
        raise SystemExit(
            f"Cannot bind to port {PORT} ({exc}). "
            f"Port may be in use or reserved — try: $env:PORT=8766; python scripts/serve.py"
        ) from exc
    except OSError as exc:
        if getattr(exc, "winerror", None) == 10048 or exc.errno in (98, 48):  # address already in use
            raise SystemExit(
                f"Port {PORT} is already in use. "
                f"Stop the other process or run: $env:PORT=8766; python scripts/serve.py"
            ) from exc
        raise
