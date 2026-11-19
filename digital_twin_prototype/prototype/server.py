"""Local browser interface for saved-model experiments and raw-input trials."""

from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import argparse
import json
import mimetypes
import uuid
import pandas as pd
from threadpoolctl import threadpool_limits
from engine import InputSession
from features import ROOT, OUT, W12, required, FIELDS, raw_from_row

import os

C22 = Path(
    os.environ.get(
        "THESIS_CH6_ROOT", OUT.parent / "Thesis results C 22-10-2026 corrected 2026-11-02"
    )
)
SESSIONS = {}
FUTURE = json.loads((W12 / "protocol/future_calibration_splits.json").read_text())


def catalogue():
    tasks = {}
    for f in FUTURE:
        if f["repeat"] == 0:
            tasks.setdefault(f["task"], []).append(
                {"alloy": f["alloy"], "paired": f["primary_budget_comparison"]}
            )
    return {
        "tasks": tasks,
        "fields": {k: {"label": v[1], "unit": v[2]} for k, v in FIELDS.items()},
        "scope": "Frozen seed-0 excluded-alloy models. Each output has a separate dataset. Manual trials do not enter the result tables.",
    }


def info(session_id):
    s = SESSIONS[session_id]
    testid = s.replay.context["fixed_test_ids"][0]
    record = s.replay.records[testid]
    return {
        "session_id": session_id,
        "task": s.task,
        "alloy": s.alloy,
        "state": s.state(),
        "required": required(s.task),
        "sample": raw_from_row(s.task, record["raw_row_id"]),
        "sample_id": testid,
        "max_demo_measurements": len(s.selection["calibration20"] or s.selection["calibration9"]),
        "history": s.history,
        "model_scope": "Excluded alloy: " + s.alloy + "; model seed 0",
    }


def dispatch(route, body):
    if route == "/api/create":
        session = InputSession(body["task"], body["alloy"])
        sid = uuid.uuid4().hex
        SESSIONS[sid] = session
        return info(sid)
    sid = body["session_id"]
    if sid not in SESSIONS:
        raise ValueError("Session expired; select the model again")
    s = SESSIONS[sid]
    if route == "/api/query":
        return s.query(body["query_id"], body["raw"], body.get("role", "prospective"))
    if route == "/api/reveal":
        event = s.reveal_next()
        return {"event": event, **info(sid)}
    if route == "/api/observe":
        s.observe(body["query_id"], body["value"], body["unit"])
        return info(sid)
    if route in {"/api/save", "/api/reload"}:
        path = OUT / "saved_sessions" / f"{sid}.json"
        if route == "/api/save":
            s.save(path)
        else:
            SESSIONS[sid] = InputSession.load(path)
        return info(sid) | {"saved_file": path.name}
    raise ValueError("Unknown action")


class Handler(BaseHTTPRequestHandler):
    def send(self, status, payload, mime="application/json"):
        content = (
            json.dumps(payload, allow_nan=False).encode() if mime == "application/json" else payload
        )
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        route = urlparse(self.path).path
        if route == "/api/catalogue":
            return self.send(200, catalogue())
        if route == "/api/results":
            # Chapter 6 results: geometry from the corrected main design, defect class from the fixed-test design.
            g = pd.read_csv(C22 / "results/summary.csv")
            g = g[
                (g.cohort == "primary")
                & (g.clipped.astype(str) == "True")
                & g.method.isin(["source_split", "target_split", "ml_auto_jk"])
            ]
            c = pd.read_csv(ROOT / "results/summary.csv")
            c = c[(c.task == "classification") & (c.cohort == "primary_paired")]
            return self.send(
                200,
                {
                    "geometry": json.loads(g.to_json(orient="records")),
                    "classification": json.loads(c.to_json(orient="records")),
                },
            )
        if route == "/":
            return self.send(
                200, (OUT / "prototype/index.html").read_bytes(), "text/html; charset=utf-8"
            )
        if route.startswith("/figures/"):
            target = (ROOT / route.lstrip("/")).resolve()
            allowed = (ROOT / "figures").resolve()
            if (
                target.parent == allowed
                and target.suffix in {".png", ".svg", ".pdf"}
                and target.is_file()
            ):
                return self.send(200, target.read_bytes(), mimetypes.guess_type(target.name)[0])
        self.send(404, {"error": "Not found"})

    def do_POST(self):
        try:
            origin = self.headers.get("Origin")
            if origin and origin != "http://" + self.headers.get("Host", ""):
                raise ValueError("Cross-origin requests are not supported")
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 1024 * 1024:
                raise ValueError("Invalid request size")
            body = json.loads(self.rfile.read(size))
            with threadpool_limits(limits=1):
                output = dispatch(urlparse(self.path).path, body)
            self.send(200, output)
        except (ValueError, KeyError, TypeError, FileNotFoundError) as error:
            self.send(400, {"error": str(error)})

    def log_message(self, format, *args):
        print(format % args, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    print(f"Research prototype: http://127.0.0.1:{args.port}", flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
