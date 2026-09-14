#!/usr/bin/env python3
"""gcs_bridge_control.py -- HTTP sidecar to start/stop run_gcs_bridge.py from
a GCS web UI button ("AI 방어 모드" in /home/yune/gcs/server/web/index.html).

Mirrors /home/yune/gcs's own tests/simulation/mock/demo_relocate_service.py
design choice: the GCS server itself (server/api.py, under gcs's
transmit-safety audit) never gets a process-spawning endpoint -- starting a
process is a different category of thing from "may this vehicle be
commanded to a point". By the same logic in the other direction, this
sidecar lives here (in MobRobGPT, next to run_gcs_bridge.py) rather than in
the gcs repo: the process being managed (run_gcs_bridge.py, the RL autonomy
module) is owned by MobRobGPT, so its lifecycle control is too. gcs's only
involvement is a browser button that calls this sidecar's HTTP API -- gcs
never learns this file exists.

This sidecar does NOT touch gcs's command/gate.py, server/api.py, or any
MAVLink/goto transmission path. It only starts and stops an OS process. The
waypoints that process sends still go exclusively through the existing,
already-validated HTTP contract (POST /api/command/{id}/goto, source=rl) --
nothing about that path changes here.

Net-deploy stays exactly as run_gcs_bridge.py already implements it: signals
only ever go to --net-log (or the --net-ros2-namespace topic), never to gcs.
This sidecar does not add, remove, or touch that behavior in any way.

All mock-vs-real-hardware differences are just CLI flags at sidecar startup
(which --gcs-url, --ally-ids, --enemy-source, --bag, ...) -- switching
deployments never requires a code change here or in run_gcs_bridge.py.

Usage (mixed demo, matches tests/simulation/mock/run_mixed_demo_live.sh -- recommended
for this demo: --enemy-source gcs, not live. live subscribes to the raw ROS2 pose topic
bag_enemy_relay.py --source pose reprojects *through*, which is the bag's own
recording-local frame, not the GCS-datum-relative frame ally telemetry uses -- gcs reads
the already-reprojected position straight from GCS `/api/state` instead, exactly like the
ally link does, so it can't drift out of that frame. See commander/gcs_cnn_env.py's module
docstring for the coordinate-frame bug this was found to cause):
    python3 tools/gcs_bridge_control.py \\
        --gcs-url http://127.0.0.1:8091 --ally-ids usv1,usv2,usv3 \\
        --llm heuristic --enemy-source gcs \\
        --target-ids usv4,usv5,usv6,usv7,usv8 --port 8093
    (--span defaults to 35; the GUI's "AI 방어 모드" span/시각화 controls override it
    per POST /start call -- see "Per-start overrides" below)

Usage (raw ROS2 topic instead, only correct if that topic is already GCS-datum-relative):
    python3 tools/gcs_bridge_control.py \\
        --gcs-url http://127.0.0.1:8091 --ally-ids usv1,usv2,usv3 \\
        --span 8 --llm heuristic --enemy-source live --port 8093

Usage (bag replay instead of live topics):
    python3 tools/gcs_bridge_control.py \\
        --gcs-url http://127.0.0.1:8091 --ally-ids usv1,usv2,usv3 \\
        --span 8 --llm heuristic \\
        --bag /home/yune/Downloads/ros_data/S03-gcs --port 8093

Per-start overrides:
    Most run_gcs_bridge.py flags are fixed at sidecar startup (see the comment above
    the argparse block below for why). --span/--viz/--satellite/--spf/--pause-start are
    the exception: the GUI's "AI 방어 모드" button lets the operator pick span and
    toggle visualization per run, since those are exactly what changes run to run (map
    scale, whether a screen is available to show a plot on) rather than deployment
    config. POST /start with a JSON body overrides that call only, e.g.:
        curl -X POST http://127.0.0.1:8093/start \\
            -H 'Content-Type: application/json' \\
            -d '{"span": 35, "viz": true}'
    Unknown fields are rejected (400); omitted fields fall back to this process's own
    --span/--viz/--satellite/--spf/--pause-start (i.e. this sidecar's own CLI defaults).
"""
import argparse
import json
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MOBROBGPT_ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

_lock = threading.Lock()
_proc: subprocess.Popen | None = None
_log_path: Path | None = None
_started_at: float | None = None


def _waypoints_path(args) -> Path:
    """Fixed name (not timestamped like the run's own log file) so GET /waypoints
    always knows where to look regardless of which launch wrote it last."""
    return Path(args.log_dir) / "waypoints_latest.json"


def _build_argv(args) -> list[str]:
    argv = [
        # -u: stdout redirected to a log file is non-tty, so Python fully block-buffers
        # it by default -- /status's log_tail would then lag minutes behind (or show
        # nothing until the process exits) instead of reflecting live progress.
        PY, "-u", str(MOBROBGPT_ROOT / "run_gcs_bridge.py"),
        "--gcs-url", args.gcs_url,
        "--ally-ids", args.ally_ids,
        "--span", str(args.span),
        "--ckpt", args.ckpt,
        "--llm", args.llm,
        "--publish-hz", str(args.publish_hz),
        "--enemy-source", args.enemy_source,
        "--waypoints-out", str(_waypoints_path(args)),
    ]
    if args.model:
        argv += ["--model", args.model]
    if args.command:
        argv += ["--command", args.command]
    if args.net_log:
        argv += ["--net-log", args.net_log]
    if args.enemy_source == "bag":
        if not args.bag:
            raise ValueError("--enemy-source bag requires --bag")
        argv += ["--bag", args.bag]
    elif args.enemy_source == "gcs":
        if not args.target_ids:
            raise ValueError("--enemy-source gcs requires --target-ids")
        argv += ["--target-ids", args.target_ids]
    argv += ["--realtime"] if args.realtime else ["--no-realtime"]
    # --spf/--satellite/--pause-start are "--viz 전용" in run_gcs_bridge.py's own
    # --help -- passing them without --viz would be silently ignored there, so this
    # sidecar keeps the same "only meaningful together" rule rather than passing
    # dead flags through.
    if args.viz:
        argv += ["--viz", "--spf", str(args.spf)]
        if args.satellite:
            argv += ["--satellite"]
        if args.pause_start:
            argv += ["--pause-start"]
    return argv


# Fields a POST /start body may override for that one launch. Deliberately narrow:
# everything else (gcs-url, ally-ids, enemy-source, target-ids, llm, ...) stays a
# sidecar-startup flag per the comment above main()'s argparse block -- span and
# whether a matplotlib window pops up are the two things that legitimately vary
# run to run (map scale; whether a screen is available to look at), not deployment
# config a browser click should get to pick.
_OVERRIDABLE_FIELDS = {"span", "viz", "satellite", "spf", "pause_start"}


def _apply_overrides(args, overrides: dict):
    """Return a copy of `args` with `overrides` applied, or raise ValueError."""
    if not overrides:
        return args
    unknown = set(overrides) - _OVERRIDABLE_FIELDS
    if unknown:
        raise ValueError(f"unknown override field(s): {sorted(unknown)}; "
                          f"allowed: {sorted(_OVERRIDABLE_FIELDS)}")
    effective = argparse.Namespace(**vars(args))
    if "span" in overrides:
        try:
            span = float(overrides["span"])
        except (TypeError, ValueError):
            raise ValueError("span must be a number")
        if not 1.0 <= span <= 1000.0:
            raise ValueError("span must be between 1 and 1000 (metres)")
        effective.span = span
    if "viz" in overrides:
        effective.viz = bool(overrides["viz"])
    if "satellite" in overrides:
        effective.satellite = bool(overrides["satellite"])
    if "spf" in overrides:
        try:
            spf = int(overrides["spf"])
        except (TypeError, ValueError):
            raise ValueError("spf must be an integer")
        if spf < 1:
            raise ValueError("spf must be >= 1")
        effective.spf = spf
    if "pause_start" in overrides:
        effective.pause_start = bool(overrides["pause_start"])
    return effective


def _is_running() -> bool:
    return _proc is not None and _proc.poll() is None


def _stop_locked(timeout: float = 5.0) -> None:
    """Caller must hold _lock. Sends SIGINT first so run_gcs_bridge.py's own
    `finally: net_sink.shutdown()` / bag.shutdown() cleanup runs -- the same
    path a Ctrl+C on its terminal would take -- before falling back to a
    harder kill for a process that refuses to exit."""
    global _proc, _started_at
    if _proc is None:
        return
    if _proc.poll() is None:
        _proc.send_signal(signal.SIGINT)
        try:
            _proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _proc.terminate()
            try:
                _proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                _proc.kill()
                _proc.wait(timeout=2)
    _proc = None
    _started_at = None


def start(args, overrides: dict | None = None) -> dict:
    global _proc, _log_path, _started_at
    effective = _apply_overrides(args, overrides)  # raises ValueError before touching any state
    argv = _build_argv(effective)
    with _lock:
        _stop_locked()
        _log_path = Path(effective.log_dir) / f"gcs_bridge_{int(time.time())}.log"
        _log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(_log_path, "wb", buffering=0)
        _proc = subprocess.Popen(
            argv, cwd=str(MOBROBGPT_ROOT),
            stdout=log_f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        )
        _started_at = time.time()
        pid = _proc.pid
    return {"ok": True, "pid": pid, "log": str(_log_path), "argv": argv,
            "span": effective.span, "viz": effective.viz}


def stop() -> dict:
    with _lock:
        was_running = _is_running()
        _stop_locked()
    return {"ok": True, "was_running": was_running}


def _log_tail(n: int = 20) -> list[str]:
    if _log_path is None or not _log_path.exists():
        return []
    try:
        with open(_log_path, "r", encoding="utf-8", errors="replace") as f:
            return f.readlines()[-n:]
    except OSError:
        return []


def status() -> dict:
    with _lock:
        running = _is_running()
        return {
            "running": running,
            "pid": _proc.pid if running else None,
            "started_at": _started_at if running else None,
            "log": str(_log_path) if _log_path else None,
            "log_tail": _log_tail(),
        }


def make_handler(args):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            print(f"[gcs-bridge-control] {fmt % a}", flush=True)

        def _send(self, code, body):
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "content-type")
            self.end_headers()

        def do_GET(self):
            if self.path == "/status":
                self._send(200, status())
                return
            if self.path == "/waypoints":
                # Gated on _is_running(), not just file existence: otherwise a
                # stopped/crashed run's last waypoints_latest.json keeps being
                # served forever, and the map would keep drawing wp1->wp2
                # lines for a policy that isn't running any more -- unlike
                # /status, which already reports "정지됨" the instant it stops.
                if not _is_running():
                    self._send(200, {})
                    return
                path = _waypoints_path(args)
                if not path.exists():
                    self._send(200, {})
                    return
                try:
                    self._send(200, json.loads(path.read_text()))
                except (OSError, json.JSONDecodeError):
                    # A reader can race the sink's temp+rename write; the next
                    # poll (gcs's web UI polls this every ~1s) just tries again.
                    self._send(200, {})
                return
            self._send(404, {"ok": False, "detail": "unknown path"})

        def _read_json_body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if length == 0:
                return {}
            raw = self.rfile.read(length)
            try:
                body = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                raise ValueError(f"invalid JSON body: {e}")
            if not isinstance(body, dict):
                raise ValueError("body must be a JSON object")
            return body

        def do_POST(self):
            if self.path == "/start":
                try:
                    overrides = self._read_json_body()
                except ValueError as e:
                    self._send(400, {"ok": False, "detail": str(e)})
                    return
                try:
                    self._send(200, start(args, overrides))
                except ValueError as e:
                    self._send(400, {"ok": False, "detail": str(e)})
                except Exception as e:
                    self._send(500, {"ok": False, "detail": str(e)})
                return
            if self.path == "/stop":
                self._send(200, stop())
                return
            self._send(404, {"ok": False, "detail": "unknown path"})

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8093)
    ap.add_argument("--log-dir", default=str(MOBROBGPT_ROOT / "logs" / "gcs_bridge_control"))

    # Passed straight through to run_gcs_bridge.py -- see its own --help for
    # what each one means. Kept as sidecar-startup flags, not per-request
    # body fields, on purpose: this is a fixed deployment config (mock vs.
    # real hardware), not something a browser click should get to choose.
    # --span/--viz/--satellite/--spf/--pause-start are the deliberate exception --
    # they set this process's own defaults, but POST /start's JSON body can
    # override any of them for that one launch (see _OVERRIDABLE_FIELDS /
    # _apply_overrides above and the module docstring's "Per-start overrides").
    ap.add_argument("--gcs-url", default="http://127.0.0.1:8080")
    ap.add_argument("--ally-ids", required=True)
    ap.add_argument("--span", type=float, default=35.0,
                     help="run_gcs_bridge.py --span 기본값(실제 운용 박스 한 변, m) -- "
                          "POST /start 본문의 span으로 매 실행마다 덮어쓸 수 있음")
    ap.add_argument("--viz", action="store_true",
                     help="run_gcs_bridge.py --viz 기본값 -- POST /start 본문의 viz(bool)로 덮어씀")
    ap.add_argument("--satellite", action="store_true",
                     help="run_gcs_bridge.py --satellite 기본값 (--viz 켜졌을 때만 의미 있음)")
    ap.add_argument("--spf", type=int, default=3,
                     help="run_gcs_bridge.py --spf 기본값 (--viz 켜졌을 때만 의미 있음)")
    ap.add_argument("--pause-start", dest="pause_start", action="store_true",
                     help="run_gcs_bridge.py --pause-start 기본값 (--viz 켜졌을 때만 의미 있음)")
    ap.add_argument("--ckpt", default="boatattack_sim/models/u-net_map.pt")
    ap.add_argument("--llm", default="heuristic", choices=["ollama", "openai", "heuristic"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--command", default=None)
    ap.add_argument("--publish-hz", type=float, default=2.0)
    ap.add_argument("--enemy-source", default="live", choices=["bag", "live", "gcs"],
                     help="live=raw ROS2 토픽 직접 구독(그 토픽이 GCS datum 기준 프레임이 "
                          "아니면, 예: bag_enemy_relay.py --source pose 로 재투영해 중계하는 "
                          "이 mixed-demo 같은 구성이면 아군과 좌표계가 어긋난다). "
                          "gcs=--target-ids 로 준 role=target 배들을 GCS `/api/state`에서 "
                          "읽는다(재투영 문제 없음, 2026-09-10 세션에서 검증) -- 이 사이드카가 "
                          "붙는 mixed-demo 구성에는 이쪽을 권장.")
    ap.add_argument("--bag", default=None)
    ap.add_argument("--target-ids", default=None,
                     help="--enemy-source gcs 일 때 필수. 쉼표구분 GCS vehicle_id(role=target) "
                          "목록, 예: usv4,usv5,usv6,usv7,usv8")
    ap.add_argument("--net-log", default=None,
                     help="net-deploy JSON-lines log path -- forwarded to run_gcs_bridge.py "
                          "unchanged; this sidecar never reads or reacts to it")
    ap.add_argument("--realtime", dest="realtime", action="store_true", default=True)
    ap.add_argument("--no-realtime", dest="realtime", action="store_false")
    args = ap.parse_args()

    if args.enemy_source == "bag" and not args.bag:
        ap.error("--enemy-source bag requires --bag")
    if args.enemy_source == "gcs" and not args.target_ids:
        ap.error("--enemy-source gcs requires --target-ids")

    server = ThreadingHTTPServer((args.host, args.port), make_handler(args))
    print(f"[gcs-bridge-control] POST http://{args.host}:{args.port}/start to launch "
          f"run_gcs_bridge.py, /stop to end it, GET /status to poll it", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop()


if __name__ == "__main__":
    main()
