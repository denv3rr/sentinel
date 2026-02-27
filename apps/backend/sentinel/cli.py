from __future__ import annotations

import argparse
import signal
import sys
import threading
import webbrowser

import uvicorn

from sentinel.main import create_app


def _url_for_browser(bind: str, port: int) -> str:
    host = "127.0.0.1" if bind == "0.0.0.0" else bind
    return f"http://{host}:{port}"


def _build_parser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Sentinel local-first security recognition app")
    parser.add_argument("--data-dir", default=None, help="Path for runtime data (SQLite/media/logs/exports)")
    parser.add_argument("--bind", default="127.0.0.1", help="Bind host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default 8765)")
    parser.add_argument("--no-open", action="store_true", help="Do not auto-open browser")
    parser.add_argument("--log-level", default="info", help="Uvicorn log level")
    return parser


def _run(parsed: argparse.Namespace, force_open: bool | None = None) -> int:
    should_open = not parsed.no_open if force_open is None else force_open
    if parsed.bind == "0.0.0.0":
        print("[warning] LAN access enabled. Keep Sentinel on trusted networks and do not expose publicly.")

    app = create_app(
        data_dir=parsed.data_dir,
        bind=parsed.bind,
        port=parsed.port,
        log_level=parsed.log_level,
    )
    url = _url_for_browser(parsed.bind, parsed.port)
    browser_timer: threading.Timer | None = None
    early_shutdown_started = threading.Event()

    if should_open:
        browser_timer = threading.Timer(0.9, lambda: webbrowser.open(url))
        browser_timer.start()

    print(f"Sentinel running at {url}")
    config = uvicorn.Config(
        app,
        host=parsed.bind,
        port=parsed.port,
        log_level=parsed.log_level,
        workers=1,
        timeout_graceful_shutdown=2,
        timeout_keep_alive=1,
    )
    server = uvicorn.Server(config)
    previous_handlers: dict[int, object] = {}
    run_exit_code: int | None = None

    def _begin_runtime_shutdown() -> None:
        if early_shutdown_started.is_set():
            return
        early_shutdown_started.set()
        app_state = getattr(app, "state", None)
        sentinel_state = getattr(app_state, "sentinel", None)
        if sentinel_state is None:
            return
        try:
            sentinel_state.begin_shutdown()
        except Exception:
            pass

    def _finalize_state_shutdown() -> None:
        app_state = getattr(app, "state", None)
        sentinel_state = getattr(app_state, "sentinel", None)
        if sentinel_state is None:
            return
        try:
            sentinel_state.shutdown()
        except Exception:
            pass

    def _request_exit(signum: int, _frame: object) -> None:
        if signum in {signal.SIGINT, signal.SIGTERM}:
            _begin_runtime_shutdown()
            server.should_exit = True

    def _request_exit_from_api() -> None:
        _begin_runtime_shutdown()
        server.should_exit = True

    setattr(app.state, "request_exit", _request_exit_from_api)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, _request_exit)
        except (AttributeError, ValueError):
            continue

    try:
        try:
            server.run()
        except KeyboardInterrupt:
            _begin_runtime_shutdown()
            server.should_exit = True
        except SystemExit as exc:
            if server.should_exit or early_shutdown_started.is_set():
                run_exit_code = 0
            else:
                code = exc.code
                run_exit_code = code if isinstance(code, int) else 1
    finally:
        _begin_runtime_shutdown()
        _finalize_state_shutdown()
        if browser_timer is not None:
            browser_timer.cancel()
            browser_timer.join(timeout=0.5)
        for sig, handler in previous_handlers.items():
            try:
                signal.signal(sig, handler)
            except (AttributeError, ValueError):
                continue
    if run_exit_code is not None:
        return run_exit_code
    if bool(getattr(server, "started", False)) or server.should_exit:
        return 0
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser("sentinel")
    parsed = parser.parse_args(argv)
    try:
        return _run(parsed)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
