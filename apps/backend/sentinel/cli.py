from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
import signal
import sys
import threading
import webbrowser

import uvicorn

from sentinel.ml.runners import run_benchmark, run_eval, run_export, run_train
from sentinel.main import create_app

_KNOWN_COMMANDS = {"serve", "train", "eval", "export", "benchmark"}


def _url_for_browser(bind: str, port: int) -> str:
    host = "127.0.0.1" if bind == "0.0.0.0" else bind
    return f"http://{host}:{port}"


def _build_parser(prog: str) -> argparse.ArgumentParser:
    """Legacy parser (no explicit command), kept for backward compatibility."""
    parser = argparse.ArgumentParser(prog=prog, description="Sentinel local-first security recognition app")
    parser.add_argument("--data-dir", default=None, help="Path for runtime data (SQLite/media/logs/exports)")
    parser.add_argument("--bind", default="127.0.0.1", help="Bind host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default 8765)")
    parser.add_argument("--no-open", action="store_true", help="Do not auto-open browser")
    parser.add_argument("--log-level", default="info", help="Uvicorn log level")
    return parser


def _build_command_parser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Sentinel local-first security recognition app")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Run Sentinel API + UI server")
    serve.add_argument("--data-dir", default=None, help="Path for runtime data (SQLite/media/logs/exports)")
    serve.add_argument("--bind", default="127.0.0.1", help="Bind host (default 127.0.0.1)")
    serve.add_argument("--port", type=int, default=8765, help="Bind port (default 8765)")
    serve.add_argument("--no-open", action="store_true", help="Do not auto-open browser")
    serve.add_argument("--log-level", default="info", help="Uvicorn log level")

    train = subparsers.add_parser("train", help="Train a detector model and emit a manifest")
    train.add_argument("--dataset-config", required=True, help="Path to dataset YAML/JSON config")
    train.add_argument("--model", default="yolov8n.pt", help="Base model or checkpoint path")
    train.add_argument("--output-dir", default="artifacts/ml/train", help="Training output directory")
    train.add_argument("--alias", default=None, help="Optional registry alias for manifest")
    train.add_argument("--profile", default="balanced", help="Detector profile metadata")
    train.add_argument("--image-size", type=int, default=960, help="Training image size")
    train.add_argument("--epochs", type=int, default=10, help="Training epochs")
    train.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    train.add_argument("--confidence", type=float, default=0.25, help="Default confidence threshold")
    train.add_argument("--seed", type=int, default=7, help="Reproducibility seed")
    train.add_argument("--deterministic", action="store_true", help="Enable deterministic flags where possible")
    train.add_argument("--registry-path", default="artifacts/ml/registry.json", help="Model registry JSON path")
    train.add_argument("--dry-run", action="store_true", help="Run pipeline without invoking training backend")

    evaluate = subparsers.add_parser("eval", help="Evaluate a registered manifest model")
    evaluate.add_argument("--model", required=True, help="Model alias or manifest path")
    evaluate.add_argument("--dataset-config", required=True, help="Path to dataset YAML/JSON config")
    evaluate.add_argument("--output-dir", default="artifacts/ml/eval", help="Evaluation output directory")
    evaluate.add_argument("--image-size", type=int, default=None, help="Override eval image size")
    evaluate.add_argument("--seed", type=int, default=7, help="Reproducibility seed")
    evaluate.add_argument("--deterministic", action="store_true", help="Enable deterministic flags where possible")
    evaluate.add_argument("--registry-path", default="artifacts/ml/registry.json", help="Model registry JSON path")
    evaluate.add_argument("--dry-run", action="store_true", help="Run pipeline without invoking eval backend")

    export = subparsers.add_parser("export", help="Export a registered manifest model")
    export.add_argument("--model", required=True, help="Model alias or manifest path")
    export.add_argument("--output-dir", default="artifacts/ml/export", help="Export output directory")
    export.add_argument("--format", dest="export_format", default="onnx", help="Export format (onnx, torchscript...)")
    export.add_argument("--image-size", type=int, default=None, help="Override export image size")
    export.add_argument("--optimize", action="store_true", help="Enable export optimization flags")
    export.add_argument("--registry-path", default="artifacts/ml/registry.json", help="Model registry JSON path")
    export.add_argument("--dry-run", action="store_true", help="Run pipeline without invoking export backend")

    benchmark = subparsers.add_parser("benchmark", help="Run detector micro-benchmarks")
    benchmark.add_argument("--model", default="yolov8n.pt", help="Model path")
    benchmark.add_argument("--profile", default="balanced", help="Detector profile")
    benchmark.add_argument("--confidence", type=float, default=0.25, help="Detector confidence")
    benchmark.add_argument("--frames", type=int, default=120, help="Benchmark frames")
    benchmark.add_argument("--width", type=int, default=640, help="Synthetic frame width")
    benchmark.add_argument("--height", type=int, default=360, help="Synthetic frame height")

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

    app.state.request_exit = _request_exit_from_api

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


def _print_result(result: object) -> None:
    if is_dataclass(result):
        payload = asdict(result)
    else:
        payload = result
    print(json.dumps(payload, indent=2, sort_keys=True))


def _dispatch_command(parsed: argparse.Namespace) -> int:
    if parsed.command == "serve":
        return _run(parsed)
    if parsed.command == "train":
        result = run_train(
            dataset_config_path=parsed.dataset_config,
            model_name=parsed.model,
            output_dir=parsed.output_dir,
            alias=parsed.alias,
            profile=parsed.profile,
            image_size=parsed.image_size,
            epochs=parsed.epochs,
            batch_size=parsed.batch_size,
            confidence=parsed.confidence,
            seed=parsed.seed,
            deterministic=parsed.deterministic,
            registry_path=parsed.registry_path,
            dry_run=parsed.dry_run,
        )
        _print_result(result)
        return 0
    if parsed.command == "eval":
        result = run_eval(
            model=parsed.model,
            dataset_config_path=parsed.dataset_config,
            output_dir=parsed.output_dir,
            image_size=parsed.image_size,
            seed=parsed.seed,
            deterministic=parsed.deterministic,
            registry_path=parsed.registry_path,
            dry_run=parsed.dry_run,
        )
        _print_result(result)
        return 0
    if parsed.command == "export":
        result = run_export(
            model=parsed.model,
            output_dir=parsed.output_dir,
            export_format=parsed.export_format,
            image_size=parsed.image_size,
            optimize=parsed.optimize,
            registry_path=parsed.registry_path,
            dry_run=parsed.dry_run,
        )
        _print_result(result)
        return 0
    if parsed.command == "benchmark":
        result = run_benchmark(
            model_name=parsed.model,
            profile=parsed.profile,
            confidence=parsed.confidence,
            frames=parsed.frames,
            frame_width=parsed.width,
            frame_height=parsed.height,
        )
        _print_result(result)
        return 0
    raise ValueError(f"Unknown command: {parsed.command}")


def main(argv: list[str] | None = None) -> int:
    args = list(argv or [])
    try:
        if args and args[0] in _KNOWN_COMMANDS:
            parser = _build_command_parser("sentinel")
            parsed = parser.parse_args(args)
            return _dispatch_command(parsed)
        parser = _build_parser("sentinel")
        parsed = parser.parse_args(args)
        return _run(parsed)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"[error] {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
