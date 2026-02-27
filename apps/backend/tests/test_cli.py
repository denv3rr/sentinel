from __future__ import annotations

import argparse
from types import SimpleNamespace

from sentinel import cli


def _parsed() -> argparse.Namespace:
    return cli._build_parser("sentinel").parse_args(["--no-open", "--bind", "127.0.0.1", "--port", "8877"])


def _fake_app(begin_shutdown_calls: list[int]) -> object:
    sentinel = SimpleNamespace(
        begin_shutdown=lambda: begin_shutdown_calls.append(1),
    )
    return SimpleNamespace(state=SimpleNamespace(sentinel=sentinel))


def test_cli_returns_zero_on_keyboard_interrupt(monkeypatch) -> None:
    shutdown_calls: list[int] = []
    monkeypatch.setattr(cli, "create_app", lambda **kwargs: _fake_app(shutdown_calls))
    monkeypatch.setattr(cli.uvicorn, "Config", lambda *args, **kwargs: object())

    class _InterruptServer:
        def __init__(self, _config: object) -> None:
            self.should_exit = False
            self.started = True

        def run(self) -> None:
            raise KeyboardInterrupt

    monkeypatch.setattr(cli.uvicorn, "Server", _InterruptServer)

    assert cli._run(_parsed()) == 0
    assert len(shutdown_calls) == 1


def test_cli_returns_nonzero_when_server_never_starts(monkeypatch) -> None:
    shutdown_calls: list[int] = []
    monkeypatch.setattr(cli, "create_app", lambda **kwargs: _fake_app(shutdown_calls))
    monkeypatch.setattr(cli.uvicorn, "Config", lambda *args, **kwargs: object())

    class _NeverStartedServer:
        def __init__(self, _config: object) -> None:
            self.should_exit = False
            self.started = False

        def run(self) -> None:
            return None

    monkeypatch.setattr(cli.uvicorn, "Server", _NeverStartedServer)

    assert cli._run(_parsed()) == 1
    assert len(shutdown_calls) == 1


def test_cli_forces_single_uvicorn_worker(monkeypatch) -> None:
    shutdown_calls: list[int] = []
    monkeypatch.setattr(cli, "create_app", lambda **kwargs: _fake_app(shutdown_calls))
    captured: dict[str, object] = {}

    def _capture_config(*_args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(cli.uvicorn, "Config", _capture_config)

    class _StartedServer:
        def __init__(self, _config: object) -> None:
            self.should_exit = False
            self.started = True

        def run(self) -> None:
            return None

    monkeypatch.setattr(cli.uvicorn, "Server", _StartedServer)

    assert cli._run(_parsed()) == 0
    assert captured["workers"] == 1
    assert len(shutdown_calls) == 1


def test_main_returns_zero_on_interrupt(monkeypatch) -> None:
    def _raise_interrupt(_parsed: argparse.Namespace, force_open: bool | None = None) -> int:
        _ = force_open
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_run", _raise_interrupt)
    assert cli.main(["--no-open"]) == 0


def test_cli_treats_system_exit_after_should_exit_as_clean(monkeypatch) -> None:
    shutdown_calls: list[int] = []
    monkeypatch.setattr(cli, "create_app", lambda **kwargs: _fake_app(shutdown_calls))
    monkeypatch.setattr(cli.uvicorn, "Config", lambda *args, **kwargs: object())

    class _SystemExitServer:
        def __init__(self, _config: object) -> None:
            self.should_exit = True
            self.started = True

        def run(self) -> None:
            raise SystemExit(1)

    monkeypatch.setattr(cli.uvicorn, "Server", _SystemExitServer)

    assert cli._run(_parsed()) == 0
    assert len(shutdown_calls) == 1
