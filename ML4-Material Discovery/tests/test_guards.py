import pytest

import ml4materials.guards as guards


def test_resource_guard_records_stable_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(guards, "current_pageouts", lambda: 10)
    monkeypatch.setattr(guards, "max_rss_mib", lambda: 100.0)
    guard = guards.ResourceGuard(512.0, True)
    assert guard.check("fixture")["pageouts"] == 10
    assert guard.receipt()["pageout_delta"] == 0
    assert guard.receipt()["pageout_monitor_available"] is True


def test_linux_pageout_counter_is_parsed(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    vmstat = tmp_path / "vmstat"
    vmstat.write_text("pgpgin 1\npswpout 42\npgpgout 2\n", encoding="utf-8")
    original_open = open

    def fake_open(path, *args, **kwargs):
        return original_open(vmstat if path == "/proc/vmstat" else path, *args, **kwargs)

    monkeypatch.setattr(guards.sys, "platform", "linux")
    monkeypatch.setattr("builtins.open", fake_open)
    assert guards.current_pageouts() == 42


def test_resource_guard_refuses_memory_and_pageout_growth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pageouts = iter([10, 10, 11])
    monkeypatch.setattr(guards, "current_pageouts", lambda: next(pageouts))
    monkeypatch.setattr(guards, "max_rss_mib", lambda: 100.0)
    guard = guards.ResourceGuard(512.0, True)
    with pytest.raises(guards.ResourceLimitError, match="pageout"):
        guard.check("growth")

    monkeypatch.setattr(guards, "current_pageouts", lambda: 10)
    monkeypatch.setattr(guards, "max_rss_mib", lambda: 600.0)
    with pytest.raises(guards.ResourceLimitError, match="RSS"):
        guards.ResourceGuard(512.0, False)

