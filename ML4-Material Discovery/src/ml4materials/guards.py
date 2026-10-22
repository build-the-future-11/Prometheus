"""Low-cost process RSS and macOS pageout stop conditions."""

from __future__ import annotations

from dataclasses import dataclass
import re
import resource
import subprocess
import sys


class ResourceLimitError(RuntimeError):
    """Raised when a frozen compute admission condition is violated."""


def current_pageouts() -> int | None:
    if sys.platform.startswith("linux"):
        try:
            vmstat = open("/proc/vmstat", encoding="utf-8").read()
        except OSError:
            return None
        match = re.search(r"^pswpout\s+(\d+)$", vmstat, flags=re.MULTILINE)
        return int(match.group(1)) if match else None
    try:
        completed = subprocess.run(
            ["vm_stat"], capture_output=True, check=True, text=True, timeout=5
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    match = re.search(r"Pageouts:\s+(\d+)\.", completed.stdout)
    return int(match.group(1)) if match else None


def max_rss_mib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


@dataclass
class ResourceGuard:
    max_allowed_rss_mib: float
    require_no_pageout_increase: bool
    starting_pageouts: int | None = None
    peak_observed_rss_mib: float = 0.0

    def __post_init__(self) -> None:
        self.starting_pageouts = current_pageouts()
        self.check("guard_initialization")

    def check(self, phase: str) -> dict[str, float | int | str | None]:
        rss = max_rss_mib()
        self.peak_observed_rss_mib = max(self.peak_observed_rss_mib, rss)
        pageouts = current_pageouts()
        if rss > self.max_allowed_rss_mib:
            raise ResourceLimitError(
                f"RSS guard refused phase {phase}: {rss:.1f} MiB > {self.max_allowed_rss_mib:.1f} MiB"
            )
        if (
            self.require_no_pageout_increase
            and self.starting_pageouts is not None
            and pageouts is not None
            and pageouts > self.starting_pageouts
        ):
            raise ResourceLimitError(
                f"pageout guard refused phase {phase}: {pageouts} > {self.starting_pageouts}"
            )
        return {
            "phase": phase,
            "max_rss_mib": rss,
            "pageouts": pageouts,
        }

    def receipt(self) -> dict[str, float | int | bool | None]:
        ending_pageouts = current_pageouts()
        return {
            "max_allowed_rss_mib": self.max_allowed_rss_mib,
            "peak_observed_rss_mib": self.peak_observed_rss_mib,
            "require_no_pageout_increase": self.require_no_pageout_increase,
            "pageout_monitor_available": (
                self.starting_pageouts is not None and ending_pageouts is not None
            ),
            "starting_pageouts": self.starting_pageouts,
            "ending_pageouts": ending_pageouts,
            "pageout_delta": (
                ending_pageouts - self.starting_pageouts
                if ending_pageouts is not None and self.starting_pageouts is not None
                else None
            ),
        }
