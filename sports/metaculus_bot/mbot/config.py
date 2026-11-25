"""Configuration: every knob is an env var with a safe default. Nothing here reads a secret into a log."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent          # sports/metaculus_bot/mbot
BOT_DIR = PKG_DIR.parent                           # sports/metaculus_bot
SPORTS_DIR = BOT_DIR.parent                        # sports/

API_BASE_URL = "https://www.metaculus.com/api"
USER_AGENT = "mbot/0.1 (metaculus FutureEval bot; personal hobby project)"

# Tournament ids / slugs (Metaculus forecasting-tools 0.3.7 MetaculusClient constants, read 2026-11-25).
FALL_2026_FUTUREEVAL = "fall-futureeval-2026"      # id 33121; forecasting ends 2027-01-06
MINIBENCH = "minibench"                            # always the current two-week round
BOT_TESTING_AREA = "bot-testing-area"              # id 32977; the sanctioned place for test runs
BOT_BENCHMARK_PROJECT = 32979                      # resolved questions for evaluation (needs the Bot Benchmarking tier)

# MiniBench only by default: the pilot plan is "MiniBench first" under a $50/month cap (notes/metaculus_bot_tourney.md,
# "Plan if go"). ~60 MiniBench questions per 2-week round (~130/month) fit the cap; adding the seasonal tournament
# (~20-25 questions/week) roughly doubles the volume. Spring 2027 is not announced yet: set MBOT_TOURNAMENTS when it is.
DEFAULT_TOURNAMENTS = MINIBENCH

SUPPORTED_TYPES = ("binary", "multiple_choice", "numeric", "discrete", "date")


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    if v is None or not v.strip() or v.strip() in {"REPLACE_ME", "1234567890", "your-token-here"}:
        return default
    return v.strip()


def _float(name: str, default: float) -> float:
    v = _env(name)
    return float(v) if v is not None else default


def _int(name: str, default: int) -> int:
    v = _env(name)
    return int(v) if v is not None else default


@dataclass
class Config:
    # --- where things go
    state_dir: Path = field(default_factory=lambda: Path(_env("MBOT_STATE_DIR", str(Path.home() / ".claude" / "quant-sports" / "data" / "metaculus_bot"))))
    fund_scripts_dir: Path = field(default_factory=lambda: Path(_env("MBOT_FUND_SCRIPTS_DIR", str(SPORTS_DIR / "scripts"))))
    # --- what to forecast
    tournaments: list[str] = field(default_factory=lambda: [t.strip() for t in (_env("MBOT_TOURNAMENTS", DEFAULT_TOURNAMENTS) or "").split(",") if t.strip()])
    min_minutes_left: float = field(default_factory=lambda: _float("MBOT_MIN_MINUTES_LEFT", 8.0))
    max_questions_per_run: int = field(default_factory=lambda: _int("MBOT_MAX_QUESTIONS_PER_RUN", 25))
    # --- models
    provider_spec: str = field(default_factory=lambda: _env("MBOT_MODELS", "anthropic:claude-opus-5-5") or "")
    n_samples: int = field(default_factory=lambda: _int("MBOT_N_SAMPLES", 5))
    min_valid_samples: int = field(default_factory=lambda: _int("MBOT_MIN_VALID_SAMPLES", 1))
    max_output_tokens: int = field(default_factory=lambda: _int("MBOT_MAX_OUTPUT_TOKENS", 16000))
    effort: str = field(default_factory=lambda: _env("MBOT_EFFORT", "high") or "high")
    # budget pacing: once less than `fallback_below_usd` of the monthly cap is left, switch to a cheaper ensemble so
    # coverage (the main driver of prize money) continues instead of stopping at the cap. Empty = no fallback.
    fallback_spec: str = field(default_factory=lambda: _env("MBOT_FALLBACK_MODELS", "") or "")
    fallback_below_usd: float = field(default_factory=lambda: _float("MBOT_FALLBACK_BELOW_USD", 10.0))
    fallback_n_samples: int = field(default_factory=lambda: _int("MBOT_FALLBACK_N_SAMPLES", 3))
    # --- money
    monthly_cap_usd: float = field(default_factory=lambda: _float("MBOT_MONTHLY_CAP_USD", 50.0))
    # --- calibration guards (notes/metaculus_bot_tourney.md sec 3: clipping helped r +.48, extremizing hurt r -.30)
    binary_clip_lo: float = field(default_factory=lambda: _float("MBOT_BINARY_CLIP_LO", 0.02))
    binary_clip_hi: float = field(default_factory=lambda: _float("MBOT_BINARY_CLIP_HI", 0.98))
    resolved_clip_lo: float = 0.005     # only when a data tool (not the LLM) says the outcome is already determined
    resolved_clip_hi: float = 0.995
    mc_floor: float = field(default_factory=lambda: _float("MBOT_MC_FLOOR", 0.02))
    extremize: float = field(default_factory=lambda: _float("MBOT_EXTREMIZE", 1.0))   # 1.0 = off (evidence says off)
    # --- research
    use_data_tools: bool = field(default_factory=lambda: (_env("MBOT_DATA_TOOLS", "1") or "1") != "0")
    research_chars: int = 9000          # cap on research text handed to each sample

    # secrets (never logged; only presence is reported)
    @property
    def metaculus_token(self) -> str | None:
        return _env("METACULUS_TOKEN")

    def secrets_present(self) -> dict[str, bool]:
        names = ["METACULUS_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY",
                 "ASKNEWS_CLIENT_ID", "ASKNEWS_SECRET", "BEA_API_KEY", "BLS_API_KEY"]
        return {n: _env(n) is not None for n in names}

    @property
    def ledger_path(self) -> Path:
        return self.state_dir / "forecasts.jsonl"

    @property
    def lock_path(self) -> Path:
        return self.state_dir / "run.lock"
