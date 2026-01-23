"""
Collection of retrain / skip policies.

Every policy object supports
    reset(history_df)            # called once before replay loop
    should_retrain(step_df, *)   # called every week
The old functional RES API is kept for backward compatibility.
"""
from __future__ import annotations
import random
from typing import Any, Dict
from contextlib import suppress

import mlflow
from src.controllers.page_hinkley import PageHinkley

# ──────────────────────────────────────────────────────────────
# 1.  Resource-Efficiency-Score policy
# ──────────────────────────────────────────────────────────────
class oldRESPolicy:
    """Benefit-to-cost rule with optional self-tuning λ."""

    def __init__(self, lam: float = 0.02, step: int = 8,
                 span: float = 0.50, auto: bool = True, **_):
        self.lam0 = self.lam = lam          # initial λ
        self.step = step                    # window length (weeks)
        self.span = span                    # ±20 % neighbourhood
        self.auto = auto
        self._hist: list[tuple[float, float]] = []  # (benefit, cost)

    # ---- core rule --------------------------------------------------
    @staticmethod
    def _res(benefit: float, cost_sec: float, lam: float) -> bool:
        if cost_sec <= 0:
            return benefit > 0
        return (benefit > 0) and (benefit / cost_sec >= lam)

    # ---- public API -------------------------------------------------
    def reset(self, *_):
        self.lam = self.lam0
        self._hist.clear()
        with suppress(Exception):
            mlflow.log_metric("lambda_auto", self.lam, step=0)


    def should_retrain(self, benefit: float, cost_sec: float,
                       step_idx: int = 0, **_) -> bool:
        if self.auto:
            self._hist.append((benefit, cost_sec))
            if len(self._hist) >= self.step:
                old = self.lam
                self.lam = self._lam_search()
                if abs(self.lam - old) > 1e-12:           # log only if changed
                    with suppress(Exception):
                        mlflow.log_metric("lambda_auto", self.lam,
                                          step=step_idx)
                self._hist.clear()
        return self._res(benefit, cost_sec, self.lam)

    # ----------------------------------------------------------
    def _lam_search(self) -> float:
        """One-step hill-climb: try λ·(1−span), λ, λ·(1+span)."""
        cand_lams = [max(1e-6, self.lam * (1 - self.span)),
                     self.lam,
                     self.lam * (1 + self.span)]

        best, best_util = self.lam, float("-inf")

        # unpack the window once
        ben, cst = zip(*self._hist)

        for lam in cand_lams:
            util = 0.0
            for b, c in zip(ben, cst):
                if (b > 0) and (b / c >= lam):      # would retrain
                    util += b - lam * c             # Δ utility that week
            if util > best_util:
                best, best_util = lam, util
        return best

    
    
# ──────────────────────────────────────────────────────────────
# 1.  Bounded-Efficiency Resource-Efficiency-Score policy
# ──────────────────────────────────────────────────────────────
class RESPolicy:
    """
    Benefit-to-cost rule with optional self-tuning λ.

    Score
        eff = benefit / (benefit + cost)            ∈ (0, 1)

    Decision
        retrain  ⇔  eff ≥ λ

    Interpretation
        max_cost_per_benefit = (1 − λ) / λ
        λ = 0.33  → willing to spend up to 2× benefit in cost
        λ = 0.50  → break-even (cost must not exceed benefit)
        λ = 0.80  → very strict (cost ≤ 0.25× benefit)
    """

    def __init__(self, lam: float = 0.33, step: int = 8,
                 span: float = 0.50, auto: bool = True, **_):
        self.lam0 = self.lam = lam          # initial λ
        self.step = step                    # window length (weeks)
        self.span = span                    # ±50 % neighbourhood
        self.auto = auto
        self._hist: list[tuple[float, float]] = []  # (benefit, cost)

    # ---- core rule --------------------------------------------------
    @staticmethod
    def _res(benefit: float, cost_sec: float, lam: float) -> bool:
        if benefit <= 0:
            return False                    # no gain → skip
        if cost_sec <= 0:
            return True                     # free retrain
        eff = benefit / (benefit + cost_sec)
        return eff >= lam

    # ---- public API -------------------------------------------------
    def reset(self, *_):
        self.lam = self.lam0
        self._hist.clear()
        with suppress(Exception):
            mlflow.log_metric("lambda_auto", self.lam, step=0)

    def should_retrain(self, benefit: float, cost_sec: float,
                       step_idx: int = 0, **_) -> bool:
        if self.auto:
            self._hist.append((benefit, cost_sec))
            if len(self._hist) >= self.step:
                old = self.lam
                self.lam = self._lam_search()
                if abs(self.lam - old) > 1e-12:          # log only if changed
                    with suppress(Exception):
                        mlflow.log_metric("lambda_auto", self.lam,
                                          step=step_idx)
                self._hist.clear()
        return self._res(benefit, cost_sec, self.lam)

    # ----------------------------------------------------------
    def _lam_search(self) -> float:
        """
        One-step hill-climb in λ-space.
        Converts λ → ρ = λ / (1-λ) to reuse the original utility.
        """
        cand_lams = [max(1e-6, self.lam * (1 - self.span)),
                     self.lam,
                     min(0.999999, self.lam * (1 + self.span))]

        ben, cst = zip(*self._hist)
        best, best_u = self.lam, float("-inf")

        for lam in cand_lams:
            rho = lam / (1.0 - lam)            # implied ratio threshold
            util = 0.0
            for b, c in zip(ben, cst):
                retrain = (b > 0) and (b / max(c, 1e-12) >= rho)
                if retrain:
                    util += b - rho * c
            if util > best_u:
                best, best_u = lam, util
        return best


# functional wrapper for legacy code ---------------------------------
def should_retrain(benefit: float, cost_sec: float, lam: float, **_) -> bool:
    return RESPolicy._res(benefit, cost_sec, lam)


# ──────────────────────────────────────────────────────────────
# 2-6.  Baseline policies
# ──────────────────────────────────────────────────────────────
class FixedIntervalPolicy:
    """Retrain every k replay steps."""
    def __init__(self, k: int = 4, **_):
        assert k > 0
        self.k, self._cnt = k, 0
    def reset(self, *_): self._cnt = 0
    def should_retrain(self, **__) -> bool:
        self._cnt += 1
        if self._cnt >= self.k:
            self._cnt = 0
            return True
        return False


class RandomBudgetPolicy:
    """Retrain with probability p (Bernoulli)."""
    def __init__(self, p: float = 0.15, seed: int | None = None, **_):
        assert 0.0 <= p <= 1.0
        self.p, self.rng = p, random.Random(seed)
    def reset(self, *_): pass
    def should_retrain(self, **__) -> bool: return self.rng.random() < self.p


class CARAPolicy:
    def __init__(self, tau: float = 0.03, **_): self.tau = tau
    def reset(self, *_): pass
    def should_retrain(self, mae_old: float, mae_new: float, **_) -> bool:
        return (mae_old - mae_new) > self.tau


class NeverPolicy:
    def reset(self, *_): pass
    def should_retrain(self, **_): return False


class AlwaysPolicy:
    def reset(self, *_): pass
    def should_retrain(self, **_): return True


# ──────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────
_REGISTRY = {
    "res": RESPolicy,
    "fixed": FixedIntervalPolicy,
    "random": RandomBudgetPolicy,
    "ph": PageHinkley,
    "page_hinkley": PageHinkley,
    "cara": CARAPolicy,
    "never": NeverPolicy,
    "always": AlwaysPolicy,
}


def make(cfg: Dict[str, Any]):
    """Instantiate a policy from YAML-like dict."""
    kind = cfg.get("kind", "res").lower()
    cls = _REGISTRY[kind]

    key_map = {
        "lambda": "lam", "lam": "lam",
        "k": "k", "p": "p", "tau": "tau",
        "auto": "auto", "step": "step", "span": "span"
    }
    kwargs = {key_map.get(k, k): v for k, v in cfg.items() if k != "kind"}
    return cls(**kwargs)




