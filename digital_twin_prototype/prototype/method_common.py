"""Shared paths, rules and methods for the 22 Oct 2026 confirmatory experiment.

Rule: interval methods receive calibration data and test INPUTS only.
Test labels enter only metrics(). verify.py checks this with a label-shuffle test.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
SRC = BASE / "Thesis_C_Results_2026-10-20"
W12 = SRC / "Thesis_C_Weeks_1_2"
W34 = SRC / "Thesis_C_Weeks_3_4"

TASKS = ["depth", "width", "length"]
ALPHA = 0.1                      # 90% nominal level, as in the frozen Weeks 3-4 protocol
BUDGETS = [9, 12, 15, 20, 30]    # 9 = smallest n with a finite 90% conformal interval
REPEATS = 50
TEST_N = 20
MIN_TEST_N = 15
PRIMARY_MIN_ROWS = 40            # same primary cohort rule as Weeks 3-4 (pool >= 20 after 20 test rows)
SECONDARY_MIN_ROWS = 29          # same 9-only rule as Weeks 3-4
SEED_BASE = 20261022
TARGET_ONLY_FEATURES = ["Power", "Velocity"]
SUSPECT_IDS = set(range(580, 583)) | set(range(837, 845))
LEAKY_COLUMNS = {"absorptivity_2", "absorptivity_2_calc", "absorption coefficient 2",
                 "width of melt pool", "depth of meltpool", "length of melt pool",
                 "d/l", "d/w", "l/w", "meltpool shape", "label"}

METHODS = {
    # name: (family, description)
    "source_split":      ("existing", "Frozen ML point, split conformal on other alloys' reserved calibration rows"),
    "target_split":      ("existing", "Frozen ML point, split conformal on target residuals (20 Oct Thesis C method)"),
    "mean_target_split": ("baseline", "Training mean, split conformal on target residuals (20 Oct simple predictor)"),
    "target_mean_jk":    ("baseline", "Target-sample mean, jackknife+ (training mean plus shift update)"),
    "target_linE_jk":    ("baseline", "Target-only OLS y ~ 1 + P/V, jackknife+"),
    "target_linPV_jk":   ("baseline", "Target-only OLS y ~ 1 + P + V, jackknife+"),
    "target_auto_jk":    ("baseline", "Target-only, choose mean / P/V / P+V by inner LOO MSE, jackknife+"),
    "ml_shift_jk":       ("proposed", "Frozen ML plus intercept update r ~ 1, jackknife+"),
    "ml_linear_jk":      ("proposed", "Frozen ML plus intercept and slope update r ~ 1 + p, jackknife+"),
    "ml_auto_jk":        ("proposed", "Frozen ML, choose shift or linear update by inner LOO MSE, jackknife+ (PRIMARY)"),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False, default=str), encoding="utf-8")


def rank_k(n, alpha=ALPHA):
    """ceil((1-alpha)(n+1)) with integer arithmetic for alpha = 0.1."""
    assert alpha == 0.1
    return (9 * (n + 1) + 9) // 10


def split_q(scores):
    s = np.sort(np.asarray(scores, float))
    if s.ndim != 1 or not np.isfinite(s).all() or (s < 0).any():
        raise ValueError("bad scores")
    k = rank_k(len(s))
    return float(s[k - 1]) if k <= len(s) else np.inf


# ---------- least squares helpers (calibration data only) ----------
def design(name, p, X):
    """Design matrix for one candidate model. p = ML prediction, X = [Power, Velocity]."""
    n = len(p)
    one = np.ones((n, 1))
    if name in ("shift", "mean"):
        return one
    if name == "linear":
        return np.column_stack([one, p])
    if name == "linE":
        return np.column_stack([one, X[:, 0] / X[:, 1]])
    if name == "linPV":
        return np.column_stack([one, X])
    raise KeyError(name)


def ols_fit(A, y):
    return np.linalg.pinv(A) @ y


def ols_loo_mse(A, y):
    """Exact leave-one-out MSE via the hat matrix; inf when a point has leverage 1."""
    P = np.linalg.pinv(A)
    h = np.einsum("ij,ji->i", A, P)
    e = y - A @ (P @ y)
    if np.any(1 - h < 1e-10):
        return np.inf
    return float(np.mean((e / (1 - h)) ** 2))


def choose(cands, p, X, t):
    """Pick the candidate with the lowest inner LOO MSE; ties go to the earlier (simpler) one."""
    scores = [ols_loo_mse(design(c, p, X), t) for c in cands]
    return cands[int(np.argmin(scores))]


def jackknife_plus(cal, test_inputs, cands, response):
    """Jackknife+ (Barber et al. 2021) for a small OLS family.

    cal: dict with y, p (ML prediction), X; test_inputs: dict with p, X.
    response 'residual' models r = y - p and adds p back; 'direct' models y.
    With more than one candidate, the choice is made INSIDE each fold, so the
    fitted algorithm is symmetric and the jackknife+ guarantee holds.
    Returns lower, upper, point, chosen-on-full-data.
    """
    y, p, X = cal["y"], cal["p"], cal["X"]
    pt, Xt = test_inputs["p"], test_inputs["X"]
    t = y - p if response == "residual" else y
    off_c = p if response == "residual" else np.zeros_like(y)
    off_t = pt if response == "residual" else np.zeros_like(pt)
    n = len(y)
    k = rank_k(n)
    if k > n:
        inf = np.full(len(pt), np.inf)
        return -inf, inf, None, None
    lo_all = np.empty((len(pt), n))
    hi_all = np.empty((len(pt), n))
    for i in range(n):
        m = np.ones(n, bool)
        m[i] = False
        c = choose(cands, p[m], X[m], t[m]) if len(cands) > 1 else cands[0]
        beta = ols_fit(design(c, p[m], X[m]), t[m])
        fit_i = (design(c, p[i:i + 1], X[i:i + 1]) @ beta)[0] + off_c[i]
        R = abs(y[i] - fit_i)
        mu_t = design(c, pt, Xt) @ beta + off_t
        lo_all[:, i] = mu_t - R
        hi_all[:, i] = mu_t + R
    lo = np.sort(lo_all, axis=1)[:, n - k]       # floor(alpha(n+1))-th smallest
    hi = np.sort(hi_all, axis=1)[:, k - 1]       # ceil((1-alpha)(n+1))-th smallest
    c_full = choose(cands, p, X, t) if len(cands) > 1 else cands[0]
    beta = ols_fit(design(c_full, p, X), t)
    point = design(c_full, pt, Xt) @ beta + off_t
    return lo, hi, point, c_full


def all_intervals(cal, test_inputs, q_source, mu_train):
    """Every method's (lower, upper, point, choice). Inputs only; no test labels."""
    out = {}
    pt = test_inputs["p"]
    out["source_split"] = (pt - q_source, pt + q_source, pt, None)
    q = split_q(np.abs(cal["y"] - cal["p"]))
    out["target_split"] = (pt - q, pt + q, pt, None)
    qm = split_q(np.abs(cal["y"] - mu_train))
    mu = np.full(len(pt), mu_train)
    out["mean_target_split"] = (mu - qm, mu + qm, mu, None)
    out["target_mean_jk"] = jackknife_plus(cal, test_inputs, ["mean"], "direct")
    out["target_linE_jk"] = jackknife_plus(cal, test_inputs, ["linE"], "direct")
    out["target_linPV_jk"] = jackknife_plus(cal, test_inputs, ["linPV"], "direct")
    out["target_auto_jk"] = jackknife_plus(cal, test_inputs, ["mean", "linE", "linPV"], "direct")
    out["ml_shift_jk"] = jackknife_plus(cal, test_inputs, ["shift"], "residual")
    out["ml_linear_jk"] = jackknife_plus(cal, test_inputs, ["linear"], "residual")
    out["ml_auto_jk"] = jackknife_plus(cal, test_inputs, ["shift", "linear"], "residual")
    return out


def metrics(y, lo, hi, point, alpha=ALPHA):
    """Winkler / Gneiting-Raftery interval score and its parts. Labels enter here only."""
    y = np.asarray(y, float)
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    # Correction 2026-11-02: every interval must be ordered (lower <= upper).
    if np.any(lo > hi + 1e-9 * np.maximum(1.0, np.abs(hi))):
        raise ValueError("Inverted interval: lower bound above upper bound")
    lower = (2 / alpha) * np.maximum(lo - y, 0)
    upper = (2 / alpha) * np.maximum(y - hi, 0)
    width = hi - lo
    # Inclusive bounds with a relative tolerance so float ties do not depend on platform.
    tol_lo = 1e-9 * np.maximum(1.0, np.abs(lo))
    tol_hi = 1e-9 * np.maximum(1.0, np.abs(hi))
    covered = (y >= lo - tol_lo) & (y <= hi + tol_hi)
    return dict(n_test=len(y), coverage=float(np.mean(covered)),
                mean_width=float(np.mean(width)), interval_score=float(np.mean(width + lower + upper)),
                lower_penalty=float(np.mean(lower)), upper_penalty=float(np.mean(upper)),
                MAE=float(np.mean(np.abs(y - point))), negative_lower_fraction=float(np.mean(lo < 0)))
