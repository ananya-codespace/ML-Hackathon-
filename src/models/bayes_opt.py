"""Bayesian optimisation implemented from scratch (Gaussian-process surrogate + Expected Improvement).

Why hand-rolled?  It needs only scikit-learn/scipy (no Optuna/skopt install on the
demo laptops), and each step can be explained in the viva:

  1. Draw `n_init` space-filling points (scrambled Sobol) in the unit hypercube and evaluate them.
  2. Fit a GP (Matern-5/2 + white noise) to all (params -> grouped-CV macro F1) pairs so far.
  3. Pick the next point that maximises Expected Improvement over the best score seen,
     searched over many random candidates + perturbations of the current best.
  4. Evaluate it, append, repeat.  Every trial is written to disk immediately.

Parameters are mapped to [0,1]; log-scaled parameters are searched in log space.
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

warnings.filterwarnings("ignore", category=ConvergenceWarning)   # GP length-scale bound notices are harmless here


def _decode(space, u):
    out = {}
    for s, x in zip(space, u):
        lo, hi, t = s["low"], s["high"], s["type"]
        if t in ("logfloat", "logint"):
            v = float(np.exp(np.log(lo) + x * (np.log(hi) - np.log(lo))))
        else:
            v = lo + x * (hi - lo)
        out[s["name"]] = int(round(v)) if t in ("int", "logint") else float(v)
        if t in ("int", "logint"):
            out[s["name"]] = int(min(max(out[s["name"]], lo), hi))
    return out


def _expected_improvement(mu, sd, best, xi=0.01):
    sd = np.maximum(sd, 1e-12)
    z = (mu - best - xi) / sd
    return (mu - best - xi) * norm.cdf(z) + sd * norm.pdf(z)


class BayesOpt:
    def __init__(self, space, objective, seed=42, n_init=8, trials_path=None, maximize=True):
        self.space, self.objective, self.seed, self.n_init = space, objective, seed, n_init
        self.rng = np.random.RandomState(seed)
        self.d = len(space)
        self.trials_path = Path(trials_path) if trials_path else None
        self.trials = []                                   # dicts: trial, phase, params, value, u, seconds
        if self.trials_path and self.trials_path.exists():  # resume after a crash / laptop sleep
            self.trials = json.loads(self.trials_path.read_text())
            print(f"[bayes] resuming with {len(self.trials)} saved trials")

    def _suggest(self):
        U = np.array([t["u"] for t in self.trials]); y = np.array([t["value"] for t in self.trials])
        kernel = (ConstantKernel(1.0, (1e-3, 1e3)) *
                  Matern(length_scale=np.ones(self.d), length_scale_bounds=(1e-2, 1e2), nu=2.5) +
                  WhiteKernel(1e-4, (1e-8, 1e-1)))
        gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=5,
                                      random_state=int(self.rng.randint(1 << 30)))
        gp.fit(U, y)
        best_u = U[np.argmax(y)]
        cand = np.vstack([self.rng.rand(3000, self.d),
                          np.clip(best_u + 0.1 * self.rng.randn(1000, self.d), 0, 1)])
        mu, sd = gp.predict(cand, return_std=True)
        ei = _expected_improvement(mu, sd, y.max())
        return cand[int(np.argmax(ei))]

    def run(self, n_trials):
        sobol = qmc.Sobol(self.d, scramble=True, seed=self.seed).random(self.n_init)
        while len(self.trials) < n_trials:
            k = len(self.trials)
            phase = "sobol_init" if k < self.n_init else "gp_ei"
            u = sobol[k] if k < self.n_init else self._suggest()
            params = _decode(self.space, u)
            t0 = time.perf_counter()
            res = self.objective(params)
            value = float(res["value"] if isinstance(res, dict) else res)
            rec = dict(trial=k, phase=phase, params=params, value=value, u=[float(x) for x in u],
                       seconds=time.perf_counter() - t0,
                       extra=(res.get("extra", {}) if isinstance(res, dict) else {}))
            self.trials.append(rec)
            best = max(t["value"] for t in self.trials)
            print(f"[bayes] trial {k + 1:>3}/{n_trials} {phase:<10} value={value:.4f} best={best:.4f} params={params}")
            if self.trials_path:
                self.trials_path.parent.mkdir(parents=True, exist_ok=True)
                self.trials_path.write_text(json.dumps(self.trials, indent=1))
        return self.best()

    def best(self):
        return max(self.trials, key=lambda t: t["value"])
