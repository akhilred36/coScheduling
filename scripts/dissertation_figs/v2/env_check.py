"""Run first on agera: prints package versions, thread settings, and checks that every model family is deterministic.

  export CUDA_VISIBLE_DEVICES=""
  python scripts/dissertation_figs/v2/env_check.py

Exit code 0 = all checks passed. Paste the output back to the analyzer session.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import platform  # noqa: E402
import sys  # noqa: E402

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'OK' if cond else 'FAIL'}] {name} {detail}")


print("python", sys.version.split()[0], platform.platform())
print("os.cpu_count", os.cpu_count())
mods = {}
for m in ("numpy", "pandas", "sklearn", "scipy", "torch", "xgboost"):
    try:
        mods[m] = __import__(m)
        print(f"{m:8s} {mods[m].__version__}")
    except Exception as e:  # noqa: BLE001
        print(f"{m:8s} MISSING ({e})")
        if m != "scipy":
            ok = False

if "torch" in mods:
    torch = mods["torch"]
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    print("torch threads", torch.get_num_threads(), "deterministic", torch.are_deterministic_algorithms_enabled(),
          "cuda visible", torch.cuda.is_available())
    check("torch CUDA hidden", not torch.cuda.is_available())

    def fit(seed):
        torch.manual_seed(seed)
        net = torch.nn.Sequential(torch.nn.Linear(27, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-2)
        g = torch.Generator().manual_seed(123)
        X, y = torch.randn(100, 27, generator=g), torch.randn(100, generator=g)
        for _ in range(200):
            opt.zero_grad()
            loss = torch.nn.functional.huber_loss(net(X).squeeze(-1), y)
            loss.backward()
            opt.step()
        return net(X).detach().numpy()
    a, b = fit(0), fit(0)
    check("torch repeat-identical", (a == b).all())
    check("torch seed-sensitive", not (a == fit(1)).all())

if "numpy" in mods and "sklearn" in mods:
    np = mods["numpy"]
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(120, 45)), rng.normal(size=120)
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Lasso, Ridge
    from sklearn.svm import SVR
    for name, mk in [("ridge", lambda: Ridge(alpha=1.0)), ("lasso", lambda: Lasso(alpha=0.01, max_iter=100000)),
                     ("svr", lambda: SVR(C=1.0)),
                     ("rf", lambda: RandomForestRegressor(n_estimators=100, random_state=3, n_jobs=1))]:
        p1, p2 = mk().fit(X, y).predict(X), mk().fit(X, y).predict(X)
        check(f"{name} repeat-identical", (p1 == p2).all())
    if "xgboost" in mods:
        from xgboost import XGBRegressor

        def xg():
            return XGBRegressor(n_estimators=100, max_depth=3, subsample=0.8, colsample_bytree=0.8,
                                tree_method="hist", device="cpu", n_jobs=1, random_state=3).fit(X, y).predict(X)
        check("xgb repeat-identical", (xg() == xg()).all())

print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
