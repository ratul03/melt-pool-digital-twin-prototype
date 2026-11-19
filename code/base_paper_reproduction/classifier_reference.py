"""Public-data path previously missed in Thesis B: notebook cell 114.
The classifier uses labeled REGRESSION rows and measured-width absorptivity.
This is a small diagnostic benchmark, not a pre-build classifier.
"""

import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score
from threadpoolctl import threadpool_limits
from reproduce import ROOT, BASE, LABELS


def main():
    feats = BASE + ["beam D", "Hatch spacing", "absorption coefficient 2", "absorption coefficient"]
    d = pd.read_csv(ROOT / "data/raw/meltpoolnet_regression.csv")
    d["row_id"] = d.index
    d = d[d["meltpool shape"].isin(LABELS)].copy()
    d[feats] = d[feats].apply(pd.to_numeric, errors="coerce")
    d = d.dropna(subset=feats)
    classes, y = np.unique(d["meltpool shape"], return_inverse=True)
    X = d[feats].to_numpy()
    ids = d.row_id.to_numpy()
    out = ROOT / "results/authors_reference"
    out.mkdir(exist_ok=True)
    mdir = ROOT / "models/authors_reference"
    mdir.mkdir(exist_ok=True)
    rows = []
    preds = []
    saved = []
    for seed in range(5):
        np.random.seed(seed)
        # Notebook reuses the estimator object and its global random state across folds.
        m = RandomForestClassifier(n_estimators=160)
        Xs, ys, idx = shuffle(X, y, ids, random_state=0)
        for fold, (tr, te) in enumerate(KFold(n_splits=5).split(X)):
            mu = Xs[tr].mean(axis=0)
            sd = Xs[tr].std(axis=0) + 1e-10
            m.fit((Xs[tr] - mu) / sd, ys[tr])
            p = m.predict((Xs[te] - mu) / sd)
            score = accuracy_score(ys[te], p)
            rows.append(
                dict(
                    seed=seed,
                    fold=fold,
                    n_train=len(tr),
                    n_test=len(te),
                    accuracy=score,
                    macro_f1=f1_score(
                        ys[te], p, average="macro", labels=np.arange(len(classes)), zero_division=0
                    ),
                )
            )
            for j, i in enumerate(te):
                preds.append(
                    dict(
                        seed=seed,
                        fold=fold,
                        row_id=int(idx[i]),
                        actual=classes[ys[i]],
                        predicted=classes[p[j]],
                    )
                )
            if seed == 0 and fold == 0:
                artifact = dict(
                    model=m,
                    features=feats,
                    mean=mu,
                    scale=sd,
                    classes=classes,
                    train_row_ids=idx[tr].tolist(),
                    test_row_ids=idx[te].tolist(),
                    global_rng_seed=seed,
                    fold=fold,
                    usage="post_build_diagnostic",
                    dataset="regression_labeled_rows",
                    source="upstream notebook cell 114",
                    physical_guarantee=False,
                )
                path = mdir / "RFC_both_absorptivities_hatch.joblib"
                joblib.dump(artifact, path, compress=3)
                a = joblib.load(path)
                assert np.array_equal(a["model"].predict((Xs[te] - a["mean"]) / a["scale"]), p)
                saved.append(
                    dict(
                        model=str(path.relative_to(ROOT)),
                        reload_pass=True,
                        n_train=len(tr),
                        n_test=len(te),
                    )
                )
    df = pd.DataFrame(rows)
    df.to_csv(out / "fold_scores.csv", index=False)
    pd.DataFrame(preds).to_csv(out / "predictions.csv", index=False)
    means = df.groupby("seed").accuracy.mean()
    upstream = json.loads((ROOT / "audit/upstream_execution.json").read_text())
    expected = next(
        x["accuracy_per_rng_seed"]
        for x in upstream
        if x["config"] == "both_abs+hatch_from_regression"
    )
    assert np.allclose(
        means, expected, rtol=0, atol=1e-12
    ), "Independent implementation differs from actual notebook function"
    summary = dict(
        n=len(d),
        class_counts=d["meltpool shape"].value_counts().to_dict(),
        accuracy_mean=float(means.mean()),
        accuracy_rng_std=float(means.std(ddof=1)),
        per_rng_seed=means.tolist(),
        matches_actual_notebook_function=True,
        model_reload=saved,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
