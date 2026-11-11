import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .skill import CLUTCH_COLUMNS, SKILL_COLUMNS, VARIANTS, add_columns, clutch_features, skill_features

SEED = 42
WEIGHT_LOGREG = 0.3
WEIGHT_HGB = 0.7
HGB_SEEDS = [42, 123]


def make_logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.003, max_iter=5000))


def make_hgb(seed):
    return HistGradientBoostingClassifier(learning_rate=0.015, max_iter=600, max_depth=5,
                                          min_samples_leaf=80, l2_regularization=10.0,
                                          max_features=0.4, random_state=seed)


def accuracy(probabilities, y):
    predictions = (probabilities > 0.5).astype(int)
    return float((predictions == y).mean())


def train_and_predict(X_train, y_train, X_new):
    logreg = make_logreg()
    logreg.fit(X_train, y_train)
    p_logreg = logreg.predict_proba(X_new)[:, 1]

    p_hgb = np.zeros(len(X_new))
    for seed in HGB_SEEDS:
        hgb = make_hgb(seed)
        hgb.fit(X_train, y_train)
        p_hgb = p_hgb + hgb.predict_proba(X_new)[:, 1] / len(HGB_SEEDS)

    return WEIGHT_LOGREG * p_logreg + WEIGHT_HGB * p_hgb


def add_player_features(X, y, teams, train_rows, X_new, new_rows, new_teams, variant, similar):
    skill_train, skill_new = skill_features(train_rows, new_rows, teams, new_teams, y, variant, similar)
    X_train = add_columns(X.iloc[train_rows], skill_train, SKILL_COLUMNS)
    X_new = add_columns(X_new, skill_new, SKILL_COLUMNS)
    if variant["clutch"]:
        clutch_train, clutch_new = clutch_features(X, train_rows, new_rows, teams, new_teams, y)
        X_train = add_columns(X_train, clutch_train, CLUTCH_COLUMNS)
        X_new = add_columns(X_new, clutch_new, CLUTCH_COLUMNS)
    return X_train, X_new


def cross_validate(X, y, teams, similar, n_splits=5):
    y = y.values
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = {}
    for name in VARIANTS:
        oof[name] = np.zeros(len(y))

    fold_number = 0
    for train_idx, valid_idx in folds.split(X, y):
        fold_number = fold_number + 1
        print("  fold", fold_number, "of", n_splits)
        for name in VARIANTS:
            X_train, X_valid = add_player_features(X, y, teams, train_idx, X.iloc[valid_idx], valid_idx,
                                                   teams, VARIANTS[name], similar)
            oof[name][valid_idx] = train_and_predict(X_train, y[train_idx], X_valid)

    scores = {}
    final = np.zeros(len(y))
    for name in VARIANTS:
        scores[name] = accuracy(oof[name], y)
        final = final + oof[name] / len(VARIANTS)
    scores["final"] = accuracy(final, y)
    return scores


def fit_and_predict(X, y, teams, X_test, test_teams, similar):
    y = y.values
    all_rows = np.arange(len(y))
    test_rows = np.arange(len(X_test))
    predictions = {}
    final = np.zeros(len(X_test))
    for name in VARIANTS:
        X_train, X_new = add_player_features(X, y, teams, all_rows, X_test, test_rows, test_teams,
                                             VARIANTS[name], similar)
        predictions[name] = train_and_predict(X_train, y, X_new)
        final = final + predictions[name] / len(VARIANTS)
    predictions["final"] = final
    return predictions
