import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SMOOTHING = 5
SKILL_COLUMNS = ["skill_p1", "skill_p1_games", "skill_p2", "skill_p2_games", "skill_diff"]
CLUTCH_COLUMNS = ["clutch_p1", "clutch_p2", "clutch_diff"]

VARIANTS = {
    "similar": {"similar": True, "all_p2": False, "clutch": False},
    "all_p2": {"similar": False, "all_p2": True, "clutch": False},
    "clutch": {"similar": False, "all_p2": False, "clutch": True},
}


def find_similar_teams(teams_by_lead):
    similar = {}
    for lead in teams_by_lead:
        keys = teams_by_lead[lead]
        for key in keys:
            pokemons = set(key.split("|"))
            similar[key] = []
            for other in keys:
                if other != key and len(pokemons & set(other.split("|"))) >= 5:
                    similar[key].append(other)
    return similar


def make_skill_table(rows, teams, y, variant):
    wins = {}
    games = {}
    for i in rows:
        p1 = teams["p1_team"][i]
        wins[p1] = wins.get(p1, 0) + y[i]
        games[p1] = games.get(p1, 0) + 1

        if variant["all_p2"]:
            if teams["p2_candidates"][i] != "":
                candidates = teams["p2_candidates"][i].split(";")
                share = 1 / len(candidates)
                for p2 in candidates:
                    wins[p2] = wins.get(p2, 0) + (1 - y[i]) * share
                    games[p2] = games.get(p2, 0) + share
        else:
            p2 = teams["p2_team_sure"][i]
            if p2 != "":
                wins[p2] = wins.get(p2, 0) + (1 - y[i])
                games[p2] = games.get(p2, 0) + 1
    return wins, games


def win_rate(team, wins, games, variant, similar):
    w = wins.get(team, 0)
    g = games.get(team, 0)
    if variant["similar"]:
        for other in similar.get(team, []):
            w = w + 0.5 * wins.get(other, 0)
            g = g + 0.5 * games.get(other, 0)
    return (w + 0.5 * SMOOTHING) / (g + SMOOTHING), g


def apply_skill_table(wins, games, rows, teams, variant, similar):
    result = np.zeros((len(rows), len(SKILL_COLUMNS)))
    for j in range(len(rows)):
        i = rows[j]
        rate1, games1 = win_rate(teams["p1_team"][i], wins, games, variant, similar)

        if variant["all_p2"] and teams["p2_candidates"][i] != "":
            rates = []
            game_counts = []
            for p2 in teams["p2_candidates"][i].split(";"):
                rate, g = win_rate(p2, wins, games, variant, similar)
                rates.append(rate)
                game_counts.append(g)
            rate2 = np.mean(rates)
            games2 = np.mean(game_counts)
        elif teams["p2_team"][i] != "":
            rate2, games2 = win_rate(teams["p2_team"][i], wins, games, variant, similar)
        else:
            rate2, games2 = 0.5, 0
        result[j] = [rate1, games1, rate2, games2, rate1 - rate2]
    return result


def skill_features(train_rows, new_rows, teams, new_teams, y, variant, similar):
    train_result = np.zeros((len(train_rows), len(SKILL_COLUMNS)))
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    for fit_part, apply_part in folds.split(train_rows, y[train_rows]):
        wins, games = make_skill_table(train_rows[fit_part], teams, y, variant)
        train_result[apply_part] = apply_skill_table(wins, games, train_rows[apply_part], teams, variant, similar)

    wins, games = make_skill_table(train_rows, teams, y, variant)
    new_result = apply_skill_table(wins, games, new_rows, new_teams, variant, similar)
    return train_result, new_result


def expected_results(X, y, rows):
    expected = np.zeros(len(rows))
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)
    for fit_part, apply_part in folds.split(rows, y[rows]):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.003, max_iter=3000))
        model.fit(X.iloc[rows[fit_part]], y[rows[fit_part]])
        expected[apply_part] = model.predict_proba(X.iloc[rows[apply_part]])[:, 1]
    return expected


def make_clutch_table(rows, surprise, teams):
    total = {}
    games = {}
    for j in range(len(rows)):
        i = rows[j]
        p1 = teams["p1_team"][i]
        total[p1] = total.get(p1, 0) + surprise[j]
        games[p1] = games.get(p1, 0) + 1
        p2 = teams["p2_team_sure"][i]
        if p2 != "":
            total[p2] = total.get(p2, 0) - surprise[j]
            games[p2] = games.get(p2, 0) + 1
    return total, games


def apply_clutch_table(total, games, rows, teams):
    result = np.zeros((len(rows), len(CLUTCH_COLUMNS)))
    for j in range(len(rows)):
        i = rows[j]
        p1 = teams["p1_team"][i]
        clutch1 = total.get(p1, 0) / (games.get(p1, 0) + SMOOTHING)
        p2 = teams["p2_team"][i]
        clutch2 = 0.0
        if p2 != "":
            clutch2 = total.get(p2, 0) / (games.get(p2, 0) + SMOOTHING)
        result[j] = [clutch1, clutch2, clutch1 - clutch2]
    return result


def clutch_features(X, train_rows, new_rows, teams, new_teams, y):
    surprise = y[train_rows] - expected_results(X, y, train_rows)

    train_result = np.zeros((len(train_rows), len(CLUTCH_COLUMNS)))
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    for fit_part, apply_part in folds.split(train_rows, y[train_rows]):
        total, games = make_clutch_table(train_rows[fit_part], surprise[fit_part], teams)
        train_result[apply_part] = apply_clutch_table(total, games, train_rows[apply_part], teams)

    total, games = make_clutch_table(train_rows, surprise, teams)
    new_result = apply_clutch_table(total, games, new_rows, new_teams)
    return train_result, new_result


def add_columns(X, values, names):
    X = X.copy()
    for c in range(len(names)):
        X[names[c]] = values[:, c]
    return X
