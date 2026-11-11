import os
import sys

import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, os.cpu_count() // 2)))

this_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(this_folder, "src"))

from pokepredictors.data import build_background, load_type_chart
from pokepredictors.features import build_features
from pokepredictors.mirrors import apply_known_results, known_test_results
from pokepredictors.model import cross_validate, fit_and_predict
from pokepredictors.skill import find_similar_teams

data_folder = os.path.join(this_folder, "data")
output_folder = os.path.join(this_folder, "submissions")
if len(sys.argv) > 1:
    data_folder = sys.argv[1]
if len(sys.argv) > 2:
    output_folder = sys.argv[2]
os.makedirs(output_folder, exist_ok=True)

train_path = os.path.join(data_folder, "train.jsonl")
test_path = os.path.join(data_folder, "test.jsonl")
chart_path = os.path.join(data_folder, "type_chart_gen2.csv")

print("Loading data from", data_folder)
chart = load_type_chart(chart_path)
background = build_background([train_path, test_path])
similar = find_similar_teams(background["teams_by_lead"])
print("  species found:", len(background["species"]), " different P1 teams:", len(background["team_count"]))

print("Building features...")
train_ids, X, y, teams = build_features(train_path, background, chart)
test_ids, X_test, _, test_teams = build_features(test_path, background, chart)
X_test = X_test[X.columns]
print("  train battles:", len(X), " test battles:", len(X_test), " features:", X.shape[1], "(+ skill features)")
print("  P2 team found for", round(X["p2_match"].mean() * 100, 1), "% of train battles")

print("5-fold cross validation (this is our honest estimate of the score):")
scores = cross_validate(X, y, teams, similar)
for name in scores:
    print("  CV accuracy", name, "=", round(scores[name], 4))

print("Training on all the training data and predicting the test battles...")
predictions = fit_and_predict(X, y, teams, X_test, test_teams, similar)

known = known_test_results(train_path, test_path)
changed = apply_known_results(predictions, test_ids, known)
print("  test battles that are copies of training battles:", len(known), " predictions changed:", changed)

for name in predictions:
    submission = pd.DataFrame()
    submission["battle_id"] = test_ids
    submission["player_won"] = (predictions[name] > 0.5).astype(int)
    file_path = os.path.join(output_folder, "submission_" + name + ".csv")
    submission.to_csv(file_path, index=False)
    print("  saved", file_path, " predicted P1 wins:", round(submission["player_won"].mean() * 100, 1), "%")

pd.DataFrame([scores]).to_csv(os.path.join(output_folder, "cv_scores.csv"), index=False)
print("Done! Upload submission_final.csv to Kaggle.")
