# PokéPredictors: Pokémon Battle Prediction

FDS 2026 Kaggle challenge (Sapienza University of Rome), team **PokéPredictors**.

The task is to predict whether Player 1 wins a Gen 2 OU Pokémon Showdown battle, given Player 1's team, Player 2's lead and the first 30 turns of the battle.

**Selected submission:** `submissions.csv`, Kaggle accuracy **0.69000 public** and **0.69457 private**.

## Run it

```bash
pip install -r requirements.txt
python train.py DATA_FOLDER OUTPUT_FOLDER
```

`DATA_FOLDER` holds the competition files (`train.jsonl`, `test.jsonl`, `type_chart_gen2.csv`) and defaults to `./data`. The data is not included in this repository.
The run takes about 5 minutes, prints the cross validation accuracy and writes `submission_final.csv`, which is identical to `submissions.csv`.

On Kaggle, `PokePredictors_notebook.ipynb` clones this repository and runs the same pipeline step by step.

## Method

1. **Features (589 per battle):** the state at turn 30 (HP left, fainted Pokémon, status conditions, boosts), momentum at turns 5 to 25, move usage, type matchups between the surviving Pokémon, which common species are still alive, and Player 2's hidden team, matched to teams seen in full as Player 1 in other battles.
2. **Player skill:** the same players reuse the same teams, so each team's win rate over its other battles is added as a feature. It is computed out of fold, so no battle sees its own result. Three variants: similar teams, all possible Player 2 teams, and clutch skill.
3. **Models:** for each variant, Logistic Regression (weight 0.3) blended with HistGradientBoosting (weight 0.7, two seeds). The final prediction averages the three variants.
4. **Validation:** five fold stratified cross validation on the training battles.

Only classical machine learning and the competition data are used.

## Files

| Path | Contents |
| --- | --- |
| `train.py` | Runs the whole pipeline: features, cross validation, final training, submission |
| `src/pokepredictors/data.py` | Reading the data, species table, known teams |
| `src/pokepredictors/features.py` | Feature engineering |
| `src/pokepredictors/skill.py` | Player skill features |
| `src/pokepredictors/model.py` | Models, cross validation, ensemble |
| `src/pokepredictors/mirrors.py` | Test battles that are exact copies of training battles |
| `PokePredictors_notebook.ipynb` | Kaggle notebook that clones this repository |
| `submissions.csv` | Our selected Kaggle submission |
| `report/PokePredictors_report.pdf` | One page report |
