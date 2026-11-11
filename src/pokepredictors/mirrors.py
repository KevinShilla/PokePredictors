import hashlib
import json

from .data import read_battles


def battle_key(battle):
    timeline = battle["battle_timeline"]
    versions = []
    for first, second in (("p1", "p2"), ("p2", "p1")):
        turns = []
        for turn in timeline:
            turns.append((turn[first + "_pokemon_state"], turn[first + "_move_details"],
                          turn[second + "_pokemon_state"], turn[second + "_move_details"]))
        data = json.dumps(turns, sort_keys=True, separators=(",", ":")).encode("utf-8")
        versions.append(hashlib.blake2b(data, digest_size=16).digest())
    return min(versions), versions[0] <= versions[1]


def known_test_results(train_path, test_path):
    train_results = {}
    conflicts = set()
    for battle in read_battles(train_path):
        key, same_orientation = battle_key(battle)
        result = int(battle["player_won"])
        canonical_result = result if same_orientation else 1 - result
        if key in train_results and train_results[key] != canonical_result:
            conflicts.add(key)
        if key not in train_results:
            train_results[key] = canonical_result

    test_results = {}
    for battle in read_battles(test_path):
        key, same_orientation = battle_key(battle)
        if key in train_results and key not in conflicts:
            result = train_results[key]
            test_results[battle["battle_id"]] = result if same_orientation else 1 - result
    return test_results


def apply_known_results(predictions, test_ids, known):
    changed = 0
    for i in range(len(test_ids)):
        battle_id = test_ids[i]
        if battle_id in known:
            if int(predictions["final"][i] > 0.5) != known[battle_id]:
                changed = changed + 1
            for name in predictions:
                predictions[name][i] = known[battle_id]
    return changed
