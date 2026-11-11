import numpy as np
import pandas as pd

from .data import STATS, read_battles, team_key

STATUSES = ["slp", "par", "tox", "psn", "frz", "brn"]
EFFECTS = ["substitute", "leechseed", "confusion", "trapped", "perish", "protect", "encore", "attract"]
BOOSTS = ["atk", "def", "spa", "spd", "spe"]
CHECKPOINTS = [5, 10, 15, 20, 25]

TRACKED_MOVES = {
    "spikes": ["spikes"],
    "rapidspin": ["rapidspin"],
    "sleep": ["lovelykiss", "sleeppowder", "spore", "hypnosis", "sing"],
    "para": ["thunderwave", "stunspore", "glare"],
    "toxic": ["toxic"],
    "heal": ["rest", "recover", "softboiled", "milkdrink", "moonlight", "morningsun", "synthesis"],
    "phaze": ["roar", "whirlwind"],
    "boom": ["explosion", "selfdestruct"],
    "setup": ["curse", "swordsdance", "growth", "agility", "amnesia", "bellydrum", "meditate", "sharpen"],
    "sleeptalk": ["sleeptalk"],
    "batonpass": ["batonpass"],
    "substitute": ["substitute"],
    "protect": ["protect", "detect"],
}


def get_effectiveness(move_type, defender_types, chart):
    multiplier = 1.0
    for t in defender_types:
        if (move_type, t) in chart:
            multiplier = multiplier * chart[(move_type, t)]
    return multiplier


def get_stat(species, name, stat, default):
    if name in species:
        return species[name][stat]
    return default


def get_types(species, name):
    if name in species:
        return species[name]["types"]
    return []


def replay_side(timeline, side, team_names):
    hp = {}
    status = {}
    for name in team_names:
        hp[name] = 1.0
        status[name] = "nostatus"

    damage_taken = 0.0
    healed = 0.0
    hp_totals = []
    faint_counts = []

    for turn in timeline:
        state = turn[side + "_pokemon_state"]
        name = state["name"]

        old_hp = 1.0
        if name in hp:
            old_hp = hp[name]
        change = state["hp_pct"] - old_hp
        if change < 0:
            damage_taken = damage_taken + (-change)
        else:
            healed = healed + change

        hp[name] = state["hp_pct"]
        status[name] = state["status"]

        fainted = 0
        for mon in status:
            if status[mon] == "fnt" or hp[mon] <= 0:
                fainted = fainted + 1
        unseen = 6 - len(hp)
        hp_totals.append(sum(hp.values()) + unseen * 1.0)
        faint_counts.append(fainted)

    return {"hp": hp, "status": status, "hp_totals": hp_totals, "faint_counts": faint_counts,
            "damage_taken": damage_taken, "healed": healed}


def add_state_features(f, side, info, timeline, species):
    hp = info["hp"]
    status = info["status"]

    alive = []
    for name in hp:
        if status[name] != "fnt" and hp[name] > 0:
            alive.append(name)

    seen_names = set()
    for turn in timeline:
        seen_names.add(turn[side + "_pokemon_state"]["name"])

    f[side + "_hp_total"] = info["hp_totals"][-1]
    f[side + "_fainted"] = info["faint_counts"][-1]
    f[side + "_alive_known"] = len(alive)
    f[side + "_n_seen"] = len(seen_names)

    alive_hps = []
    low_hp = 0
    for name in alive:
        alive_hps.append(hp[name])
        if hp[name] < 0.25:
            low_hp = low_hp + 1
    if len(alive) > 0:
        f[side + "_mean_alive_hp"] = np.mean(alive_hps)
    else:
        f[side + "_mean_alive_hp"] = 0.0
    f[side + "_n_low_hp"] = low_hp

    for s in STATUSES:
        count = 0
        for name in alive:
            if status[name] == s:
                count = count + 1
        f[side + "_status_" + s] = count
    statused = 0
    healthy = 0
    for name in alive:
        if status[name] in STATUSES:
            statused = statused + 1
        if status[name] == "nostatus" and hp[name] > 0.5:
            healthy = healthy + 1
    f[side + "_statused"] = statused
    f[side + "_healthy"] = healthy

    f[side + "_dmg_taken"] = info["damage_taken"]
    f[side + "_healed"] = info["healed"]

    bulk = 0
    offense = 0
    speeds = []
    for name in alive:
        bulk = bulk + hp[name] * get_stat(species, name, "base_hp", 80)
        attack = get_stat(species, name, "base_atk", 80)
        special = get_stat(species, name, "base_spa", 80)
        offense = offense + max(attack, special)
        speeds.append(get_stat(species, name, "base_spe", 70))
    f[side + "_bulk_left"] = bulk
    f[side + "_offense_left"] = offense
    if len(alive) > 0:
        f[side + "_speed_left"] = np.mean(speeds)
    else:
        f[side + "_speed_left"] = 0.0

    last = timeline[-1][side + "_pokemon_state"]
    f[side + "_active_hp"] = last["hp_pct"]
    if last["status"] in STATUSES:
        f[side + "_active_status"] = STATUSES.index(last["status"]) + 1
    elif last["status"] == "fnt":
        f[side + "_active_status"] = -1
    else:
        f[side + "_active_status"] = 0
    f[side + "_active_boost_sum"] = sum(last["boosts"].values())
    for b in BOOSTS:
        f[side + "_active_boost_" + b] = last["boosts"][b]
    for e in EFFECTS:
        if e in last["effects"]:
            f[side + "_active_" + e] = 1
        else:
            f[side + "_active_" + e] = 0
    f[side + "_active_spe"] = get_stat(species, last["name"], "base_spe", 70)
    total = 0
    for s in STATS:
        total = total + get_stat(species, last["name"], s, 80)
    f[side + "_active_bst"] = total

    for c in CHECKPOINTS:
        f[side + "_hp_total_t" + str(c)] = info["hp_totals"][c - 1]
        f[side + "_fainted_t" + str(c)] = info["faint_counts"][c - 1]


def add_move_features(f, side, opponent, timeline, species, chart):
    n_moves = 0
    n_no_move = 0
    n_status = 0
    n_physical = 0
    n_special = 0
    n_priority = 0
    bp_sum = 0.0
    super_effective = 0
    resisted = 0
    immune = 0
    stab = 0
    tracked = {}
    for key in TRACKED_MOVES:
        tracked[key] = 0

    for turn in timeline:
        move = turn[side + "_move_details"]
        if move is None:
            n_no_move = n_no_move + 1
            continue
        n_moves = n_moves + 1

        if move["category"] == "status":
            n_status = n_status + 1
        if move["category"] == "physical":
            n_physical = n_physical + 1
        if move["category"] == "special":
            n_special = n_special + 1
        if move["priority"] > 0:
            n_priority = n_priority + 1

        power = move["base_power"]
        if power is None:
            power = 0
        bp_sum = bp_sum + power

        for key in TRACKED_MOVES:
            if move["name"] in TRACKED_MOVES[key]:
                tracked[key] = tracked[key] + 1

        if move["category"] != "status" and power > 0:
            user = turn[side + "_pokemon_state"]["name"]
            if move["type"] in get_types(species, user):
                stab = stab + 1
            target = turn[opponent + "_pokemon_state"]["name"]
            eff = get_effectiveness(move["type"], get_types(species, target), chart)
            if eff > 1:
                super_effective = super_effective + 1
            if eff > 0 and eff < 1:
                resisted = resisted + 1
            if eff == 0:
                immune = immune + 1

    f[side + "_n_moves"] = n_moves
    f[side + "_n_no_move"] = n_no_move
    f[side + "_n_status_moves"] = n_status
    f[side + "_n_physical"] = n_physical
    f[side + "_n_special"] = n_special
    f[side + "_n_priority"] = n_priority
    f[side + "_bp_sum"] = bp_sum
    f[side + "_n_super_eff"] = super_effective
    f[side + "_n_resisted"] = resisted
    f[side + "_n_immune"] = immune
    f[side + "_n_stab"] = stab
    for key in TRACKED_MOVES:
        f[side + "_mv_" + key] = tracked[key]


def add_differences(f):
    keys = list(f.keys())
    for key in keys:
        if key.startswith("p1_"):
            rest = key[3:]
            if ("p2_" + rest) in f and ("diff_" + rest) not in f:
                f["diff_" + rest] = f[key] - f["p2_" + rest]


def get_alive(info):
    alive = []
    for name in info["hp"]:
        if info["status"][name] != "fnt" and info["hp"][name] > 0:
            alive.append(name)
    return alive


def get_bst(species, name):
    if name not in species:
        return 500
    total = 0
    for s in STATS:
        total = total + species[name][s]
    return total


def add_species_features(f, p1_info, p2_info, top_species):
    p1_alive = get_alive(p1_info)
    p2_alive = get_alive(p2_info)
    for name in top_species:
        if name in p1_alive:
            f["p1sp_" + name] = p1_info["hp"][name]
        else:
            f["p1sp_" + name] = 0.0
        if name in p1_info["hp"]:
            f["p1has_" + name] = 1
        else:
            f["p1has_" + name] = 0
        if name in p2_alive:
            f["p2sp_" + name] = p2_info["hp"][name]
        else:
            f["p2sp_" + name] = 0.0
        if name in p2_info["hp"]:
            f["p2seen_" + name] = 1
        else:
            f["p2seen_" + name] = 0


def add_misc_features(f, battle, p1_info, species, chart):
    timeline = battle["battle_timeline"]
    last = timeline[-1]
    before_last = timeline[-2]

    p1_types = get_types(species, last["p1_pokemon_state"]["name"])
    p2_types = get_types(species, last["p2_pokemon_state"]["name"])
    best1 = 1.0
    if len(p1_types) > 0:
        best1 = 0
        for t in p1_types:
            best1 = max(best1, get_effectiveness(t, p2_types, chart))
    best2 = 1.0
    if len(p2_types) > 0:
        best2 = 0
        for t in p2_types:
            best2 = max(best2, get_effectiveness(t, p1_types, chart))
    f["act_p1_stab_eff"] = best1
    f["act_p2_stab_eff"] = best2
    f["act_eff_diff"] = best1 - best2

    for side in ["p1", "p2"]:
        power = 0
        for turn in timeline[-5:]:
            move = turn[side + "_move_details"]
            if move is not None and move["base_power"]:
                power = power + move["base_power"]
        f[side + "_bp_last5"] = power
    f["diff_bp_last5"] = f["p1_bp_last5"] - f["p2_bp_last5"]

    for side in ["p1", "p2"]:
        last_status = {}
        new_status = 0
        for turn in timeline:
            state = turn[side + "_pokemon_state"]
            if state["status"] in STATUSES and last_status.get(state["name"]) != state["status"]:
                new_status = new_status + 1
            last_status[state["name"]] = state["status"]
        f[side + "_new_status"] = new_status
    f["diff_new_status"] = f["p1_new_status"] - f["p2_new_status"]

    p1_alive = get_alive(p1_info)
    leftovers = 0
    for p in battle["p1_team_details"]:
        if p.get("item") == "leftovers" and p["name"] in p1_alive:
            leftovers = leftovers + 1
    f["p1_alive_leftovers"] = leftovers

    for side in ["p1", "p2"]:
        now = last[side + "_pokemon_state"]
        before = before_last[side + "_pokemon_state"]
        if now["name"] == before["name"]:
            f["last_turn_" + side + "_hp_change"] = now["hp_pct"] - before["hp_pct"]
        else:
            f["last_turn_" + side + "_hp_change"] = 0


def add_matchup_features(f, timeline, p1_info, p2_info, species, chart):
    p1_alive = get_alive(p1_info)
    p2_alive = get_alive(p2_info)

    move_types = {"p1": {}, "p2": {}}
    for turn in timeline:
        for side in ["p1", "p2"]:
            move = turn[side + "_move_details"]
            if move is not None and move["category"] != "status" and move["base_power"]:
                name = turn[side + "_pokemon_state"]["name"]
                if name not in move_types[side]:
                    move_types[side][name] = set()
                move_types[side][name].add(move["type"])

    def best_hit(attacker, side, defender):
        types = set(get_types(species, attacker))
        if attacker in move_types[side]:
            types = types | move_types[side][attacker]
        best = 1.0
        if len(types) > 0:
            best = 0
            for t in types:
                best = max(best, get_effectiveness(t, get_types(species, defender), chart))
        return best

    advantages = []
    weighted = 0
    for a in p1_alive:
        for b in p2_alive:
            adv = best_hit(a, "p1", b) - best_hit(b, "p2", a)
            advantages.append(adv)
            weighted = weighted + adv * p1_info["hp"][a] * p2_info["hp"][b]
    if len(advantages) > 0:
        f["mu_adv_mean"] = np.mean(advantages)
    else:
        f["mu_adv_mean"] = 0
    f["mu_adv_wsum"] = weighted

    count = 0
    for b in p2_alive:
        answered = False
        for a in p1_alive:
            if best_hit(a, "p1", b) > 1:
                answered = True
        if not answered:
            count = count + 1
    f["p2_alive_unanswered"] = count
    count = 0
    for a in p1_alive:
        answered = False
        for b in p2_alive:
            if best_hit(b, "p2", a) > 1:
                answered = True
        if not answered:
            count = count + 1
    f["p1_alive_unanswered"] = count

    p1_frozen = 0
    for a in p1_alive:
        if p1_info["status"][a] == "frz":
            p1_frozen = p1_frozen + 1
    p2_frozen = 0
    for b in p2_alive:
        if p2_info["status"][b] == "frz":
            p2_frozen = p2_frozen + 1
    p2_unseen = 6 - len(p2_info["hp"])
    f["p1_frozen"] = p1_frozen
    f["p2_frozen"] = p2_frozen
    f["p1_eff_alive"] = len(p1_alive) - p1_frozen
    f["p2_eff_alive"] = len(p2_alive) - p2_frozen + p2_unseen
    f["diff_eff_alive"] = f["p1_eff_alive"] - f["p2_eff_alive"]

    p1_strength = 0
    for a in p1_alive:
        if p1_info["status"][a] != "frz":
            p1_strength = p1_strength + p1_info["hp"][a] * get_bst(species, a)
    p2_strength = 0
    for b in p2_alive:
        if p2_info["status"][b] != "frz":
            p2_strength = p2_strength + p2_info["hp"][b] * get_bst(species, b)
    f["p1_strength"] = p1_strength
    f["p2_strength_seen"] = p2_strength
    f["p2_strength"] = p2_strength + p2_unseen * 500
    f["diff_strength"] = f["p1_strength"] - f["p2_strength"]

    weights = {"par": 0.75, "slp": 0.6, "tox": 0.75, "psn": 0.85, "brn": 0.8, "frz": 0.1, "nostatus": 1.0}
    p1_adj = 0
    for a in p1_alive:
        p1_adj = p1_adj + p1_info["hp"][a] * weights.get(p1_info["status"][a], 1)
    p2_adj = 0
    for b in p2_alive:
        p2_adj = p2_adj + p2_info["hp"][b] * weights.get(p2_info["status"][b], 1)
    f["p1_adj_hp"] = p1_adj
    f["p2_adj_hp"] = p2_adj + p2_unseen
    f["diff_adj_hp"] = f["p1_adj_hp"] - f["p2_adj_hp"]


def find_p2_team(battle, background):
    seen = set()
    seen.add(battle["p2_lead_details"]["name"])
    for turn in battle["battle_timeline"]:
        seen.add(turn["p2_pokemon_state"]["name"])

    lead = battle["p2_lead_details"]["name"]
    candidates = []
    if lead in background["teams_by_lead"]:
        for key in background["teams_by_lead"][lead]:
            team = set(key.split("|"))
            if seen.issubset(team):
                candidates.append(key)

    if len(candidates) == 0:
        return "", candidates, seen
    best = candidates[0]
    for key in candidates:
        if background["team_count"][key] > background["team_count"][best]:
            best = key
    return best, candidates, seen


def add_p2_team_features(f, battle, background, p2_team, n_candidates, seen):
    species = background["species"]
    if p2_team != "":
        f["p2_match"] = 1
    else:
        f["p2_match"] = 0
    f["p2_n_cands"] = n_candidates

    unseen = []
    if p2_team != "":
        for name in p2_team.split("|"):
            if name not in seen:
                unseen.append(name)
    f["p2_pred_unseen_n"] = len(unseen)
    total = 0
    for name in unseen:
        total = total + get_bst(species, name)
    f["p2_pred_unseen_bst"] = total
    for name in background["top_species"]:
        if name in unseen:
            f["p2pred_" + name] = 1
        else:
            f["p2pred_" + name] = 0

    p1_names = []
    for p in battle["p1_team_details"]:
        p1_names.append(p["name"])
    f["p1_team_count"] = background["team_count"][team_key(p1_names)]
    if p2_team != "":
        f["p2_team_count"] = background["team_count"][p2_team]
    else:
        f["p2_team_count"] = 0


def battle_features(battle, background, chart):
    species = background["species"]
    f = {}
    team = battle["p1_team_details"]
    lead2 = battle["p2_lead_details"]
    timeline = battle["battle_timeline"]

    team_names = []
    for p in team:
        team_names.append(p["name"])

    for s in STATS:
        values = []
        for p in team:
            values.append(p[s])
        f["p1_team_mean_" + s] = np.mean(values)
        f["p1_lead_" + s] = team[0][s]
        f["p2_lead_" + s] = lead2[s]
        f["lead_diff_" + s] = team[0][s] - lead2[s]

    total = 0
    leftovers = 0
    for p in team:
        for s in STATS:
            total = total + p[s]
        if p.get("item") == "leftovers":
            leftovers = leftovers + 1
    f["p1_team_total_stats"] = total
    f["p1_n_leftovers"] = leftovers

    types1 = []
    for t in team[0]["types"]:
        if t != "notype":
            types1.append(t)
    types2 = []
    for t in lead2["types"]:
        if t != "notype":
            types2.append(t)
    best1 = 0
    for t in types1:
        best1 = max(best1, get_effectiveness(t, types2, chart))
    best2 = 0
    for t in types2:
        best2 = max(best2, get_effectiveness(t, types1, chart))
    f["lead_p1_best_eff"] = best1
    f["lead_p2_best_eff"] = best2
    f["lead_eff_diff"] = best1 - best2

    p1_info = replay_side(timeline, "p1", team_names)
    p2_info = replay_side(timeline, "p2", [])
    add_state_features(f, "p1", p1_info, timeline, species)
    add_state_features(f, "p2", p2_info, timeline, species)

    f["p2_unseen"] = 6 - f["p2_n_seen"]
    add_differences(f)

    if f["p1_active_spe"] > f["p2_active_spe"]:
        f["active_speed_adv"] = 1
    else:
        f["active_speed_adv"] = 0
    f["hp_trend_last10"] = f["diff_hp_total"] - f["diff_hp_total_t20"]
    f["hp_trend_last5"] = f["diff_hp_total"] - f["diff_hp_total_t25"]

    add_move_features(f, "p1", "p2", timeline, species, chart)
    add_move_features(f, "p2", "p1", timeline, species, chart)
    add_differences(f)

    p1_disabled = 0
    p2_disabled = 0
    for turn in timeline:
        if turn["p1_pokemon_state"]["status"] in ["slp", "frz"]:
            p1_disabled = p1_disabled + 1
        if turn["p2_pokemon_state"]["status"] in ["slp", "frz"]:
            p2_disabled = p2_disabled + 1
    f["p1_turns_disabled"] = p1_disabled
    f["p2_turns_disabled"] = p2_disabled
    f["diff_turns_disabled"] = p1_disabled - p2_disabled

    add_species_features(f, p1_info, p2_info, background["top_species"])
    add_misc_features(f, battle, p1_info, species, chart)
    add_matchup_features(f, timeline, p1_info, p2_info, species, chart)
    p2_team, candidates, seen = find_p2_team(battle, background)
    add_p2_team_features(f, battle, background, p2_team, len(candidates), seen)

    teams = {"p1_team": team_key(team_names), "p2_team": p2_team,
             "p2_candidates": ";".join(candidates)}
    if len(candidates) == 1:
        teams["p2_team_sure"] = p2_team
    else:
        teams["p2_team_sure"] = ""
    return f, teams


def build_features(path, background, chart):
    rows = []
    ids = []
    labels = []
    team_rows = []
    for battle in read_battles(path):
        f, teams = battle_features(battle, background, chart)
        rows.append(f)
        team_rows.append(teams)
        ids.append(battle["battle_id"])
        if "player_won" in battle:
            labels.append(int(battle["player_won"]))

    X = pd.DataFrame(rows).astype(float)
    ids = pd.Series(ids, name="battle_id")
    teams = pd.DataFrame(team_rows)
    if len(labels) > 0:
        y = pd.Series(labels, name="player_won")
    else:
        y = None
    return ids, X, y, teams
