import json

import pandas as pd

STATS = ["base_hp", "base_atk", "base_def", "base_spa", "base_spd", "base_spe"]
N_TOP_SPECIES = 60


def read_battles(path):
    with open(path, encoding="utf-8") as file:
        for line in file:
            if line.strip() != "":
                yield json.loads(line)


def load_type_chart(path):
    df = pd.read_csv(path)
    chart = {}
    for i in range(len(df)):
        attacker = df["attacking_type"][i]
        defender = df["defending_type"][i]
        chart[(attacker, defender)] = df["multiplier"][i]
    return chart


def team_key(names):
    return "|".join(sorted(names))


def build_background(paths):
    species = {}
    species_count = {}
    team_count = {}
    teams_by_lead = {}

    for path in paths:
        for battle in read_battles(path):
            pokemons = battle["p1_team_details"] + [battle["p2_lead_details"]]
            for p in pokemons:
                if p["name"] in species:
                    continue
                types = []
                for t in p["types"]:
                    if t != "notype":
                        types.append(t)
                info = {"types": types}
                for s in STATS:
                    info[s] = p[s]
                species[p["name"]] = info

            used = []
            for p in battle["p1_team_details"]:
                used.append(p["name"])
            p2_seen = set()
            for turn in battle["battle_timeline"]:
                p2_seen.add(turn["p2_pokemon_state"]["name"])
            used = used + sorted(p2_seen)
            for name in used:
                species_count[name] = species_count.get(name, 0) + 1

            names = []
            for p in battle["p1_team_details"]:
                names.append(p["name"])
            key = team_key(names)
            team_count[key] = team_count.get(key, 0) + 1
            lead = names[0]
            if lead not in teams_by_lead:
                teams_by_lead[lead] = []
            if key not in teams_by_lead[lead]:
                teams_by_lead[lead].append(key)

    all_names = sorted(species_count.keys(), key=lambda name: (-species_count[name], name))
    top_species = all_names[:N_TOP_SPECIES]

    return {"species": species, "top_species": top_species,
            "team_count": team_count, "teams_by_lead": teams_by_lead}
