"""
Builds a synthetic (sentence -> parameter vector) dataset to train the
text-to-parameter model on.

Why synthetic: no dataset exists that pairs "city description text" with
these exact 10 generator parameters. So, the same trick as the procedural
generator itself: we write rules that produce (sentence, parameters) pairs
and sample them thousands of times.

Key idea -- PARTIAL mentions, not every combination:
    A real prompt like "a seaside city with a dense center and lots of green
    space" describes ~3 of the 10 parameters. So for each example we pick a
    random SUBSET of parameters to mention (each with its schema
    mention_prob), phrase only those, and leave the rest at their schema
    default. That teaches the model what we actually want: use what's said,
    fall back to sensible defaults for what isn't.

    It's also why we don't need every combination of all 10 parameters:
    each output head learns its own phrase -> value mapping, so a few
    thousand random examples are enough for those mappings to recombine
    freely at prediction time.

    python text2city/generate_dataset.py --n 15000 --seed 0 --out data/text2city_dataset.csv
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from text2city.schema import PARAM_BY_NAME, PARAM_NAMES, default_vector  # noqa: E402

# ---------------------------------------------------------------------------
# Phrase banks. Every phrase must read naturally after "A city ..." so any
# subset can be joined into one sentence.
# ---------------------------------------------------------------------------

TERRAIN_PHRASES = {
    "flat": ["on flat land", "built on completely flat terrain", "on level ground"],
    "hills": ["set in rolling hills", "spread across gentle hills", "on hilly terrain"],
    "mountains": ["surrounded by mountains", "nestled in rugged mountains", "high in the mountains"],
    "valley": ["nestled in a deep valley", "tucked into a valley", "set in a river valley"],
    "coastal_harbor": ["built around a harbor", "hugging the coastline", "by the sea", "on the coast",
                       "a seaside setting", "along the seaside", "a coastal city", "a coastal setting",
                       "coastal terrain"],
    "peninsula": ["built on a peninsula", "surrounded by water on three sides", "on a narrow peninsula",
                  "a peninsula city", "peninsula terrain"],
    "river": ["split by a winding river", "built along a river", "with a river cutting through it",
              "a river city", "river terrain"],
}


def terrain_phrases(name: str) -> list[str]:
    """Phrases for a terrain type -- with a fallback for a type you invented yourself."""
    return TERRAIN_PHRASES.get(name, [f"with {name.replace('_', ' ')} terrain"])


CITY_ARCHETYPE_PHRASES = {
    "monocentric": ["with a single downtown core", "built around one central business district",
                    "with one clear city center", "a monocentric city", "a monocentric layout",
                    "this city is monocentric"],
    "polycentric": ["with several distinct business districts", "made up of multiple downtown clusters",
                    "with scattered urban centers", "with multiple business hubs",
                    "with two separate downtown areas", "made up of several separate city centers",
                    "a polycentric city", "a polycentric layout", "this city is polycentric"],
    "linear": ["stretched out in a long corridor", "elongated along a single axis", "laid out in a line",
               "long and narrow", "thin and elongated in shape", "a linear city", "a linear layout",
               "this city is linear", "a wide, elongated shape", "a wide and stretched out shape"],
}

# Exact-count phrases, each tied to its own literal number. "one main hub"
# must never be paired with a target of 1.8 just because it shared a range
# with "two hubs" -- the value has to match what the words say.
_CLUSTER_NOUNS = ["hub", "center", "cluster", "district", "node"]
_NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def _cluster_noun_phrases(count: int) -> list[str]:
    word = _NUMBER_WORDS[count]
    plural = "" if count == 1 else "s"
    prefix = "main " if count == 1 else ""
    return [f"with {word} {prefix}{noun}{plural}" for noun in _CLUSTER_NOUNS]


NUM_CLUSTERS_EXACT = {
    1: _cluster_noun_phrases(1) + ["with just one main hub", "with a single center"],
    2: _cluster_noun_phrases(2) + ["with a couple of centers", "with just one or two hubs"],
    3: _cluster_noun_phrases(3) + ["with a handful of clusters", "with several separate hubs"],
    4: _cluster_noun_phrases(4) + ["with several separate districts"],
    5: _cluster_noun_phrases(5) + ["with many scattered clusters"],
    6: _cluster_noun_phrases(6) + ["with a large number of distinct hubs"],
}

# Continuous parameters: a list of levels, each ((low, high), [phrases]).
# A phrase from a level is always paired with a value inside that level.
CONTINUOUS_LEVELS = {
    "density": [
        ((0.0, 0.15), ["barely built up", "almost entirely empty", "sparsely inhabited", "sparse", "very sparse"]),
        ((0.15, 0.35), ["lightly built up", "fairly sparse", "not very built up"]),
        ((0.35, 0.6), ["moderately built up", "with medium density", "reasonably built up"]),
        ((0.6, 0.8), ["densely populated", "densely built", "quite crowded with buildings", "dense"]),
        ((0.8, 1.0), ["extremely dense", "packed with buildings", "ultra-dense and crowded", "very dense",
                      "highly concentrated"]),
    ],
    "cluster_tightness": [
        ((0.0, 0.3), ["with buildings spread loosely across the city", "with a sprawling, spread-out layout",
                      "spread out", "sparse layout"]),
        ((0.3, 0.6), ["with a moderately clustered downtown", "with buildings loosely grouped together"]),
        ((0.6, 1.0), ["with a tightly clustered downtown core", "with a compact, concentrated skyline",
                      "concentrated", "compact", "tightly clustered"]),
    ],
    "relief": [
        ((0.0, 0.5), ["with almost no elevation change", "on very flat, gentle ground"]),
        ((0.5, 1.0), ["with gently varying elevation", "with mild rolling terrain"]),
        ((1.2, 2.0), ["with dramatic peaks and valleys", "with extreme, jagged elevation changes"]),
    ],
    "water_level": [
        ((0.0, 0.1), ["with barely any water", "with just a trickle of water"]),
        ((0.1, 0.25), ["with a modest amount of water", "with a small river or bay"]),
        ((0.25, 0.45), ["with a fair amount of water", "with a sizeable harbor"]),
        ((0.45, 0.6), ["with a large body of water", "with an expansive bay"]),
    ],
    "green_amount": [
        ((0.0, 0.15), ["with very few parks", "with almost no greenery"]),
        ((0.15, 0.4), ["with some green spaces", "with a handful of parks"]),
        ((0.4, 0.7), ["with lots of green areas", "with many parks scattered around", "full of parks and trees"]),
        ((0.7, 1.0), ["covered almost entirely in greenery", "mostly parks and forest", "covered in green space",
                      "mostly green space throughout"]),
    ],
    "road_continuity": [
        ((0.0, 0.15), ["with almost no roads at all", "with barely any streets", "with very few roads"]),
        ((0.15, 0.3), ["with a broken, patchy street network", "with disconnected, fragmented roads"]),
        ((0.3, 0.7), ["with a fairly normal street layout", "with an ordinary road network"]),
        ((0.7, 1.0), ["with a fully connected grid of streets", "with a dense, well-connected road network"]),
    ],
}

GREEN_OVERRIDE_PHRASES = [
    "reclaimed almost entirely by nature",
    "overgrown, with greenery taking over the streets",
]

SENTENCE_STARTERS = [
    "A city {items}.",
    "Generate a city {items}.",
    "I'd like a city {items}.",
    "Make me a city {items}.",
    "Design a city {items}.",
]


def _sample_continuous(rng: random.Random, levels: list):
    """Pick one level, then a value inside it and a phrase from it. Returns (value, phrase)."""
    # TODO 1: pick ONE level at random, then a value inside that level's
    # range and one of that level's phrases.
    # Each level looks like ((low, high), [phrase, phrase, ...]), so you can
    # unpack one with:   (low, high), phrases = level
    # Hints: rng.choice(a_list) picks an item; rng.uniform(a, b) picks a float.
    level = rng.choice(levels)
    (low, high), phrases = level
    value = rng.uniform(low, high)
    phrase = rng.choice(phrases)
    return value, phrase


def _join_phrases(phrases: list[str]) -> str:
    """['a', 'b', 'c'] -> 'a, b, and c'   ['a'] -> 'a'"""
    if len(phrases) == 1:
        return phrases[0]
    return ", ".join(phrases[:-1]) + ", and " + phrases[-1]


def generate_example(rng: random.Random) -> dict:
    params = default_vector()
    mentioned_phrases = []

    # --- num_clusters and city_archetype are resolved together, first ---
    # They're the one pair with a real logical link: "five centers" always
    # means a non-monocentric city. Handling them together guarantees the
    # sentence never says "five centers" while the label says monocentric.
    num_clusters_spec = PARAM_BY_NAME["num_clusters"]
    city_spec = PARAM_BY_NAME["city_archetype"]

    num_clusters_mentioned = rng.random() <= num_clusters_spec.mention_prob
    if num_clusters_mentioned:
        count = rng.choice(list(NUM_CLUSTERS_EXACT.keys()))
        mentioned_phrases.append(rng.choice(NUM_CLUSTERS_EXACT[count]))
        params["num_clusters"] = max(1.0, min(6.0, count + rng.uniform(-0.25, 0.25)))
    implies_non_monocentric = num_clusters_mentioned and params["num_clusters"] > 1.5

    non_mono = [c for c in city_spec.choices if c != "monocentric"]
    if rng.random() <= city_spec.mention_prob:
        choices = non_mono if (implies_non_monocentric and non_mono) else city_spec.choices
        value = rng.choice(choices)
        mentioned_phrases.append(rng.choice(CITY_ARCHETYPE_PHRASES[value]))
        params["city_archetype"] = value
    elif implies_non_monocentric and non_mono:
        params["city_archetype"] = rng.choice(non_mono)

    # --- every other parameter: mention it or leave it at its default ---
    for spec in PARAM_BY_NAME.values():
        if spec.name in ("num_clusters", "city_archetype"):
            continue  # handled above

        # TODO 2: should this sentence mention this parameter? It should with
        # probability spec.mention_prob. If it should NOT, `continue` -- the
        # parameter then keeps its default value, which is the whole point.
        # Hint: rng.random() is a float in [0, 1).
        if rng.random() > spec.mention_prob:
            continue

        if spec.name == "terrain":
            value = rng.choice(spec.choices)
            phrase = rng.choice(terrain_phrases(value))
        elif spec.name == "green_override":
            value = True
            phrase = rng.choice(GREEN_OVERRIDE_PHRASES)
        else:
            value, phrase = _sample_continuous(rng, CONTINUOUS_LEVELS[spec.name])

        # TODO 3: record it -- store `value` in params under this parameter's
        # name, and add `phrase` to mentioned_phrases. Both, or the sentence
        # and its label won't agree.
        params[spec.name] = value
        mentioned_phrases.append(phrase)

    if not mentioned_phrases:
        # guarantee at least one mention so the sentence isn't empty
        (low, high), phrases = CONTINUOUS_LEVELS["density"][2]
        mentioned_phrases.append(rng.choice(phrases))
        params["density"] = rng.uniform(low, high)

    rng.shuffle(mentioned_phrases)
    sentence = rng.choice(SENTENCE_STARTERS).format(items=_join_phrases(mentioned_phrases))
    return {"text": sentence, **params}


def generate_dataset(n: int, seed: int = 0) -> list:
    rng = random.Random(seed)
    return [generate_example(rng) for _ in range(n)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=15000, help="number of examples to generate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="data/text2city_dataset.csv")
    args = parser.parse_args()

    rows = generate_dataset(args.n, seed=args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text"] + PARAM_NAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} examples to {out_path}")
    print("\nsample rows:")
    for row in rows[:5]:
        print(f"  {row['text']}")
        print(f"    -> {({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items() if k != 'text'})}")


if __name__ == "__main__":
    main()
