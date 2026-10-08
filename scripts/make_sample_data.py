"""Generate the synthetic demo dataset in ``data/sample/sample_tweets.csv``.

MiDe22 may only be redistributed as tweet IDs (Twitter/X terms), so the
repository ships a *synthetic* dataset with the same schema (``text,label``)
and a similar class balance (~70 % misinformation). It is used by the tests,
the CI smoke run and the Streamlit demo. Every text is template-generated;
none of them is a real tweet.

Usage::

    python scripts/make_sample_data.py
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

SEED = 2024
N_FAKE, N_TRUE = 168, 72
LABEL_NOISE = 0.06

TOPICS = {
    "covid": ["the vaccine", "5G towers", "face masks", "the lockdown", "the new variant"],
    "ukraine": ["the bombing", "the evacuation", "the grain deal", "the ceasefire", "the convoy"],
    "refugees": ["the new arrivals", "the asylum policy", "the border camp", "aid funding"],
    "misc": ["the election", "the earthquake", "the power outage", "the wildfire", "the bank"],
}
FAKE_TEMPLATES = [
    "BREAKING: {x} secretly causes {harm} and the media is hiding it!! share before deleted",
    "They don't want you to know: {x} led to {harm}. Wake up people #truth",
    "Insider confirms {x} was planned to trigger {harm}. Mainstream media silent...",
    "{x} is a hoax created by elites so that they can control {group}. RT!!",
    "100% proof {x} causes {harm}. Doctors are afraid to speak. #exposed",
    "Leaked video shows {x} was staged. Everything you saw on TV is fake",
    "My cousin works there: {x} resulted in {harm} but nobody reports it #coverup",
    "Why is nobody talking about this? {x} triggers {harm} in {group}",
]
TRUE_TEMPLATES = [
    "Officials report that {x} affected {num} people in the region, according to ministry data.",
    "Update: authorities say {x} is under investigation; more details expected tomorrow.",
    "Reuters: {x} reported by local agencies, {num} cases confirmed so far.",
    "Fact check: claims that {x} causes {harm} are false, experts say.",
    "The health ministry published new guidance on {x} today. Full statement on the website.",
    "According to the UN agency, {x} reached {num} people this week.",
    "Live updates on {x}: officials confirm figures and urge people to rely on verified sources.",
]
HARMS = ["infertility", "mass illness", "blackouts", "food shortages", "chaos", "panic", "crime"]
GROUPS = ["children", "our town", "ordinary people", "the elderly", "everyone"]


def _fill(template: str, rng: random.Random) -> str:
    topic = rng.choice(list(TOPICS))
    return template.format(
        x=rng.choice(TOPICS[topic]),
        harm=rng.choice(HARMS),
        group=rng.choice(GROUPS),
        num=f"{rng.randint(2, 900) * 10:,}",
    )


def generate(seed: int = SEED) -> list[tuple[str, int]]:
    rng = random.Random(seed)
    rows = [(_fill(rng.choice(FAKE_TEMPLATES), rng), 1) for _ in range(N_FAKE)]
    rows += [(_fill(rng.choice(TRUE_TEMPLATES), rng), 0) for _ in range(N_TRUE)]
    # Flip a few labels so that the task is not trivially separable.
    rows = [(t, 1 - y) if rng.random() < LABEL_NOISE else (t, y) for t, y in rows]
    rng.shuffle(rows)
    return rows


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "data" / "sample" / "sample_tweets.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(generate())
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
