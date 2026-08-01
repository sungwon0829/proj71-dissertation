from datasets import load_dataset
from pathlib import Path

RAW = Path(r"C:\proj71\data\raw")
RAW.mkdir(parents=True, exist_ok=True)

SETS = {
    "esconv": "thu-coai/esconv",
    "counsel_chat": "nbertagnolli/counsel-chat",
    "psychocounsel_pref": "Psychotherapy-LLM/PsychoCounsel-Preference",
}

for name, repo in SETS.items():
    print(f"\n=== {name}: {repo} ===")
    ds = load_dataset(repo)
    print(ds)
    ds.save_to_disk(str(RAW / name))
    print(f"saved -> {RAW / name}")