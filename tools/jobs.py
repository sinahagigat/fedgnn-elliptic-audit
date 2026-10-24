"""
The v7 job table. One job = one experiment config x a block of seeds.
Each job writes its own artifacts file, so several Colab accounts can run
different jobs at the same time without touching each other's files.

    python tools/jobs.py            # print the table
"""

SEED_BLOCKS_4 = ["42-46", "47-51", "52-56", "57-61"]
SEED_BLOCKS_10 = ["42-43", "44-45", "46-47", "48-49", "50-51",
                  "52-53", "54-55", "56-57", "58-59", "60-61"]

JOBS = {}

for i, s in enumerate(SEED_BLOCKS_4, 1):
    JOBS[f"A{i}"] = {"experiment": "A", "config": "configs/v7_A_louvain.yaml", "seeds": s}
    JOBS[f"B{i}"] = {"experiment": "B", "config": "configs/v7_B_random.yaml", "seeds": s}
for k in range(5):
    for i, s in enumerate(SEED_BLOCKS_4, 1):
        JOBS[f"C{k}_{i}"] = {"experiment": f"C{k}", "config": f"configs/v7_C{k}_fixed.yaml", "seeds": s}
# the budget experiment is ~5-10x slower per run, so it gets smaller blocks
for i, s in enumerate(SEED_BLOCKS_10, 1):
    JOBS[f"D{i}"] = {"experiment": "D", "config": "configs/v7_D_budget.yaml", "seeds": s}

for i, s in enumerate(SEED_BLOCKS_4, 1):
    JOBS[f"E{i}"] = {"experiment": "E", "config": "configs/v7_E_budget_mlp.yaml", "seeds": s}

# rough GPU hours per job on a Colab T4, from the v6a timings
HOURS = {"A": 0.9, "B": 1.0, "C": 0.8, "D": 2.5, "E": 0.5}

# suggested order: A and B first (they also reproduce the paper's runs),
# then C, then D
PRIORITY = ([f"A{i}" for i in range(1, 5)] + [f"B{i}" for i in range(1, 5)]
            + [f"C{k}_{i}" for k in range(5) for i in range(1, 5)]
            + [f"D{i}" for i in range(1, 11)]
            + [f"E{i}" for i in range(1, 5)])

if __name__ == "__main__":
    total = 0.0
    for name in PRIORITY:
        j = JOBS[name]
        h = HOURS[j["experiment"][0]]
        total += h
        print(f"{name:6s} {j['experiment']:3s} seeds {j['seeds']:6s} ~{h:.1f} h  {j['config']}")
    print(f"\n{len(JOBS)} jobs, ~{total:.0f} GPU hours in total")
