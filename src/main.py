# src/main.py
import json
import os
import time

from src.dataloader import create_dataloader_v1, create_dataloader_v2


def load_star_wars_text(path="data/SW_EpisodeIV_VI.json"):
    with open(path) as f:
        data = json.load(f)
    return "\n".join(entry["Line"] for entry in data)


def load_star_wars_lines(path="data/SW_EpisodeIV_VI.json"):
    with open(path) as f:
        data = json.load(f)
    return [entry["Line"] for entry in data]


def time_baseline_loader(text, repeats=1, **kwargs):
    start = time.perf_counter()
    for _ in range(repeats):
        dataloader = create_dataloader_v1(text, **kwargs)
        for _ in dataloader:
            pass
    return time.perf_counter() - start


def time_v2_loader(lines, repeats=1, **kwargs):
    start = time.perf_counter()
    for _ in range(repeats):
        dataloader = create_dataloader_v2(lines, **kwargs)
        for _ in dataloader:
            pass
    return time.perf_counter() - start


def main():
    print(f"CPU count: {os.cpu_count()}")

    text = load_star_wars_text()
    baseline_total = time_baseline_loader(text, repeats=10)
    print(f"Baseline (create_dataloader_v1), 10 runs total: {baseline_total:.4f}s")
    print(
        f"Baseline (create_dataloader_v1), average per run: {baseline_total / 10:.4f}s"
    )

    lines = load_star_wars_lines()
    v2_total = time_v2_loader(lines, repeats=10)
    print(f"v2 (create_dataloader_v2), 10 runs total: {v2_total:.4f}s")
    print(f"v2 (create_dataloader_v2), average per run: {v2_total / 10:.4f}s")


if __name__ == "__main__":
    main()
