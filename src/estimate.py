from src import config


def gepa_light_calls(val_size):
    """How many student calls GEPA's 'light' budget allows for one predictor.

    From DSPy 3.4.1's auto_budget: V + 5n + 35N + ((N + 1) // 5 + 1) * V with n = 6 and N = 10, V = validation tasks.
    """
    n, trials = 6, 10
    return val_size + 5 * n + 35 * trials + ((trials + 1) // 5 + 1) * val_size


def estimate(plan):
    """Print and return the estimated calls and dollars per model before an LLM run.

    plan is a list of rows: {"model": name, "calls": n, "basis": "assumed" or "measured",
    and either "seconds": total GPU seconds (priced at the T4 rate) or
    "input_tokens" and "output_tokens": average per call (priced at the Gemini rates)}.
    Returns {model: {"calls": n, "dollars": x, "basis": text}}, summed over that model's rows.
    """
    totals = {}
    for row in plan:
        if "seconds" in row:
            dollars = row["seconds"] * config.T4_PER_SECOND
        else:
            per_call = (
                row["input_tokens"] * config.GEMINI_INPUT_PER_MILLION
                + row["output_tokens"] * config.GEMINI_OUTPUT_PER_MILLION
            ) / 1e6
            dollars = row["calls"] * per_call
        total = totals.setdefault(
            row["model"], {"calls": 0, "dollars": 0.0, "basis": "measured"}
        )
        total["calls"] += row["calls"]
        total["dollars"] += dollars
        if row["basis"] == "assumed":
            total["basis"] = "assumed"

    print(f"{'model':24}{'calls':>8}{'est. dollars':>14}  basis")
    for model, total in totals.items():
        print(
            f"{model:24}{total['calls']:>8}{total['dollars']:>14.2f}  {total['basis']}"
        )
    return totals
