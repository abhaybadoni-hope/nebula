"""First learning baseline for the NEBULA receiver."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from simulator.rl_adapter import ReceiverRLAdapter


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a simple CEM optimizer on the NEBULA receiver"
    )
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--population", type=int, default=5)
    parser.add_argument("--elite", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/cem_training.jsonl"),
    )
    args = parser.parse_args()

    if args.elite > args.population:
        raise ValueError("--elite cannot exceed --population")

    rng = random.Random(args.seed)
    env = ReceiverRLAdapter(seed=args.seed)

    # Start with a broad distribution over normalized actions [-1, 1].
    mean = [
        0.369674437322012,
        0.27670652861099754,
        0.33331526214636176,
        0.7801971410738049,
        -0.027727059712872038,
    ]
    std = [0.15] * 5

    args.output.parent.mkdir(parents=True, exist_ok=True)

    best_reward = -math.inf
    best_action = None

    with args.output.open("w", encoding="utf-8") as output:
        evaluation_number = 0

        for iteration in range(args.iterations):
            candidates = []

            for _ in range(args.population):
                action = tuple(
                    max(-1.0, min(1.0, rng.gauss(mean[i], std[i])))
                    for i in range(5)
                )

                step = env.step(action)
                evaluation_number += 1

                candidates.append((step.reward, action))

                if step.reward > best_reward:
                    best_reward = step.reward
                    best_action = action

                row = {
                    "iteration": iteration + 1,
                    "evaluation": evaluation_number,
                    "reward": step.reward,
                    "action": action,
                    "success": step.info.get("failure_stage") is None
                    and step.reward > -100.0,
                    "failure_stage": step.info.get("failure_stage"),
                }

                output.write(json.dumps(row) + "\n")
                output.flush()

                print(json.dumps(row), flush=True)

            # Highest-reward actions become the next generation's center.
            candidates.sort(key=lambda item: item[0], reverse=True)
            elites = candidates[: args.elite]

            for i in range(5):
                values = [action[i] for _, action in elites]
                mean[i] = sum(values) / len(values)

                if len(values) > 1:
                    variance = sum(
                        (value - mean[i]) ** 2 for value in values
                    ) / len(values)
                    std[i] = max(0.15, math.sqrt(variance))
                else:
                    std[i] = max(0.15, std[i] * 0.7)

            print(
                json.dumps(
                    {
                        "iteration_complete": iteration + 1,
                        "best_reward": best_reward,
                        "best_action": best_action,
                        "mean": mean,
                        "std": std,
                    },
                    indent=2,
                ),
                flush=True,
            )

    print(
        json.dumps(
            {
                "evaluations": evaluation_number,
                "best_reward": best_reward,
                "best_action": best_action,
                "output": str(args.output),
            },
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
