"""Small standalone PyTorch PPO agent.

[AUTOCKT-REPLICATED] policy/value network width & depth: two hidden layers of
64 units each, matching the verified

    "model": {"fcnet_hiddens": [64, 64]}

in autockt/val_autobag_ray.py (github.com/ksettaluri6/AutoCkt) -- NOT "3
hidden layers x 50 neurons"; that figure is refuted by the actual training
script and is not used here.

[AUTOCKT-REPLICATED] action head structure: independent categorical heads of
3 logits each (one per NEBULA parameter), matching AutoCkt's
`spaces.Tuple([spaces.Discrete(3)] * len(params_id))` -- 7 heads in AutoCkt,
5 in NEBULA (see rl/autockt_action.py).

[UNVERIFIED-FROM-SOURCE] everything else about the network/optimizer.
AutoCkt's own repo does not pin activation function, shared-vs-separate
policy/value trunks, learning rate, discount (gamma), GAE lambda, clip
epsilon, entropy coefficient, or value-loss coefficient -- they are either
commented out or entirely absent from val_autobag_ray.py's `config_train`,
so the official run fell back to whatever Ray RLlib 0.6.3 (a 2018-era
release) happened to default to internally. Ray 0.6.3's internal defaults
are not reproduced here (out of scope for this session's source
verification). The constants below are ordinary PPO-literature defaults, not
AutoCkt replication, and are why this module -- unlike parameter_grid.py,
autockt_action.py, autockt_state.py, and autockt_reward.py -- is only
partially source-verified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.distributions import Categorical

# [AUTOCKT-REPLICATED]
HIDDEN_SIZES: tuple[int, int] = (64, 64)
NUM_ACTION_CHOICES = 3  # ACTION_DELTAS = (-1, 0, +2), see rl/parameter_grid.py

# [UNVERIFIED-FROM-SOURCE] standard PPO-literature defaults; see module docstring.
DEFAULT_GAMMA = 0.99
DEFAULT_GAE_LAMBDA = 0.95
DEFAULT_CLIP_EPS = 0.2
DEFAULT_ENTROPY_COEF = 0.01
DEFAULT_VALUE_COEF = 0.5
DEFAULT_LEARNING_RATE = 3e-4


class PolicyNetwork(nn.Module):
    """[UNVERIFIED-FROM-SOURCE: separate, not shared, policy/value trunks --
    a standard, defensible choice, not confirmed against Ray RLlib 0.6.3's
    internal model-building code, which was out of scope for this session's
    source verification.]
    """

    def __init__(self, state_dim: int, num_heads: int, hidden_sizes: tuple[int, int] = HIDDEN_SIZES):
        super().__init__()
        h1, h2 = hidden_sizes
        self.body = nn.Sequential(nn.Linear(state_dim, h1), nn.Tanh(), nn.Linear(h1, h2), nn.Tanh())
        self.heads = nn.ModuleList([nn.Linear(h2, NUM_ACTION_CHOICES) for _ in range(num_heads)])

    def forward(self, state: torch.Tensor) -> list[torch.Tensor]:
        features = self.body(state)
        return [head(features) for head in self.heads]


class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_sizes: tuple[int, int] = HIDDEN_SIZES):
        super().__init__()
        h1, h2 = hidden_sizes
        self.body = nn.Sequential(
            nn.Linear(state_dim, h1), nn.Tanh(), nn.Linear(h1, h2), nn.Tanh(), nn.Linear(h2, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.body(state).squeeze(-1)


@dataclass
class Transition:
    """[NEBULA ADAPTATION -- repair #2] `terminated`/`truncated` replace a
    single, conflated `done` flag so PPOAgent.compute_gae can bootstrap each
    episode-ending transition correctly: `terminated` means the episode
    genuinely ended (the AutoCkt terminal bonus was reached -- there is no
    real continuation, so it is zero-bootstrapped); `truncated` means the
    episode was only cut off by AutoCktReceiverEnv.horizon -- the true
    continuation value is unknown but nonzero, so it is bootstrapped from
    `bootstrap_value` (the value net's own estimate of the actual
    post-episode state, captured by rl/trainer.py::collect_rollout at the
    moment of truncation) instead of being treated as terminal.
    """

    state: tuple[float, ...]
    choices: tuple[int, ...]
    log_prob: float
    value: float
    reward: float
    terminated: bool = False
    truncated: bool = False
    bootstrap_value: float = 0.0  # only meaningful when truncated=True


class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        num_heads: int = 5,
        *,
        gamma: float = DEFAULT_GAMMA,
        gae_lambda: float = DEFAULT_GAE_LAMBDA,
        clip_eps: float = DEFAULT_CLIP_EPS,
        entropy_coef: float = DEFAULT_ENTROPY_COEF,
        value_coef: float = DEFAULT_VALUE_COEF,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        seed: int = 0,
    ):
        torch.manual_seed(seed)
        self.state_dim = state_dim
        self.num_heads = num_heads
        self.policy = PolicyNetwork(state_dim, num_heads)
        self.value_net = ValueNetwork(state_dim)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value_net.parameters()), lr=learning_rate
        )

    def act(self, state: Sequence[float], *, deterministic: bool = False) -> tuple[tuple[int, ...], float, float]:
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            logits_list = self.policy(state_t)
            choices = []
            log_prob_sum = 0.0
            for logits in logits_list:
                dist = Categorical(logits=logits.squeeze(0))
                choice = dist.probs.argmax() if deterministic else dist.sample()
                log_prob_sum += float(dist.log_prob(choice).item())
                choices.append(int(choice.item()))
            value = float(self.value_net(state_t).item())
        return tuple(choices), log_prob_sum, value

    def _evaluate_actions(
        self, states: torch.Tensor, choices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits_list = self.policy(states)
        log_probs = torch.zeros(states.shape[0])
        entropy = torch.zeros(states.shape[0])
        for head_index, logits in enumerate(logits_list):
            dist = Categorical(logits=logits)
            log_probs = log_probs + dist.log_prob(choices[:, head_index])
            entropy = entropy + dist.entropy()
        values = self.value_net(states)
        return log_probs, entropy, values

    def compute_gae(
        self, transitions: Sequence[Transition], last_value: float
    ) -> tuple[list[float], list[float]]:
        """[UNVERIFIED-FROM-SOURCE] standard GAE(gamma, lambda); AutoCkt's
        own repo does not specify an advantage estimator.

        [NEBULA ADAPTATION -- repair #2] distinguishes true termination from
        horizon truncation (see Transition docstring): a `terminated`
        transition is zero-bootstrapped (no real continuation); a
        `truncated` transition bootstraps from its own `bootstrap_value`
        instead of zero. Both still stop the recursive GAE(lambda) carry
        term from crossing the episode boundary (`mask=0.0`) -- only the
        *value* used in `delta` changes for the truncated case, not whether
        advantage-smoothing continues past the reset. A transition that is
        neither (mid-episode) bootstraps from the next transition's own
        value estimate, exactly as before; `last_value` remains the fallback
        bootstrap for whatever, if anything, follows the final transition in
        `transitions` (collect_rollout always ends a batch on a completed
        episode, so in practice this fallback is not exercised by rl/trainer.py).
        """

        rewards = [t.reward for t in transitions]
        values = [t.value for t in transitions]
        n = len(transitions)
        advantages = [0.0] * n
        gae = 0.0
        for i in reversed(range(n)):
            t = transitions[i]
            if t.terminated:
                bootstrap, mask = 0.0, 0.0
            elif t.truncated:
                bootstrap, mask = t.bootstrap_value, 0.0
            else:
                bootstrap = values[i + 1] if i + 1 < n else last_value
                mask = 1.0
            delta = rewards[i] + self.gamma * bootstrap - values[i]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[i] = gae
        returns = [a + v for a, v in zip(advantages, values)]
        return advantages, returns

    def update(
        self,
        transitions: Sequence[Transition],
        last_value: float,
        *,
        epochs: int = 4,
        minibatch_size: int = 32,
    ) -> dict[str, float]:
        if not transitions:
            raise ValueError("update requires at least one transition")
        advantages, returns = self.compute_gae(transitions, last_value)
        states = torch.as_tensor([t.state for t in transitions], dtype=torch.float32)
        choices = torch.as_tensor([t.choices for t in transitions], dtype=torch.long)
        old_log_probs = torch.as_tensor([t.log_prob for t in transitions], dtype=torch.float32)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32)
        returns_t = torch.as_tensor(returns, dtype=torch.float32)
        if advantages_t.numel() > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        n = states.shape[0]
        totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        minibatches = 0
        for _ in range(epochs):
            permutation = torch.randperm(n)
            for start in range(0, n, minibatch_size):
                idx = permutation[start : start + minibatch_size]
                log_probs, entropy, values = self._evaluate_actions(states[idx], choices[idx])
                ratio = torch.exp(log_probs - old_log_probs[idx])
                surrogate1 = ratio * advantages_t[idx]
                surrogate2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages_t[idx]
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                value_loss = nn.functional.mse_loss(values, returns_t[idx])
                entropy_bonus = entropy.mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_bonus

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                totals["policy_loss"] += float(policy_loss.item())
                totals["value_loss"] += float(value_loss.item())
                totals["entropy"] += float(entropy_bonus.item())
                minibatches += 1
        return {key: value / max(1, minibatches) for key, value in totals.items()}
