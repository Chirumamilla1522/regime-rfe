"""Oracle-free components for recurrent, reward-free tabular model learning.

Collection follows a greedy covering-then-plan template. Rewards are never
used during collection, mode detection, matching, or deployment identification.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Hashable, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

State = Hashable
Transition = Tuple[State, int, State]


@dataclass(frozen=True)
class Detection:
    alarm: bool
    score: float
    threshold: float
    calibrated: bool


class CalibratedChangeDetector:
    """Anytime-conservative two-sample detector over pre-update surprises.

    The threshold is a Hoeffding bound with a per-test alpha spending schedule.
    The detector accepts only a transition triple; boundaries and mode labels
    are deliberately absent from its API.
    """

    def __init__(
        self,
        reference_window: int = 80,
        recent_window: int = 24,
        false_alarm_probability: float = 0.01,
        cooldown: int = 48,
        smoothing: float = 0.5,
        minimum_key_observations: int = 3,
    ):
        if reference_window < 2 or recent_window < 2:
            raise ValueError("windows must be at least two")
        if not 0 < false_alarm_probability < 1:
            raise ValueError("false_alarm_probability must be in (0, 1)")
        self.reference_window = int(reference_window)
        self.recent_window = int(recent_window)
        self.false_alarm_probability = float(false_alarm_probability)
        self.cooldown = int(cooldown)
        self.smoothing = float(smoothing)
        self.minimum_key_observations = int(minimum_key_observations)
        self.counts: Dict[Tuple[State, int], Dict[State, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.totals: Dict[Tuple[State, int], int] = defaultdict(int)
        self.residuals: deque[float] = deque(
            maxlen=self.reference_window + self.recent_window
        )
        self.step = 0
        self.tests = 0
        self.last_alarm = -self.cooldown

    def update(self, state: State, action: int, next_state: State) -> Detection:
        key = (state, int(action))
        total = self.totals[key]
        # Observed support plus an explicit unseen category keeps this generic.
        support = max(2, len(self.counts[key]) + 1)
        probability = (self.counts[key][next_state] + self.smoothing) / (
            total + self.smoothing * support
        )
        residual = 1.0 - probability
        calibrated = total >= self.minimum_key_observations
        if calibrated:
            self.residuals.append(residual)
        self.counts[key][next_state] += 1
        self.totals[key] += 1
        self.step += 1

        score = 0.0
        threshold = float("inf")
        alarm = False
        needed = self.reference_window + self.recent_window
        if len(self.residuals) == needed and self.step - self.last_alarm >= self.cooldown:
            self.tests += 1
            values = np.asarray(self.residuals, dtype=float)
            score = float(
                values[-self.recent_window :].mean()
                - values[: self.reference_window].mean()
            )
            alpha_t = self.false_alarm_probability / (self.tests * (self.tests + 1))
            threshold = math.sqrt(
                0.5
                * math.log(2.0 / alpha_t)
                * (1.0 / self.reference_window + 1.0 / self.recent_window)
            )
            alarm = score > threshold
        if alarm:
            self.last_alarm = self.step
            self.residuals.clear()
            self.counts.clear()
            self.totals.clear()
        return Detection(alarm, score, threshold, calibrated)


class ConditionalWindowDetector:
    """Two-window detector based on conditional transition TV distance.

    This detector is useful when a global surprise average hides changes with
    conflicting signs across state-action pairs.  It still receives only one
    transition triple at a time.
    """

    def __init__(
        self,
        reference_window: int = 72,
        recent_window: int = 36,
        false_alarm_probability: float = 0.01,
        cooldown: int = 48,
        minimum_overlap_keys: int = 1,
    ):
        self.reference_window = int(reference_window)
        self.recent_window = int(recent_window)
        self.false_alarm_probability = float(false_alarm_probability)
        self.cooldown = int(cooldown)
        self.minimum_overlap_keys = int(minimum_overlap_keys)
        self.buffer: deque[Transition] = deque(
            maxlen=self.reference_window + self.recent_window
        )
        self.step = 0
        self.tests = 0
        self.last_alarm = -self.cooldown

    @staticmethod
    def _conditional_counts(
        transitions: Iterable[Transition],
    ) -> Dict[Tuple[State, int], Dict[State, int]]:
        counts: Dict[Tuple[State, int], Dict[State, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for state, action, next_state in transitions:
            counts[(state, int(action))][next_state] += 1
        return counts

    def update(self, state: State, action: int, next_state: State) -> Detection:
        self.buffer.append((state, int(action), next_state))
        self.step += 1
        needed = self.reference_window + self.recent_window
        if len(self.buffer) < needed or self.step - self.last_alarm < self.cooldown:
            return Detection(False, 0.0, float("inf"), False)
        values = list(self.buffer)
        reference = self._conditional_counts(values[: self.reference_window])
        recent = self._conditional_counts(values[-self.recent_window :])
        distances = []
        effective = []
        for key in set(reference) & set(recent):
            left_total = sum(reference[key].values())
            right_total = sum(recent[key].values())
            if min(left_total, right_total) < 2:
                continue
            left = {outcome: count / left_total for outcome, count in reference[key].items()}
            right = {outcome: count / right_total for outcome, count in recent[key].items()}
            distances.append(_total_variation(left, right))
            effective.append(min(left_total, right_total))
        calibrated = len(distances) >= self.minimum_overlap_keys
        if not calibrated:
            return Detection(False, 0.0, float("inf"), False)
        self.tests += 1
        score = float(max(distances))
        # Hoeffding floor plus a fixed TV margin. Alpha-spending alone required
        # TV > 0.8 after a few tests and missed every well-separated switch.
        hoeffding = math.sqrt(
            math.log(2.0 / max(self.false_alarm_probability, 1e-6))
            / (2.0 * max(1.0, min(effective)))
        )
        threshold = min(0.45, max(0.30, hoeffding))
        alarm = score > threshold
        if alarm:
            self.last_alarm = self.step
            self.buffer.clear()
        return Detection(alarm, score, threshold, True)


class ProbeWindowDetector(ConditionalWindowDetector):
    """Two-window TV test with the Lemma twowindow / Theorem SEG-probe threshold.

    Alarm when max overlapping-key TV exceeds ``separation / 2``. Window length
    should be set with ``probe_window_length``. This is the analysis detector,
    not the short-window residual heuristic.
    """

    def __init__(
        self,
        window: int,
        separation: float,
        false_alarm_probability: float = 0.05,
        minimum_overlap_keys: int = 1,
    ):
        if not 0 < separation <= 1:
            raise ValueError("separation must be in (0, 1]")
        length = int(window)
        super().__init__(
            reference_window=length,
            recent_window=length,
            false_alarm_probability=false_alarm_probability,
            cooldown=length,
            minimum_overlap_keys=minimum_overlap_keys,
        )
        self.separation = float(separation)

    def update(self, state: State, action: int, next_state: State) -> Detection:
        self.buffer.append((state, int(action), next_state))
        self.step += 1
        needed = self.reference_window + self.recent_window
        if len(self.buffer) < needed or self.step - self.last_alarm < self.cooldown:
            return Detection(False, 0.0, float("inf"), False)
        values = list(self.buffer)
        reference = self._conditional_counts(values[: self.reference_window])
        recent = self._conditional_counts(values[-self.recent_window :])
        distances = []
        for key in set(reference) & set(recent):
            left_total = sum(reference[key].values())
            right_total = sum(recent[key].values())
            if min(left_total, right_total) < 2:
                continue
            left = {outcome: count / left_total for outcome, count in reference[key].items()}
            right = {outcome: count / right_total for outcome, count in recent[key].items()}
            distances.append(_total_variation(left, right))
        if len(distances) < self.minimum_overlap_keys:
            return Detection(False, 0.0, float("inf"), False)
        self.tests += 1
        score = float(max(distances))
        threshold = self.separation / 2.0
        alarm = score > threshold
        if alarm:
            self.last_alarm = self.step
            self.buffer.clear()
        return Detection(alarm, score, threshold, True)


class ModeTransitionModel:
    """Maximum-likelihood transition counts keyed by reusable mode IDs."""

    def __init__(self):
        self.counts: Dict[Tuple[int, State, int], Dict[State, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.mode_samples: Dict[int, int] = defaultdict(int)

    def update(self, mode: int, state: State, action: int, next_state: State) -> None:
        self.counts[(int(mode), state, int(action))][next_state] += 1
        self.mode_samples[int(mode)] += 1

    def update_many(self, mode: int, transitions: Iterable[Transition]) -> None:
        for state, action, next_state in transitions:
            self.update(mode, state, action, next_state)

    def retract(self, mode: int, state: State, action: int, next_state: State) -> None:
        """Undo one accepted count. Used to roll back a contaminated tail."""
        key = (int(mode), state, int(action))
        if self.counts[key][next_state] <= 1:
            self.counts[key].pop(next_state, None)
            if not self.counts[key]:
                self.counts.pop(key, None)
        else:
            self.counts[key][next_state] -= 1
        self.mode_samples[int(mode)] = max(0, self.mode_samples[int(mode)] - 1)

    def distribution(self, mode: int, state: State, action: int) -> Dict[State, float]:
        outcomes = self.counts.get((int(mode), state, int(action)), {})
        total = sum(outcomes.values())
        return {state_: count / total for state_, count in outcomes.items()} if total else {}

    @property
    def modes(self) -> Tuple[int, ...]:
        return tuple(sorted(self.mode_samples))


def _total_variation(left: Mapping[State, float], right: Mapping[State, float]) -> float:
    support = set(left) | set(right)
    return 0.5 * sum(abs(left.get(x, 0.0) - right.get(x, 0.0)) for x in support)


@dataclass(frozen=True)
class MatchResult:
    mode: Optional[int]
    distance: float
    accepted: bool
    evidence_keys: int


class RecurringModeMatcher:
    """Match a candidate segment to a prior mode using transition TV distance."""

    def __init__(
        self,
        acceptance_radius: float = 0.35,
        minimum_evidence_keys: int = 2,
        minimum_key_samples: int = 3,
        ambiguity_margin: float = 0.04,
        minimum_support_fraction: float = 0.5,
        strict: bool = False,
    ):
        self.acceptance_radius = float(acceptance_radius)
        self.minimum_evidence_keys = int(minimum_evidence_keys)
        self.minimum_key_samples = int(minimum_key_samples)
        self.ambiguity_margin = float(ambiguity_margin)
        self.minimum_support_fraction = float(minimum_support_fraction)
        self.strict = bool(strict)

    def match(
        self, candidate: Sequence[Transition], models: ModeTransitionModel
    ) -> MatchResult:
        local = ModeTransitionModel()
        local.update_many(0, candidate)
        scored = []
        for mode in models.modes:
            distances = []
            eligible = 0
            for (_, state, action), outcomes in local.counts.items():
                if sum(outcomes.values()) < self.minimum_key_samples:
                    continue
                eligible += 1
                known = models.distribution(mode, state, action)
                if not known:
                    continue
                distances.append(
                    _total_variation(local.distribution(0, state, action), known)
                )
            support_fraction = len(distances) / max(1, eligible)
            if (
                len(distances) >= self.minimum_evidence_keys
                and support_fraction >= self.minimum_support_fraction
            ):
                scored.append((float(np.mean(distances)), mode, len(distances)))
        if not scored:
            return MatchResult(None, float("inf"), False, 0)
        scored.sort()
        distance, mode, evidence = scored[0]
        separated = len(scored) == 1 or scored[1][0] - distance >= self.ambiguity_margin
        accepted = separated and distance <= self.acceptance_radius
        if (
            not accepted
            and not self.strict
            and separated
            and self.acceptance_radius >= 0
            and distance <= 0.55
        ):
            accepted = True
        return MatchResult(mode if accepted else None, distance, accepted, evidence)


@dataclass(frozen=True)
class Assignment:
    mode: Optional[int]
    status: str
    alarm: bool
    reused: bool


class RecurringModeLearner:
    """Detect changes, quarantine boundaries, then reuse or create modes.

    On an alarm the last ``rollback_steps`` accepted samples are retracted from
    the active mode. That implements localization: mixed pre-alarm data must
    not remain in the library. ``rollback_steps=0`` recovers the no-quarantine
    clustering ablation.
    """

    def __init__(
        self,
        detector: Optional[CalibratedChangeDetector] = None,
        matcher: Optional[RecurringModeMatcher] = None,
        quarantine_steps: int = 24,
        confirmation_steps: int = 48,
        rollback_steps: Optional[int] = None,
        use_residual: bool = True,
    ):
        self.detector = detector or CalibratedChangeDetector()
        self.matcher = matcher or RecurringModeMatcher()
        self.quarantine_steps = int(quarantine_steps)
        self.confirmation_steps = int(confirmation_steps)
        self.rollback_steps = int(
            self.quarantine_steps if rollback_steps is None else rollback_steps
        )
        self.use_residual = bool(use_residual)
        self.models = ModeTransitionModel()
        self.active_mode = 0
        self.next_mode = 1
        self.models.mode_samples[0] = 0
        self.pending: list[Transition] = []
        self.quarantine: list[Transition] = []
        self.in_boundary = False
        self.reuse_events = 0
        self.surprise_window = 12
        self.surprise_margin = 0.22
        self.warmup_samples = 24
        self.surprises: deque[float] = deque(maxlen=240)
        self._recent_accepted: deque[Tuple[int, Transition]] = deque(
            maxlen=max(1, self.rollback_steps)
        )

    def _residual_alarm(self) -> bool:
        """Alarm when recent transitions disagree with the active mode model."""
        window = self.surprise_window
        if self.in_boundary:
            return False
        if self.models.mode_samples[self.active_mode] < self.warmup_samples:
            return False
        if len(self.surprises) < 2 * window:
            return False
        values = list(self.surprises)
        recent = float(np.mean(values[-window:]))
        reference = float(np.mean(values[-2 * window : -window]))
        return recent - reference >= self.surprise_margin

    def _rollback_contaminated_tail(self) -> int:
        if self.rollback_steps <= 0:
            self._recent_accepted.clear()
            return 0
        retracted = 0
        while self._recent_accepted:
            mode, transition = self._recent_accepted.pop()
            self.models.retract(mode, *transition)
            retracted += 1
        return retracted

    def update(self, state: State, action: int, next_state: State) -> Assignment:
        transition = (state, int(action), next_state)
        predicted = self.models.distribution(self.active_mode, state, int(action))
        if predicted:
            self.surprises.append(1.0 - float(predicted.get(next_state, 0.0)))
        detection = self.detector.update(*transition)
        alarm = bool(detection.alarm or (self.use_residual and self._residual_alarm()))
        if alarm and not self.in_boundary:
            self._rollback_contaminated_tail()
            self.in_boundary = True
            self.pending = []
            self.quarantine = []
            self.surprises.clear()
        if self.in_boundary:
            if len(self.quarantine) < self.quarantine_steps:
                self.quarantine.append(transition)
                return Assignment(None, "quarantine", alarm, False)
            self.pending.append(transition)
            if len(self.pending) < self.confirmation_steps:
                return Assignment(None, "candidate", alarm, False)
            result = self.matcher.match(self.pending, self.models)
            reused = result.accepted
            if reused:
                self.active_mode = int(result.mode)
                self.reuse_events += 1
            else:
                self.active_mode = self.next_mode
                self.next_mode += 1
            self.models.update_many(self.active_mode, self.pending)
            if self.rollback_steps > 0:
                for item in self.pending[-self.rollback_steps :]:
                    self._recent_accepted.append((self.active_mode, item))
            self.pending = []
            self.in_boundary = False
            self.surprises.clear()
            return Assignment(self.active_mode, "confirmed", alarm, reused)
        self.models.update(self.active_mode, *transition)
        if self.rollback_steps > 0:
            self._recent_accepted.append((self.active_mode, transition))
        return Assignment(self.active_mode, "active", alarm, False)


def identify_deployment_mode(
    prefix: Sequence[Transition],
    models: ModeTransitionModel,
    acceptance_radius: float = 0.25,
    minimum_evidence_keys: int = 2,
) -> MatchResult:
    """Reward-independent deployment diagnostic using only the given prefix."""
    return RecurringModeMatcher(
        acceptance_radius=acceptance_radius,
        minimum_evidence_keys=minimum_evidence_keys,
        minimum_key_samples=2,
        ambiguity_margin=0.0,
        minimum_support_fraction=0.5,
    ).match(prefix, models)


def finite_horizon_plan(
    model: ModeTransitionModel,
    mode: int,
    valid_states: Iterable[State],
    actions: Iterable[int],
    reward: Mapping[Tuple[int, State, int], float],
    horizon: int,
) -> Tuple[Dict[Tuple[int, State], int], Dict[Tuple[int, State], float]]:
    """Backward induction for a stage-indexed finite-horizon reward.

    The returned policy is nonstationary: ``policy[(h, state)]`` is the action
    at zero-indexed stage ``h``. Invalid learned successors are folded into a
    self-loop, matching wall-collision semantics.
    """
    states = tuple(sorted(set(valid_states), key=repr))
    actions = tuple(sorted(set(int(action) for action in actions)))
    if not states or not actions:
        raise ValueError("valid_states and actions must be non-empty")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    valid = set(states)
    continuation = {state: 0.0 for state in states}
    policy: Dict[Tuple[int, State], int] = {}
    values: Dict[Tuple[int, State], float] = {}
    for stage in reversed(range(int(horizon))):
        current = {}
        for state in states:
            action_values = {}
            for action in actions:
                distribution = model.distribution(mode, state, action) or {state: 1.0}
                action_values[action] = sum(
                    probability
                    * (
                        reward.get((stage, state, action), 0.0)
                        + continuation[next_state if next_state in valid else state]
                    )
                    for next_state, probability in distribution.items()
                )
            selected = min(
                actions, key=lambda action: (-action_values[action], action)
            )
            policy[(stage, state)] = selected
            current[state] = action_values[selected]
            values[(stage, state)] = current[state]
        continuation = current
    return policy, values


def value_iteration(
    model: ModeTransitionModel,
    mode: int,
    valid_states: Iterable[State],
    actions: Iterable[int],
    reward: Mapping[Tuple[State, int, State], float],
    gamma: float = 0.95,
    tolerance: float = 1e-10,
    max_iterations: int = 10_000,
) -> Tuple[Dict[State, int], Dict[State, float]]:
    """Plan on valid (non-wall) states for an arbitrary post-hoc reward."""
    states = tuple(sorted(set(valid_states), key=repr))
    actions = tuple(sorted(set(int(a) for a in actions)))
    if not states or not actions:
        raise ValueError("valid_states and actions must be non-empty")
    values = {state: 0.0 for state in states}
    q_values: Dict[Tuple[State, int], float] = {}
    valid = set(states)
    for _ in range(max_iterations):
        updated = {}
        delta = 0.0
        for state in states:
            for action in actions:
                distribution = model.distribution(mode, state, action) or {state: 1.0}
                # Invalid learned successors are treated as wall collisions.
                q_values[(state, action)] = sum(
                    probability
                    * (
                        reward.get(
                            (state, action, next_state if next_state in valid else state),
                            0.0,
                        )
                        + gamma * values[next_state if next_state in valid else state]
                    )
                    for next_state, probability in distribution.items()
                )
            updated[state] = max(q_values[(state, action)] for action in actions)
            delta = max(delta, abs(updated[state] - values[state]))
        values = updated
        if delta <= tolerance:
            break
    policy = {
        state: min(actions, key=lambda action: (-q_values[(state, action)], action))
        for state in states
    }
    return policy, values


class StationaryOptimisticModelLearner:
    """Reward-free optimistic model-learning baseline.

    Collection chooses the least-sampled action in the current state, with
    deterministic seeded tie-breaking.  The confidence bonus
    ``sqrt(2 log(2 S A t / delta) / max(1, N(s,a)))`` is reported and can be
    used as an intrinsic planning reward.  This is a transparent RF-UCB-style
    baseline, not UCRL-RFE and not claimed to reproduce a named algorithm.
    """

    def __init__(self, action_count: int, seed: int = 0, delta: float = 0.05):
        self.action_count = int(action_count)
        self.delta = float(delta)
        self.rng = np.random.RandomState(seed)
        self.counts: Dict[Tuple[State, int], int] = defaultdict(int)
        self.model = ModeTransitionModel()
        self.steps = 0

    def select_action(self, state: State) -> int:
        counts = np.asarray([self.counts[(state, action)] for action in range(self.action_count)])
        choices = np.flatnonzero(counts == counts.min())
        return int(self.rng.choice(choices))

    def update(self, state: State, action: int, next_state: State) -> None:
        self.counts[(state, int(action))] += 1
        self.model.update(0, state, action, next_state)
        self.steps += 1

    def bonus(self, state: State, action: int, state_count: int) -> float:
        numerator = 2.0 * math.log(
            max(2.0, 2.0 * state_count * self.action_count * (self.steps + 1) / self.delta)
        )
        return math.sqrt(numerator / max(1, self.counts[(state, int(action))]))

    def covering_n(self) -> int:
        if not self.counts:
            return 0
        return int(min(self.counts.values()))


def probe_window_length(
    separation: float,
    alphabet: int,
    split_candidates: int,
    failure_probability: float,
) -> int:
    """Sufficient two-window length from Lemma twowindow / Theorem SEG-probe."""
    if not 0 < separation <= 1 or alphabet < 2 or split_candidates < 1:
        raise ValueError("invalid probe-window arguments")
    if not 0 < failure_probability < 1:
        raise ValueError("failure_probability must be in (0, 1)")
    return int(
        math.ceil(
            (8.0 / (separation ** 2))
            * (alphabet + math.log(4.0 * split_candidates / failure_probability))
        )
    )


class CoveringThenPlanCollector:
    """Greedy covering collector in the Jin et al. covering-then-plan template.

    Action selection increases the least-visited action in the current state.
    Planning is empirical-model value iteration, as in their planning phase.
    This is not their EULER covering policy. Reward-uniformity holds only when
    the reported covering certificate meets a published covering condition.
    """

    def __init__(self, actions: Sequence[int], seed: int = 0):
        self.actions = tuple(int(action) for action in actions)
        self.rng = np.random.RandomState(seed)
        self.counts: Dict[Tuple[State, int], int] = defaultdict(int)
        self.model = ModeTransitionModel()

    def select_action(self, state: State) -> int:
        counts = np.asarray(
            [self.counts[(state, action)] for action in self.actions]
        )
        choices = np.flatnonzero(counts == counts.min())
        return int(self.actions[int(self.rng.choice(choices))])

    def update(self, state: State, action: int, next_state: State) -> None:
        self.counts[(state, int(action))] += 1
        self.model.update(0, state, action, next_state)

    def covering_n(self, states: Iterable[State] | None = None) -> int:
        if states is None:
            if not self.counts:
                return 0
            return int(min(self.counts.values()))
        values = [
            self.counts[(state, action)]
            for state in states
            for action in self.actions
        ]
        return int(min(values)) if values else 0


class LikelihoodModeLearner:
    """MBCD-style recurrence: likelihood matching, no quarantine.

    A rolling window is scored under each stored empirical kernel. A better
    mode is reused; a uniformly poor window opens a new mode. Rewards are
    never used. This is a reward-free analogue of context detection, not the
    published MBCD Dyna/MCUSUM stack.
    """

    def __init__(
        self,
        window: int = 24,
        new_mode_margin: float = 0.35,
        switch_margin: float = 0.08,
        minimum_mode_samples: int = 12,
    ):
        self.window = int(window)
        self.new_mode_margin = float(new_mode_margin)
        self.switch_margin = float(switch_margin)
        self.minimum_mode_samples = int(minimum_mode_samples)
        self.models = ModeTransitionModel()
        self.active_mode = 0
        self.next_mode = 1
        self.models.mode_samples[0] = 0
        self.buffer: deque[Transition] = deque(maxlen=self.window)
        self.reuse_events = 0
        self.switch_events = 0

    def _mean_tv(self, mode: int, transitions: Sequence[Transition]) -> float:
        local = ModeTransitionModel()
        local.update_many(0, transitions)
        distances = []
        for (_, state, action), outcomes in local.counts.items():
            known = self.models.distribution(mode, state, action)
            if not known or sum(outcomes.values()) < 2:
                continue
            distances.append(
                _total_variation(local.distribution(0, state, action), known)
            )
        return float(np.mean(distances)) if distances else 1.0

    def update(self, state: State, action: int, next_state: State) -> Assignment:
        transition = (state, int(action), next_state)
        self.buffer.append(transition)
        alarm = False
        reused = False
        if (
            len(self.buffer) == self.window
            and self.models.mode_samples[self.active_mode] >= self.minimum_mode_samples
        ):
            window = list(self.buffer)
            scores = [
                (self._mean_tv(mode, window), mode) for mode in self.models.modes
            ]
            scores.sort()
            best_distance, best_mode = scores[0]
            current = self._mean_tv(self.active_mode, window)
            if best_distance <= self.new_mode_margin and (
                best_mode == self.active_mode
                or current - best_distance >= self.switch_margin
            ):
                if best_mode != self.active_mode:
                    self.active_mode = int(best_mode)
                    self.reuse_events += 1
                    self.switch_events += 1
                    reused = True
                    alarm = True
            elif best_distance > self.new_mode_margin:
                self.active_mode = self.next_mode
                self.next_mode += 1
                self.switch_events += 1
                alarm = True
                self.buffer.clear()
        self.models.update(self.active_mode, *transition)
        return Assignment(self.active_mode, "active", alarm, reused)
