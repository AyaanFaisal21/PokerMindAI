from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class Action(str, Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    RAISE = "raise"
    ALL_IN = "all_in"


class Street(str, Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"


class Position(str, Enum):
    BTN = "BTN"
    SB = "SB"
    BB = "BB"
    UTG = "UTG"
    UTG1 = "UTG+1"
    HJ = "HJ"
    CO = "CO"


class OpponentStats(BaseModel):
    vpip: float = Field(default=0.25, ge=0.0, le=1.0, description="Voluntarily put money in pot rate")
    pfr: float = Field(default=0.18, ge=0.0, le=1.0, description="Pre-flop raise rate")
    aggression_freq: float = Field(default=0.45, ge=0.0, le=1.0, description="Aggression frequency")
    fold_to_cbet: float = Field(default=0.55, ge=0.0, le=1.0, description="Fold to continuation bet")
    wtsd: float = Field(default=0.28, ge=0.0, le=1.0, description="Went to showdown rate")
    hands_observed: int = Field(default=0, description="Number of hands tracked")


class GameState(BaseModel):
    session_id: str
    hand_id: str
    street: Street
    hole_cards: list[str] = Field(description="e.g. ['Ah', 'Kd']")
    community_cards: list[str] = Field(default_factory=list, description="Board cards")
    pot_size: float
    stack_size: float
    to_call: float = Field(default=0.0)
    position: Position
    num_players: int = Field(default=2, ge=2, le=9)
    opponent_stats: OpponentStats = Field(default_factory=OpponentStats)
    available_actions: list[str] = Field(default_factory=lambda: ["fold", "call", "raise"])
    big_blind: float = Field(default=1.0)
    betting_history: list[dict] = Field(default_factory=list)
    timestamp_ms: int = Field(default=0)


class AgentResponse(BaseModel):
    agent: str
    recommended_action: Action
    raise_sizing: Optional[float] = Field(default=None, description="Raise amount if action is raise")
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    key_factors: list[str] = Field(default_factory=list)
    latency_ms: float = Field(default=0.0)


class DebateRound(BaseModel):
    round_num: int
    game_state: GameState
    responses: list[AgentResponse] = Field(default_factory=list)
    consensus_reached: bool = False
    disagreement_axes: list[str] = Field(default_factory=list)


class FinalDecision(BaseModel):
    session_id: str
    hand_id: str
    action: Action
    raise_sizing: Optional[float] = None
    confidence: float
    debate_rounds: int
    agent_votes: dict[str, str] = Field(default_factory=dict)
    reasoning_trace: list[str] = Field(default_factory=list)
    total_latency_ms: float
    individual_latencies: dict[str, float] = Field(default_factory=dict)


class SessionStats(BaseModel):
    session_id: str
    hands_played: int = 0
    vpip_hands: int = 0
    pfr_hands: int = 0
    total_aggression: int = 0
    total_actions: int = 0
    folds: int = 0
    calls: int = 0
    raises: int = 0
    action_timings_ms: list[float] = Field(default_factory=list)
    session_start_ts: float = 0.0
    last_action_ts: float = 0.0

    @property
    def vpip(self) -> float:
        return self.vpip_hands / self.hands_played if self.hands_played > 0 else 0.0

    @property
    def pfr(self) -> float:
        return self.pfr_hands / self.hands_played if self.hands_played > 0 else 0.0

    @property
    def aggression_freq(self) -> float:
        denominator = self.raises + self.calls + self.folds
        return self.raises / denominator if denominator > 0 else 0.0
