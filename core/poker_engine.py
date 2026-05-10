"""
Texas Hold'em evaluation utilities.
Provides equity estimation via Monte Carlo rollout and pot-odds/EV helpers.
"""
from __future__ import annotations
import random
import time
from itertools import combinations
from typing import Optional

RANKS = "23456789TJQKA"
SUITS = "cdhs"
RANK_VALUES = {r: i for i, r in enumerate(RANKS, 2)}

HAND_RANKS = [
    "High Card", "One Pair", "Two Pair", "Three of a Kind",
    "Straight", "Flush", "Full House", "Four of a Kind",
    "Straight Flush", "Royal Flush",
]


def _rank(card: str) -> int:
    return RANK_VALUES[card[0].upper()]


def _suit(card: str) -> str:
    return card[1].lower()


def _full_deck(exclude: list[str]) -> list[str]:
    exclude_norm = {c.upper()[0] + c[1].lower() for c in exclude}
    return [r + s for r in RANKS for s in SUITS if (r + s) not in exclude_norm]


def _hand_rank(cards: list[str]) -> tuple:
    """Return a sortable tuple representing hand strength for 5-card hands."""
    ranks = sorted([_rank(c) for c in cards], reverse=True)
    suits = [_suit(c) for c in cards]

    flush = len(set(suits)) == 1
    straight = (max(ranks) - min(ranks) == 4 and len(set(ranks)) == 5) or ranks == [14, 5, 4, 3, 2]

    rank_counts = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1

    counts = sorted(rank_counts.values(), reverse=True)
    count_ranks = sorted(rank_counts.keys(), key=lambda r: (rank_counts[r], r), reverse=True)

    if flush and straight:
        category = 9 if ranks[0] == 14 else 8
    elif counts[0] == 4:
        category = 7
    elif counts[:2] == [3, 2]:
        category = 6
    elif flush:
        category = 5
    elif straight:
        category = 4
    elif counts[0] == 3:
        category = 3
    elif counts[:2] == [2, 2]:
        category = 2
    elif counts[0] == 2:
        category = 1
    else:
        category = 0

    return (category, *count_ranks)


def _best_5_of_7(cards: list[str]) -> tuple:
    return max(_hand_rank(list(combo)) for combo in combinations(cards, 5))


def monte_carlo_equity(
    hole_cards: list[str],
    community_cards: list[str],
    num_opponents: int = 1,
    simulations: int = 1000,
) -> float:
    """Estimate win equity via Monte Carlo rollout. Returns 0.0–1.0."""
    known = hole_cards + community_cards
    deck = _full_deck(known)
    board_needed = 5 - len(community_cards)

    wins = 0
    for _ in range(simulations):
        random.shuffle(deck)
        ptr = 0
        board = community_cards + deck[ptr : ptr + board_needed]
        ptr += board_needed

        my_strength = _best_5_of_7(hole_cards + board)
        won = True
        for _ in range(num_opponents):
            opp_hole = deck[ptr : ptr + 2]
            ptr += 2
            if ptr > len(deck):
                break
            opp_strength = _best_5_of_7(opp_hole + board)
            if opp_strength >= my_strength:
                won = False
                break
        if won:
            wins += 1

    return wins / simulations


def pot_odds(to_call: float, pot_size: float) -> float:
    """Minimum equity needed to break even on a call."""
    if to_call <= 0:
        return 0.0
    total = pot_size + to_call
    return to_call / total


def expected_value(equity: float, pot_size: float, to_call: float) -> float:
    """Simple EV calculation for calling."""
    win_amount = pot_size
    return equity * win_amount - (1 - equity) * to_call


def spr(stack: float, pot: float) -> float:
    """Stack-to-pot ratio."""
    return stack / pot if pot > 0 else float("inf")


def optimal_raise_size(
    pot: float,
    street: str,
    equity: float,
    spr_val: float,
) -> float:
    """Heuristic GTO raise sizing."""
    base_fraction = {
        "preflop": 3.0,
        "flop": 0.67,
        "turn": 0.75,
        "river": 1.0,
    }.get(street, 0.75)

    if equity > 0.75:
        base_fraction *= 1.25
    elif equity < 0.4:
        base_fraction *= 0.6

    if spr_val < 3:
        base_fraction = min(base_fraction, 1.0)

    return round(pot * base_fraction, 2)


class PokerEngine:
    def __init__(self, simulations: int = 800):
        self.simulations = simulations

    def evaluate(
        self,
        hole_cards: list[str],
        community_cards: list[str],
        num_opponents: int = 1,
    ) -> dict:
        t0 = time.perf_counter()
        equity = monte_carlo_equity(
            hole_cards, community_cards, num_opponents, self.simulations
        )
        elapsed = (time.perf_counter() - t0) * 1000

        return {
            "equity": round(equity, 4),
            "hand_category": HAND_RANKS[_best_5_of_7(hole_cards + community_cards)[0]]
            if len(hole_cards + community_cards) >= 5
            else "Preflop",
            "eval_ms": round(elapsed, 1),
        }

    def analyze(
        self,
        hole_cards: list[str],
        community_cards: list[str],
        pot_size: float,
        to_call: float,
        stack_size: float,
        street: str,
        num_opponents: int = 1,
    ) -> dict:
        ev_data = self.evaluate(hole_cards, community_cards, num_opponents)
        equity = ev_data["equity"]

        odds = pot_odds(to_call, pot_size)
        ev = expected_value(equity, pot_size, to_call)
        spr_val = spr(stack_size, pot_size)
        raise_size = optimal_raise_size(pot_size, street, equity, spr_val)

        return {
            **ev_data,
            "pot_odds_needed": round(odds, 4),
            "call_ev": round(ev, 2),
            "spr": round(spr_val, 2),
            "suggested_raise_size": raise_size,
            "has_pot_odds": equity >= odds,
            "is_value_bet_candidate": equity > 0.6,
            "is_bluff_candidate": equity < 0.35 and spr_val > 4,
        }
