"""
WebSocket protocol for Quoridor GUI communication.

Protocol Messages (JSON):

GUI -> Client (moves from human player):
{
    "type": "move",
    "move_type": "pawn" | "wall",
    "x": int,           # destination x for pawn, wall anchor x for wall
    "y": int,           # destination y for pawn, wall anchor y for wall
    "orientation": "h" | "v"  # only for walls: horizontal or vertical
}

{
    "type": "quit"
}

Client -> GUI (game state updates):
{
    "type": "gamestate",
    "players": [
        {"x": int, "y": int, "walls": int, "name": str},
        {"x": int, "y": int, "walls": int, "name": str}
    ],
    "walls": [
        {"x": int, "y": int, "orientation": "h" | "v"},
        ...
    ],
    "current_player": int,  # 0 or 1
    "score": float,         # optional evaluation score
    "winner": int | null    # null if game ongoing, 0 or 1 if won
}

{
    "type": "start",
    "player_names": [str, str]  # names for player 0 and 1
}

{
    "type": "request_move",
    "player": int  # which player should move (0 = human at GUI)
}

Coordinate system (matching C++ engine):
- (0,0) is top-left from GUI perspective
- Player 0 starts at (4,0), needs to reach y=8
- Player 1 starts at (4,8), needs to reach y=0
- Wall x,y is the top-left anchor of a 2-cell wall
- Horizontal wall blocks vertical movement
- Vertical wall blocks horizontal movement
"""
import json
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class PlayerState:
    x: int
    y: int
    walls: int
    name: str


@dataclass
class WallState:
    x: int
    y: int
    orientation: str  # "h" or "v"


@dataclass
class GameState:
    players: list[PlayerState]
    walls: list[WallState]
    current_player: int
    score: float = 0.0
    winner: Optional[int] = None

    def to_json(self) -> str:
        d = {
            "type": "gamestate",
            "players": [asdict(p) for p in self.players],
            "walls": [asdict(w) for w in self.walls],
            "current_player": self.current_player,
            "score": self.score,
            "winner": self.winner
        }
        return json.dumps(d)

    @classmethod
    def from_json(cls, data: str) -> "GameState":
        d = json.loads(data)
        players = [PlayerState(**p) for p in d["players"]]
        walls = [WallState(**w) for w in d["walls"]]
        return cls(
            players=players,
            walls=walls,
            current_player=d["current_player"],
            score=d.get("score", 0.0),
            winner=d.get("winner")
        )


@dataclass
class MoveMessage:
    move_type: str  # "pawn" or "wall"
    x: int
    y: int
    orientation: Optional[str] = None  # "h" or "v" for walls

    def to_json(self) -> str:
        d = {"type": "move", "move_type": self.move_type, "x": self.x, "y": self.y}
        if self.orientation:
            d["orientation"] = self.orientation
        return json.dumps(d)

    @classmethod
    def from_json(cls, data: str) -> "MoveMessage":
        d = json.loads(data)
        return cls(
            move_type=d["move_type"],
            x=d["x"],
            y=d["y"],
            orientation=d.get("orientation")
        )


def parse_message(data: str) -> dict:
    """Parse incoming JSON message."""
    return json.loads(data)


def make_quit_message() -> str:
    return json.dumps({"type": "quit"})


def make_start_message(player_names: list[str]) -> str:
    return json.dumps({"type": "start", "player_names": player_names})


def make_request_move_message(player: int) -> str:
    return json.dumps({"type": "request_move", "player": player})
