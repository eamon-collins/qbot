#!/usr/bin/env python3
"""
Samples viewer for Quoridor training data.

Loads .qsamples files and allows browsing through training samples,
displaying the board state and statistics.
"""
import os
import sys
import struct
import numpy as np
import pygame
from typing import List, Tuple
from pathlib import Path

# Action space constants (must match C++ and Python)
NUM_PAWN_ACTIONS = 81   # 9x9 board destinations
NUM_WALL_ACTIONS = 128  # 8x8 * 2 orientations
NUM_ACTIONS = NUM_PAWN_ACTIONS + NUM_WALL_ACTIONS  # 209 total

from window import Window
from player import Players
from wall import Walls
from colors import Colors
from protocol import GameState, PlayerState, WallState


class Sample:
    """Parsed training sample."""
    def __init__(self, state_tensor, policy, value):
        self.state_tensor = state_tensor  # (6, 9, 9) numpy array
        self.policy = policy              # (209,) numpy array
        self.value = value                # float

        # These will be set by the loader
        self.p1_row = 0
        self.p1_col = 0
        self.p2_row = 0
        self.p2_col = 0
        self.p1_fences = 0
        self.p2_fences = 0
        self.current_player = 0

        # These will be set from the raw data
        self.h_walls = []
        self.v_walls = []

    def _extract_walls(self, channel: np.ndarray) -> List[Tuple[int, int]]:
        """Extract wall positions from channel."""
        walls = []
        for r in range(8):
            for c in range(8):
                if channel[r, c] == 1.0:
                    walls.append((c, r))
        return walls

    def to_gamestate(self) -> GameState:
        """Convert sample to GameState for GUI rendering."""
        # Positions are already in protocol format (y=0 is top)
        # GUI expects y=0 at top, y=8 at bottom, which matches protocol
        players = [
            PlayerState(x=self.p1_col, y=self.p1_row, walls=self.p1_fences, name="P1"),
            PlayerState(x=self.p2_col, y=self.p2_row, walls=self.p2_fences, name="P2")
        ]

        walls = []
        for x, y in self.h_walls:
            walls.append(WallState(x=x, y=7-y, orientation="h"))
        for x, y in self.v_walls:
            walls.append(WallState(x=x, y=7-y, orientation="v"))

        return GameState(
            players=players,
            walls=walls,
            current_player=self.current_player,
            score=self.value,
            winner=None
        )

    def get_top_policy_moves(self, k=10) -> List[Tuple[str, int, int, float]]:
        """Get top k moves by policy value."""
        # Get indices sorted by policy value (descending)
        top_indices = np.argsort(self.policy)[::-1][:k]

        moves = []
        for idx in top_indices:
            prob = self.policy[idx]
            if prob < 0.001:  # Skip very low probability moves
                continue

            if idx < NUM_PAWN_ACTIONS:
                # Pawn move: row * 9 + col
                row = idx // 9
                col = idx % 9
                moves.append(("pawn", col, row, prob))
            else:
                # Wall move
                wall_idx = idx - NUM_PAWN_ACTIONS
                if wall_idx < 64:
                    # Horizontal wall
                    row = wall_idx // 8
                    col = wall_idx % 8
                    moves.append(("h_wall", col, row, prob))
                else:
                    # Vertical wall
                    wall_idx -= 64
                    row = wall_idx // 8
                    col = wall_idx % 8
                    moves.append(("v_wall", col, row, prob))

        return moves


def load_samples_direct(samples_path: str) -> List[Sample]:
    """Load all samples from a .qsamples file by reading the raw binary."""
    print(f"Loading samples from {samples_path}...")
    samples = []

    # Read header
    with open(samples_path, 'rb') as f:
        header_data = f.read(64)
        magic, version, flags, sample_count = struct.unpack('<IHHI', header_data[:12])

        print(f"File info: magic={hex(magic)}, version={version}, samples={sample_count}")

        # Read each sample (24 + 836 + 4 = 864 bytes each)
        for i in range(sample_count):
            sample_data = f.read(864)
            if len(sample_data) < 864:
                break

            # Parse CompactState (24 bytes)
            p1_row, p1_col, p2_row, p2_col = struct.unpack('BBBB', sample_data[0:4])
            p1_fences, p2_fences = struct.unpack('BB', sample_data[4:6])
            sample_flags = sample_data[6]
            fences_h, fences_v = struct.unpack('<QQ', sample_data[8:24])

            # Parse policy (209 floats = 836 bytes)
            policy = np.frombuffer(sample_data[24:24+209*4], dtype=np.float32).copy()

            # Parse value (1 float = 4 bytes)
            value = struct.unpack('<f', sample_data[24+209*4:24+209*4+4])[0]

            # Build state tensor from compact state
            state_tensor = np.zeros((6, 9, 9), dtype=np.float32)

            # FLAG_P1_TO_MOVE = 0x04
            is_p1_turn = (sample_flags & 0x04) != 0

            # Determine current player and opponent
            if is_p1_turn:
                my_row, my_col, my_fences = p1_row, p1_col, p1_fences
                opp_row, opp_col, opp_fences = p2_row, p2_col, p2_fences
                current_player = 0
            else:
                my_row, my_col, my_fences = p2_row, p2_col, p2_fences
                opp_row, opp_col, opp_fences = p1_row, p1_col, p1_fences
                current_player = 1

            # Channel 0: Current player's pawn
            state_tensor[0, my_row, my_col] = 1.0

            # Channel 1: Opponent's pawn
            state_tensor[1, opp_row, opp_col] = 1.0

            # Channel 2: Horizontal walls
            for r in range(8):
                for c in range(8):
                    if (fences_h >> (r * 8 + c)) & 1:
                        state_tensor[2, r, c] = 1.0

            # Channel 3: Vertical walls
            for r in range(8):
                for c in range(8):
                    if (fences_v >> (r * 8 + c)) & 1:
                        state_tensor[3, r, c] = 1.0

            # Channel 4: Current player's fences
            state_tensor[4, :, :] = my_fences / 10.0

            # Channel 5: Opponent's fences
            state_tensor[5, :, :] = opp_fences / 10.0

            sample = Sample(state_tensor, policy, value)
            sample.current_player = current_player  # Override with correct value
            sample.p1_row, sample.p1_col = p1_row, p1_col
            sample.p2_row, sample.p2_col = p2_row, p2_col
            sample.p1_fences, sample.p2_fences = p1_fences, p2_fences

            # Extract walls from tensor
            sample.h_walls = sample._extract_walls(state_tensor[2])
            sample.v_walls = sample._extract_walls(state_tensor[3])

            samples.append(sample)

    print(f"Loaded {len(samples)} samples")
    return samples


class SamplesViewer:
    """GUI for viewing training samples."""

    def __init__(self, samples: List[Sample]):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.win = Window()
        self.coords = self.win.coords
        self.players = Players(2, self.coords)
        self.walls = Walls()
        self.samples = samples
        self.current_index = 0
        self.running = True

        # Display first sample
        self.update_display()

    def update_display(self):
        """Update GUI to show current sample."""
        if not self.samples:
            return

        sample = self.samples[self.current_index]
        gs = sample.to_gamestate()

        # Print stats to terminal
        print("\n" + "=" * 80)
        print(f"Sample {self.current_index + 1}/{len(self.samples)}")
        print("=" * 80)
        print(f"Current player: P{sample.current_player + 1}")
        print(f"Value: {sample.value:.4f} (from current player's perspective)")
        print(f"P1 position: ({sample.p1_col}, {sample.p1_row}), fences: {sample.p1_fences}")
        print(f"P2 position: ({sample.p2_col}, {sample.p2_row}), fences: {sample.p2_fences}")
        print(f"Horizontal walls: {len(sample.h_walls)}")
        print(f"Vertical walls: {len(sample.v_walls)}")
        print()
        print("Top 10 policy moves:")
        for i, (move_type, x, y, prob) in enumerate(sample.get_top_policy_moves(10), 1):
            print(f"  {i}. {move_type:8s} at ({x}, {y}): {prob:.4f} ({prob*100:.2f}%)")
        print("=" * 80)
        print("Navigation: Left/Right arrows (or A/D), Home/End, PgUp/PgDn, Q to quit")

        # Reset GUI state
        self.coords.reset()
        for i, ps in enumerate(gs.players):
            p = self.players.players[i]
            gui_y = 8 - ps.y
            p.coord = self.coords.find_coord(ps.x, gui_y)
            p.coord.is_occuped = True
            p.walls_remain = ps.walls
            p.set_name(ps.name)

        self.walls.reset()
        for ws in gs.walls:
            gui_y = 7 - ws.y
            c1 = self.coords.find_coord(ws.x, gui_y)
            c1.link_coord()

            if ws.orientation == "h":
                c2 = self.coords.find_coord(ws.x, gui_y + 1)
            else:
                c2 = self.coords.find_coord(ws.x + 1, gui_y)

            c2.link_coord()
            from wall import Wall
            w = Wall(c1, c2, self.win)
            w.set_color(Colors.black)
            self.walls.add_wall(w)

        # Update info text
        self.win.update_info(f"Sample {self.current_index + 1}/{len(self.samples)} - "
                           f"Use arrow keys to navigate, Q to quit")

    def handle_input(self):
        """Handle keyboard input for navigation."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    self.running = False

                elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                    # Previous sample
                    if self.current_index > 0:
                        self.current_index -= 1
                        self.update_display()

                elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    # Next sample
                    if self.current_index < len(self.samples) - 1:
                        self.current_index += 1
                        self.update_display()

                elif event.key == pygame.K_HOME:
                    # First sample
                    self.current_index = 0
                    self.update_display()

                elif event.key == pygame.K_END:
                    # Last sample
                    self.current_index = len(self.samples) - 1
                    self.update_display()

                elif event.key == pygame.K_PAGEUP:
                    # Jump back 10 samples
                    self.current_index = max(0, self.current_index - 10)
                    self.update_display()

                elif event.key == pygame.K_PAGEDOWN:
                    # Jump forward 10 samples
                    self.current_index = min(len(self.samples) - 1, self.current_index + 10)
                    self.update_display()

    def render(self):
        """Render current sample."""
        pos = pygame.mouse.get_pos()
        sample = self.samples[self.current_index]
        self.win.redraw_window(self.players, self.walls, pos,
                              score=sample.value, current_player=sample.current_player)

    def run(self):
        """Main viewer loop."""
        while self.running:
            self.clock.tick(40)
            self.handle_input()
            self.render()

        pygame.quit()


def main(samples_path: str):
    """Entry point for samples viewer."""
    samples = load_samples_direct(samples_path)

    if not samples:
        print("No samples loaded!")
        return

    viewer = SamplesViewer(samples)
    viewer.run()
