#!/usr/bin/env python3
"""
Samples viewer for Quoridor training data.

Loads .qsamples files and allows browsing through training samples,
or .qbot tree files and plays through game paths.

Usage:
    # View training samples from .qsamples file
    python gui/samples_viewer.py samples/tree_0.qsamples

    # View game path from tree file (auto-detects .qbot extension)
    python gui/samples_viewer.py samples/tree_3.qbot

    # Explicitly specify tree mode and choose which game path to view
    python gui/samples_viewer.py --tree samples/tree_3.qbot --path 1

For tree files, the viewer will:
  - Extract all game paths (sequences marked as on_game_path)
  - Display each node along the path with full statistics:
    * Visit count
    * Total value (accumulated backpropagated value)
    * Q-value / NN value (mean value in relative perspective)
    * Terminal value (if node is terminal)
    * Prior probability
    * Game outcome (from P1's perspective)
"""
import os
import sys
import struct
import numpy as np
import pygame
import argparse
from typing import List, Tuple, Optional, Dict
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


# =============================================================================
# Tree File Support (.qbot files)
# =============================================================================

class TreeNode:
    """Parsed tree node from .qbot file."""
    def __init__(self, data: bytes, index: int):
        # Parse SerializedNode (56 bytes)
        # Layout: first_child(4), next_sibling(4), parent(4),
        #         p1(3), p2(3), move_data(2), flags(1), reserved(1), ply(2),
        #         fences_h(8), fences_v(8), visits(4), total_value(4), prior(4), terminal_value(4)
        (self.first_child, self.next_sibling, self.parent,
         self.p1_row, self.p1_col, self.p1_fences,
         self.p2_row, self.p2_col, self.p2_fences,
         self.move_data, self.flags, self.reserved, self.ply,
         self.fences_horizontal, self.fences_vertical,
         self.visits, self.total_value, self.prior, self.terminal_value) = struct.unpack(
            '<III BBBBBB HBBH QQ Ifff', data)

        self.index = index

        # Compute derived values
        self.nn_value = self.total_value / self.visits if self.visits > 0 else 0.0  # Q-value
        self.is_on_game_path = (self.flags & 0x08) != 0
        self.is_terminal = (self.flags & 0x02) != 0
        self.is_p1_to_move = (self.flags & 0x04) != 0

    def is_null(self, value: int) -> bool:
        """Check if a node index is NULL_NODE."""
        return value == 0xFFFFFFFF

    def has_children(self) -> bool:
        return not self.is_null(self.first_child)

    def to_sample(self) -> Sample:
        """Convert tree node to Sample for display."""
        # Build state tensor
        state_tensor = np.zeros((6, 9, 9), dtype=np.float32)

        # Determine current player and opponent
        if self.is_p1_to_move:
            my_row, my_col, my_fences = self.p1_row, self.p1_col, self.p1_fences
            opp_row, opp_col, opp_fences = self.p2_row, self.p2_col, self.p2_fences
            current_player = 0
        else:
            my_row, my_col, my_fences = self.p2_row, self.p2_col, self.p2_fences
            opp_row, opp_col, opp_fences = self.p1_row, self.p1_col, self.p1_fences
            current_player = 1

        # Channel 0: Current player's pawn
        state_tensor[0, my_row, my_col] = 1.0

        # Channel 1: Opponent's pawn
        state_tensor[1, opp_row, opp_col] = 1.0

        # Channel 2: Horizontal walls
        for r in range(8):
            for c in range(8):
                if (self.fences_horizontal >> (r * 8 + c)) & 1:
                    state_tensor[2, r, c] = 1.0

        # Channel 3: Vertical walls
        for r in range(8):
            for c in range(8):
                if (self.fences_vertical >> (r * 8 + c)) & 1:
                    state_tensor[3, r, c] = 1.0

        # Channel 4: Current player's fences
        state_tensor[4, :, :] = my_fences / 10.0

        # Channel 5: Opponent's fences
        state_tensor[5, :, :] = opp_fences / 10.0

        # Create sample with placeholder policy (we don't have visit distribution in tree format)
        policy = np.ones(NUM_ACTIONS, dtype=np.float32) / NUM_ACTIONS

        sample = Sample(state_tensor, policy, self.nn_value)
        sample.current_player = current_player
        sample.p1_row, sample.p1_col = self.p1_row, self.p1_col
        sample.p2_row, sample.p2_col = self.p2_row, self.p2_col
        sample.p1_fences, sample.p2_fences = self.p1_fences, self.p2_fences

        # Extract walls
        sample.h_walls = sample._extract_walls(state_tensor[2])
        sample.v_walls = sample._extract_walls(state_tensor[3])

        return sample


def load_tree_file(tree_path: str) -> Tuple[List[TreeNode], Optional[int]]:
    """Load tree from .qbot file."""
    print(f"Loading tree from {tree_path}...")

    with open(tree_path, 'rb') as f:
        # Read header (64 bytes)
        # struct TreeFileHeader: magic(4), version(2), flags(2), node_count(4), root_index(4), reserved1(4), timestamp(8), reserved[32]
        header_data = f.read(64)
        magic, version, flags, node_count, root_index = struct.unpack('<IHHII', header_data[:16])

        if magic != 0x51424F54:  # "QBOT"
            raise ValueError(f"Invalid tree file: bad magic {hex(magic)}")
        if version > 1:
            raise ValueError(f"Unsupported tree version: {version}")

        print(f"File info: magic={hex(magic)}, version={version}, nodes={node_count}, root={root_index}")

        # Read all nodes
        nodes = []
        for i in range(node_count):
            node_data = f.read(56)
            if len(node_data) < 56:
                break
            nodes.append(TreeNode(node_data, i))

        print(f"Loaded {len(nodes)} nodes")
        return nodes, root_index if root_index != 0xFFFFFFFF else None


def extract_game_paths(nodes: List[TreeNode], root_index: int) -> List[List[TreeNode]]:
    """Extract all game paths (sequences of nodes marked as on_game_path) from tree."""
    if root_index >= len(nodes):
        return []

    paths = []

    def dfs_path(node_idx: int, current_path: List[TreeNode]):
        """DFS to find all paths that reach terminal nodes."""
        if node_idx >= len(nodes):
            return

        node = nodes[node_idx]

        # Only follow nodes on game paths
        if not node.is_on_game_path:
            return

        current_path.append(node)

        # If terminal, save this path
        if node.is_terminal or not node.has_children():
            paths.append(current_path.copy())
            current_path.pop()
            return

        # Recurse to children that are on game path
        child_idx = node.first_child
        found_game_path_child = False
        while not node.is_null(child_idx):
            child = nodes[child_idx]
            if child.is_on_game_path:
                dfs_path(child_idx, current_path)
                found_game_path_child = True
            child_idx = child.next_sibling

        # If we didn't find any game path children, this is an end point
        if not found_game_path_child:
            paths.append(current_path.copy())

        current_path.pop()

    dfs_path(root_index, [])

    print(f"Found {len(paths)} game paths")
    for i, path in enumerate(paths):
        print(f"  Path {i+1}: {len(path)} nodes")

    return paths


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

    def __init__(self, samples: List[Sample], tree_nodes: Optional[List[TreeNode]] = None,
                 game_outcome: Optional[float] = None, all_paths: Optional[List[List[TreeNode]]] = None,
                 current_path_index: int = 0):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.win = Window()
        self.coords = self.win.coords
        self.players = Players(2, self.coords)
        self.walls = Walls()
        self.samples = samples
        self.tree_nodes = tree_nodes  # Original tree nodes if viewing tree
        self.game_outcome = game_outcome  # Game outcome if viewing a game path
        self.all_paths = all_paths  # All game paths (for cycling)
        self.current_path_index = current_path_index  # Which path we're viewing
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
        if self.tree_nodes:
            if self.all_paths:
                print(f"Game Path {self.current_path_index + 1}/{len(self.all_paths)} - "
                      f"Node {self.current_index + 1}/{len(self.samples)}")
            else:
                print(f"Tree Node {self.current_index + 1}/{len(self.samples)} (Game Path)")
        else:
            print(f"Sample {self.current_index + 1}/{len(self.samples)}")
        print("=" * 80)
        print(f"Current player: P{sample.current_player + 1}")

        # If viewing tree nodes, show additional stats
        if self.tree_nodes and self.current_index < len(self.tree_nodes):
            node = self.tree_nodes[self.current_index]
            print(f"Node stats:")
            print(f"  Visits: {node.visits}")
            print(f"  Total value: {node.total_value:.4f}")
            print(f"  Q-value (NN value): {node.nn_value:.4f} (relative perspective)")
            if node.is_terminal:
                print(f"  Terminal value: {node.terminal_value:.4f} (TERMINAL)")
            print(f"  Prior: {node.prior:.4f}")
            print(f"  Ply: {node.ply}")
            if self.game_outcome is not None:
                print(f"Game outcome: {self.game_outcome:+.2f} ({'P1 wins' if self.game_outcome > 0 else 'P2 wins' if self.game_outcome < 0 else 'Draw'})")
        else:
            print(f"Value: {sample.value:.4f} (from current player's perspective)")

        print(f"P1 position: ({sample.p1_col}, {sample.p1_row}), fences: {sample.p1_fences}")
        print(f"P2 position: ({sample.p2_col}, {sample.p2_row}), fences: {sample.p2_fences}")
        print(f"Horizontal walls: {len(sample.h_walls)}")
        print(f"Vertical walls: {len(sample.v_walls)}")

        # Only show policy for samples (tree nodes don't have good policy)
        if not self.tree_nodes:
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
        if self.tree_nodes:
            if self.all_paths:
                self.win.update_info(f"Path {self.current_path_index + 1}/{len(self.all_paths)} - "
                                   f"Node {self.current_index + 1}/{len(self.samples)} - "
                                   f"Arrow keys to navigate, Q to quit")
            else:
                self.win.update_info(f"Tree Node {self.current_index + 1}/{len(self.samples)} - "
                                   f"Use arrow keys to navigate, Q to quit")
        else:
            self.win.update_info(f"Sample {self.current_index + 1}/{len(self.samples)} - "
                               f"Use arrow keys to navigate, Q to quit")

    def switch_to_path(self, path_index: int):
        """Switch to a different game path."""
        if not self.all_paths or path_index >= len(self.all_paths):
            return

        self.current_path_index = path_index
        path = self.all_paths[path_index]

        # Determine game outcome from terminal node
        terminal_node = path[-1]
        if terminal_node.is_terminal:
            # terminal_value is in relative perspective, convert to absolute
            if terminal_node.is_p1_to_move:
                self.game_outcome = terminal_node.terminal_value
            else:
                self.game_outcome = -terminal_node.terminal_value
        else:
            self.game_outcome = None

        # Convert path nodes to samples
        self.samples = [node.to_sample() for node in path]
        self.tree_nodes = path
        self.current_index = 0
        self.update_display()

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
                    elif self.all_paths and self.current_path_index > 0:
                        # At beginning of current path, switch to previous path (at its end)
                        print(f"\n>>> Start of path {self.current_path_index + 1}, switching to path {self.current_path_index} <<<")
                        self.switch_to_path(self.current_path_index - 1)
                        self.current_index = len(self.samples) - 1  # Go to last node
                        self.update_display()

                elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    # Next sample
                    if self.current_index < len(self.samples) - 1:
                        self.current_index += 1
                        self.update_display()
                    elif self.all_paths and self.current_path_index < len(self.all_paths) - 1:
                        # At end of current path, switch to next path
                        print(f"\n>>> End of path {self.current_path_index + 1}, switching to path {self.current_path_index + 2} <<<")
                        self.switch_to_path(self.current_path_index + 1)

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


def main():
    """Entry point for samples viewer."""
    parser = argparse.ArgumentParser(description='View Quoridor training samples or game trees')
    parser.add_argument('file', help='Path to .qsamples or .qbot file')
    parser.add_argument('--tree', action='store_true', help='View .qbot tree file (play through game paths)')
    parser.add_argument('--path', type=int, default=0, help='Which game path to view (0-indexed, default: 0)')

    args = parser.parse_args()

    if args.tree or args.file.endswith('.qbot'):
        # Load tree file
        nodes, root_index = load_tree_file(args.file)
        if not nodes or root_index is None:
            print("No tree loaded or no root node!")
            return

        # Extract game paths
        paths = extract_game_paths(nodes, root_index)
        if not paths:
            print("No game paths found in tree!")
            return

        # Select which path to view
        path_index = args.path
        if path_index >= len(paths):
            print(f"Path index {path_index} out of range (0-{len(paths)-1})")
            return

        path = paths[path_index]
        print(f"\nStarting with game path {path_index + 1}/{len(paths)} ({len(path)} moves)")
        print("Use arrow keys to navigate. At path end, press right to view next path.")

        # Determine game outcome from terminal node
        terminal_node = path[-1]
        if terminal_node.is_terminal:
            # terminal_value is in relative perspective
            # Convert to absolute (P1's perspective)
            if terminal_node.is_p1_to_move:
                game_outcome = terminal_node.terminal_value
            else:
                game_outcome = -terminal_node.terminal_value
        else:
            game_outcome = None

        # Convert path nodes to samples for display
        samples = [node.to_sample() for node in path]

        viewer = SamplesViewer(samples, tree_nodes=path, game_outcome=game_outcome,
                              all_paths=paths, current_path_index=path_index)
        viewer.run()

    else:
        # Load samples file
        samples = load_samples_direct(args.file)

        if not samples:
            print("No samples loaded!")
            return

        viewer = SamplesViewer(samples)
        viewer.run()


if __name__ == '__main__':
    main()
