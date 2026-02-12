#!/usr/bin/env python3
"""
Standalone Quoridor GUI with WebSocket server.

Starts a pygame window and WebSocket server. When a client connects,
the GUI displays the game state and sends player moves back to the client.

Usage:
    python main.py [--host HOST] [--port PORT]

The GUI acts as the display and input interface for player 0 (human).
The connected client (e.g., C++ bot) controls the game flow and player 1.
"""
import os
import sys
import asyncio
import argparse
import json
from typing import Optional
from dataclasses import dataclass

# Disable OpenGL to avoid GLX errors in environments without full GL support
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
os.environ['SDL_RENDER_DRIVER'] = 'software'
os.environ['SDL_VIDEODRIVER'] = 'x11'
# Disable GLX (OpenGL for X11) explicitly
os.environ['SDL_VIDEO_X11_FORCE_EGL'] = '0'
os.environ['SDL_FRAMEBUFFER_ACCELERATION'] = '0'

import pygame

import websockets.asyncio.server

from window import Window, pos_in_rect
from player import Players, Player
from wall import Walls, Wall
from pathfinder import PathFinder
from colors import Colors
from protocol import GameState, PlayerState, WallState, MoveMessage, parse_message


@dataclass
class GUIState:
    """Shared state between pygame loop and websocket handler."""
    connected: bool = False
    waiting_for_move: bool = False
    pending_move: Optional[MoveMessage] = None
    game_state: Optional[GameState] = None
    should_quit: bool = False
    player_names: list[str] = None
    active_handler_id: int = 0  # Incremented on each new connection

    def __post_init__(self):
        if self.player_names is None:
            self.player_names = ["Human", "Bot"]


class QuoridorGUI:
    def __init__(self):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.win = Window()
        self.coords = self.win.coords
        self.players = Players(2, self.coords)
        self.walls = Walls()
        self.pf = PathFinder()
        self.state = GUIState()
        self.current_player = 0
        self.score = 0.0

    def apply_gamestate(self, gs: GameState):
        """Update GUI state from received GameState."""
        self.state.game_state = gs
        self.current_player = gs.current_player
        self.score = gs.score

        # Reset and rebuild players
        self.coords.reset()
        for i, ps in enumerate(gs.players):
            p = self.players.players[i]
            # Coordinate system: GUI y is inverted from protocol
            gui_y = 8 - ps.y
            p.coord = self.coords.find_coord(ps.x, gui_y)
            p.coord.is_occuped = True
            p.walls_remain = ps.walls
            p.set_name(ps.name)

        # Reset and rebuild walls
        self.walls.reset()
        self.pf.reset()
        for ws in gs.walls:
            # Protocol orientation: "h" = horizontal (blocks vertical), "v" = vertical (blocks horizontal)
            # GUI orientation: "s" = south (horizontal), "e" = east (vertical)
            gui_y = 7 - ws.y  # Adjust for GUI coordinate system
            c1 = self.coords.find_coord(ws.x, gui_y)
            c1.link_coord()

            if ws.orientation == "h":
                # Horizontal wall - c2 below c1 (same column) so Wall uses orient="s"
                c2 = self.coords.find_coord(ws.x, gui_y + 1)
            else:
                # Vertical wall - c2 right of c1 (same row) so Wall uses orient="e"
                c2 = self.coords.find_coord(ws.x + 1, gui_y)

            c2.link_coord()
            w = Wall(c1, c2, self.win)
            w.set_color(Colors.black)
            self.walls.add_wall(w)
            self.pf.add_wall(w)

        # Update info text
        if gs.winner is not None:
            winner_name = gs.players[gs.winner].name
            self.win.update_info(f"{winner_name} wins!", Colors.green if gs.winner == 0 else Colors.red)
        elif gs.current_player == 0:
            self.win.update_info("Your turn - make a move")
        else:
            self.win.update_info("Waiting for opponent...")

    def handle_input(self) -> Optional[MoveMessage]:
        """Process pygame events and return move if player made one."""
        pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.state.should_quit = True
                return None

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.win.button_quit.click(pos):
                    self.state.should_quit = True
                    return None

                if not self.state.waiting_for_move:
                    continue

                # Check for pawn move (click on board square)
                for c in self.coords.coords:
                    if pos_in_rect(c.rect, pos):
                        # Convert GUI coords back to protocol coords
                        proto_y = 8 - c.y
                        return MoveMessage(move_type="pawn", x=c.x, y=proto_y)

                    # Check for wall placement
                    wall_east = c.wall_east
                    wall_south = c.wall_south
                    for w in [wall_east, wall_south]:
                        if w is not None and pos_in_rect(w.rect_small, pos) and self.walls.can_add(w):
                            if self.pf.play_wall(w, self.players):
                                # Convert to protocol format
                                proto_y = 7 - c.y
                                if w == c.wall_south:
                                    orientation = "h"
                                else:
                                    orientation = "v"
                                self.pf.remove_wall(w)  # Don't commit yet, client will send updated state
                                return MoveMessage(move_type="wall", x=c.x, y=proto_y, orientation=orientation)

            elif event.type == pygame.KEYDOWN and self.state.waiting_for_move:
                # Keyboard movement for pawn
                p = self.players.players[0]
                dx, dy = 0, 0
                if event.key in (pygame.K_LEFT, pygame.K_a):
                    dx = -1
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    dx = 1
                elif event.key in (pygame.K_UP, pygame.K_w):
                    dy = -1  # GUI y is inverted
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    dy = 1

                if dx != 0 or dy != 0:
                    new_x = p.coord.x + dx
                    new_y = p.coord.y + dy
                    if 0 <= new_x <= 8 and 0 <= new_y <= 8:
                        proto_y = 8 - new_y
                        return MoveMessage(move_type="pawn", x=new_x, y=proto_y)

        return None

    def render(self):
        """Draw the current game state."""
        pos = pygame.mouse.get_pos()
        self.win.redraw_window(self.players, self.walls, pos, self.score, self.current_player)

    def run_frame(self) -> Optional[MoveMessage]:
        """Run one frame of the game loop. Returns move if player made one."""
        self.clock.tick(40)
        move = self.handle_input()
        self.render()
        return move


async def websocket_handler(websocket, gui: QuoridorGUI):
    """Handle WebSocket connection from game client."""
    # Assign this handler a unique ID and invalidate any previous handler
    gui.state.active_handler_id += 1
    my_handler_id = gui.state.active_handler_id
    gui.state.waiting_for_move = False  # Reset state from any previous handler

    print(f"Client connected from {websocket.remote_address} (handler {my_handler_id})")
    gui.state.connected = True
    gui.win.update_info("Client connected! Waiting for game start...")

    def is_active():
        """Check if this handler is still the active one."""
        return gui.state.active_handler_id == my_handler_id and not gui.state.should_quit

    try:
        async for message in websocket:
            if not is_active():
                print(f"Handler {my_handler_id} superseded, exiting")
                break

            data = parse_message(message)
            msg_type = data.get("type")

            if msg_type == "start":
                # Game starting, set player names
                names = data.get("player_names", ["Human", "Bot"])
                gui.state.player_names = names
                gui.players.set_names(names)
                gui.win.update_info("Game started!")
                print(f"Game started: {names[0]} vs {names[1]}")

            elif msg_type == "gamestate":
                # Full game state update
                gs = GameState(
                    players=[PlayerState(**p) for p in data["players"]],
                    walls=[WallState(**w) for w in data["walls"]],
                    current_player=data["current_player"],
                    score=data.get("score", 0.0),
                    winner=data.get("winner")
                )
                gui.apply_gamestate(gs)
                gui.render()  # Render immediately (but don't consume events)

            elif msg_type == "request_move":
                # Client is asking GUI for a move
                player = data.get("player", 0)
                if player == 0:
                    gui.state.waiting_for_move = True
                    gui.win.update_info("Your turn - make a move")

                    # Wait for player to make a move
                    while gui.state.waiting_for_move and is_active():
                        move = gui.run_frame()
                        if move:
                            gui.state.waiting_for_move = False
                            await websocket.send(move.to_json())
                            gui.win.update_info("Move sent, waiting...")
                            break
                        await asyncio.sleep(0.001)

                    if gui.state.should_quit:
                        await websocket.send(json.dumps({"type": "quit"}))
                        break

            if gui.state.should_quit:
                break

    except websockets.exceptions.ConnectionClosed:
        print(f"Client disconnected (handler {my_handler_id})")
    finally:
        # Only clear connected state if we're still the active handler
        if gui.state.active_handler_id == my_handler_id:
            gui.state.connected = False
            gui.state.waiting_for_move = False
            gui.win.update_info("Disconnected. Waiting for connection...")
        print(f"Handler {my_handler_id} exiting")


async def run_server(host: str, port: int, gui: QuoridorGUI):
    """Run the WebSocket server."""
    # ping_interval sends pings every 5s, ping_timeout closes connection if no pong within 10s
    async with websockets.asyncio.server.serve(lambda ws: websocket_handler(ws, gui), host, port,
                                                ping_interval=5, ping_timeout=10):
        print(f"WebSocket server running on ws://{host}:{port}")
        gui.win.update_info(f"Waiting for connection on port {port}...")

        # Keep rendering while waiting for connections
        while not gui.state.should_quit:
            if not gui.state.connected:
                gui.run_frame()
            await asyncio.sleep(0.01)


async def main_async(host: str, port: int):
    """Main async entry point."""
    gui = QuoridorGUI()
    gui.players.set_names(["Human", "Bot"])

    try:
        await run_server(host, port, gui)
    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Quoridor GUI with WebSocket server")
    parser.add_argument("--host", default="localhost", help="Host to bind to (default: localhost)")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on (default: 8765)")
    parser.add_argument("--samples", type=str, help="View training samples from .qsamples file")
    args = parser.parse_args()

    # If --samples flag is provided, run samples viewer instead
    if args.samples:
        from samples_viewer import main as samples_main
        samples_main(args.samples)
    else:
        asyncio.run(main_async(args.host, args.port))


if __name__ == "__main__":
    main()
