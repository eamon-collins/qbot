#!/usr/bin/env python3
"""
Test client for Quoridor GUI WebSocket server.

This simulates what the C++ bot would do:
1. Connect to the GUI
2. Send initial game state
3. Request moves from the human player
4. Make bot moves
5. Update game state after each move
"""
import asyncio
import json
import random
import websockets


class TestBot:
    def __init__(self):
        self.players = [
            {"x": 4, "y": 0, "walls": 10, "name": "Human"},
            {"x": 4, "y": 8, "walls": 10, "name": "Bot"}
        ]
        self.walls = []
        self.current_player = 0
        self.winner = None

    def get_gamestate(self, score=0.0):
        return {
            "type": "gamestate",
            "players": self.players,
            "walls": self.walls,
            "current_player": self.current_player,
            "score": score,
            "winner": self.winner
        }

    def apply_move(self, move):
        """Apply a move from the human player."""
        if move["move_type"] == "pawn":
            self.players[0]["x"] = move["x"]
            self.players[0]["y"] = move["y"]
            # Check win condition
            if move["y"] == 8:
                self.winner = 0
        elif move["move_type"] == "wall":
            self.walls.append({
                "x": move["x"],
                "y": move["y"],
                "orientation": move["orientation"]
            })
            self.players[0]["walls"] -= 1

    def make_bot_move(self):
        """Make a simple bot move (just move toward goal)."""
        p = self.players[1]
        if p["y"] > 0:
            p["y"] -= 1
        # Check win condition
        if p["y"] == 0:
            self.winner = 1


async def run_test_client(uri: str):
    print(f"Connecting to {uri}...")
    async with websockets.connect(uri) as ws:
        print("Connected!")
        bot = TestBot()

        # Send start message
        await ws.send(json.dumps({
            "type": "start",
            "player_names": ["Human", "TestBot"]
        }))
        await asyncio.sleep(0.1)

        # Send initial game state
        await ws.send(json.dumps(bot.get_gamestate()))
        await asyncio.sleep(0.1)

        # Game loop
        while bot.winner is None:
            # Request move from human player
            print("Requesting move from human player...")
            await ws.send(json.dumps({"type": "request_move", "player": 0}))

            # Wait for response
            response = await ws.recv()
            data = json.loads(response)

            if data.get("type") == "quit":
                print("Human quit the game")
                break

            if data.get("type") == "move":
                print(f"Human move: {data}")
                bot.apply_move(data)
                bot.current_player = 1

                # Send updated state
                await ws.send(json.dumps(bot.get_gamestate(score=random.uniform(-0.5, 0.5))))
                await asyncio.sleep(0.5)

                if bot.winner is not None:
                    break

                # Bot's turn
                print("Bot making move...")
                bot.make_bot_move()
                bot.current_player = 0

                # Send updated state
                score = 0.3 if bot.winner == 1 else random.uniform(-0.3, 0.3)
                await ws.send(json.dumps(bot.get_gamestate(score=score)))
                await asyncio.sleep(0.3)

        # Game over
        if bot.winner is not None:
            winner_name = bot.players[bot.winner]["name"]
            print(f"Game over! {winner_name} wins!")
            # Send final state
            await ws.send(json.dumps(bot.get_gamestate()))

        await asyncio.sleep(2)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test client for Quoridor GUI")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    uri = f"ws://{args.host}:{args.port}"
    asyncio.run(run_test_client(uri))


if __name__ == "__main__":
    main()
