"""Pathfinding for wall validation in Quoridor."""
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder


class PathFinder:
    def __init__(self):
        self.finder = AStarFinder()
        self.side = 17
        self.matrix = [[1]*self.side for _ in range(self.side)]
        self.make_walls()

    def make_walls(self):
        for i in range(self.side):
            if i % 2 == 1:
                for j in range(self.side):
                    if j % 2 == 1:
                        self.matrix[i][j] = 0

    def pos_player_in_grid(self, player):
        return (2*player.coord.y, 2*player.coord.x)

    def add_wall(self, wall):
        if wall.orient == "e":
            y, x = 2*wall.coord1.x + 1, 2*wall.coord1.y
            for i in range(3):
                self.matrix[x + i][y] = 0
        elif wall.orient == "s":
            y, x = 2*wall.coord1.x, 2*wall.coord1.y + 1
            for i in range(3):
                self.matrix[x][y + i] = 0

    def remove_wall(self, wall):
        if wall.orient == "e":
            y, x = 2*wall.coord1.x + 1, 2*wall.coord1.y
            for i in range(3):
                self.matrix[x + i][y] = 1
        elif wall.orient == "s":
            y, x = 2*wall.coord1.x, 2*wall.coord1.y + 1
            for i in range(3):
                self.matrix[x][y + i] = 1

    def find_path(self, player, show=False):
        x, y = self.pos_player_in_grid(player)
        grid = Grid(matrix=self.matrix)
        if player.orient == "north":
            x_end = self.side - 1
            for y_end in range(0, self.side, 2):
                start = grid.node(y, x)
                end = grid.node(y_end, x_end)
                path, runs = self.finder.find_path(start, end, grid)
                if path != []:
                    return True
                grid.cleanup()

        elif player.orient == "east":
            y_end = 0
            for x_end in range(0, self.side, 2):
                start = grid.node(y, x)
                end = grid.node(y_end, x_end)
                path, runs = self.finder.find_path(start, end, grid)
                if path != []:
                    return True
                grid.cleanup()

        elif player.orient == "south":
            x_end = 0
            for y_end in range(0, self.side, 2):
                start = grid.node(y, x)
                end = grid.node(y_end, x_end)
                path, runs = self.finder.find_path(start, end, grid)
                if path != []:
                    return True
                grid.cleanup()

        elif player.orient == "west":
            y_end = self.side - 1
            for x_end in range(0, self.side, 2):
                start = grid.node(y, x)
                end = grid.node(y_end, x_end)
                path, runs = self.finder.find_path(start, end, grid)
                if path != []:
                    return True
                grid.cleanup()

        return False

    def play_wall(self, wall, players):
        """Return True if the wall doesn't block players."""
        self.add_wall(wall)
        for p in players.players:
            if not self.find_path(p):
                self.remove_wall(wall)
                return False
        return True

    def reset(self):
        self.matrix.clear()
        self.matrix = [[1]*self.side for _ in range(self.side)]
        self.make_walls()
