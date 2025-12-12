"""Player management for Quoridor GUI."""
import pygame
from colors import Colors


class Player:
    def __init__(self, num_player, walls_remain, orient, color, coord, radius=20):
        self.num_player = num_player
        self.orient = orient
        self.color = color
        self.coord = coord
        self.radius = radius
        self.name = ''
        self.walls_remain = walls_remain

    def set_name(self, name):
        if name != '':
            self.name = name

    def has_walls(self):
        return self.walls_remain > 0

    def draw(self, win):
        (x, y) = self.coord.middle
        pygame.draw.circle(win.win, self.color, (x, y), self.radius)
        font = pygame.font.SysFont("comicsans", 40)
        text = font.render(self.name[0] if self.name else str(self.num_player), 1, Colors.white)
        win.win.blit(text, (x - self.radius // 2, y - self.radius // 2))

    def has_win(self, coord):
        if self.orient == "north" and coord.y == 8:
            return True
        if self.orient == "east" and coord.x == 0:
            return True
        if self.orient == "south" and coord.y == 0:
            return True
        if self.orient == "west" and coord.x == 8:
            return True
        return False


class Players:
    def __init__(self, nb_players, coords):
        self.nb_players = nb_players
        if nb_players == 4:
            self.players = [
                Player(0, 5, "north", Colors.red, coords.find_coord(4, 0)),
                Player(1, 5, "east", Colors.blue, coords.find_coord(8, 4)),
                Player(2, 5, "south", Colors.green, coords.find_coord(4, 8)),
                Player(3, 5, "west", Colors.yellow, coords.find_coord(0, 4))]
        elif nb_players == 2:
            self.players = [
                Player(0, 10, "north", Colors.red, coords.find_coord(4, 0)),
                Player(1, 10, "south", Colors.green, coords.find_coord(4, 8))]
        elif nb_players == 3:
            self.players = [
                Player(0, 7, "north", Colors.red, coords.find_coord(4, 0)),
                Player(1, 7, "east", Colors.blue, coords.find_coord(8, 4)),
                Player(2, 7, "south", Colors.green, coords.find_coord(4, 8))]

    def draw(self, win):
        for p in self.players:
            if p.name != '':
                p.draw(win)

    def get_player(self, num_player):
        return self.players[num_player]

    def set_names(self, names):
        for player, name in zip(self.players, names):
            player.set_name(name)

    def reset(self, coords):
        if self.nb_players == 2:
            walls_remain = 10
        elif self.nb_players == 3:
            walls_remain = 7
        elif self.nb_players == 4:
            walls_remain = 5
        for p in self.players:
            if p.orient == "north":
                p.coord = coords.find_coord(4, 0)
            elif p.orient == "east":
                p.coord = coords.find_coord(8, 4)
            elif p.orient == "south":
                p.coord = coords.find_coord(4, 8)
            elif p.orient == "west":
                p.coord = coords.find_coord(0, 4)
            p.walls_remain = walls_remain
