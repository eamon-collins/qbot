"""Board coordinate system for Quoridor GUI."""
import pygame
from wall import Wall


class Coord:
    def __init__(self, x, y, win, coords):
        self.win = win
        self.coords = coords

        self.x = x
        self.y = y
        self.tuple = (x, y)
        self.is_occuped = False

        self.top_left = self.make_top_left()
        self.middle = self.make_middle()
        self.rect = self.make_rect()

        self.north = None
        self.east = None
        self.south = None
        self.west = None
        self.wall_east = None
        self.wall_south = None

    def coord_north(self):
        if self.y - 1 >= 0:
            return self.coords.find_coord(self.x, self.y - 1)
        return None

    def coord_east(self):
        if self.x + 1 <= 8:
            return self.coords.find_coord(self.x + 1, self.y)
        return None

    def coord_south(self):
        if self.y + 1 <= 8:
            return self.coords.find_coord(self.x, self.y + 1)
        return None

    def coord_west(self):
        if self.x - 1 >= 0:
            return self.coords.find_coord(self.x - 1, self.y)
        return None

    def make_top_left(self):
        win = self.win
        x = ((win.wall_width + win.case_side)*self.x
             + win.wall_width + win.top_left[0])
        y = ((win.wall_width + win.case_side)*self.y
             + win.wall_width + win.top_left[1])
        return (x, y)

    def make_middle(self):
        win = self.win
        x = ((win.wall_width + win.case_side)*self.x
             + (win.wall_width + win.case_side // 2)
             + win.top_left[0])
        y = ((win.wall_width + win.case_side)*self.y
             + (win.wall_width + win.case_side // 2)
             + win.top_left[1])
        return (x, y)

    def make_rect(self):
        win = self.win
        x, y = self.top_left
        return (x, y, win.case_side, win.case_side)

    def make_wall_east(self):
        if self.east is not None and self.y != 8:
            return Wall(self, self.east, self.win)
        return None

    def make_wall_south(self):
        if self.south is not None and self.x != 8:
            return Wall(self, self.south, self.win)
        return None

    def link_coord(self):
        self.north = self.coord_north()
        self.east = self.coord_east()
        self.south = self.coord_south()
        self.west = self.coord_west()

    def make_walls(self):
        self.wall_east = self.make_wall_east()
        self.wall_south = self.make_wall_south()

    def make_cross_walls(self):
        if self.wall_east is not None:
            self.wall_east.make_cross_wall()
        if self.wall_south is not None:
            self.wall_south.make_cross_wall()

    def same_row(self, other):
        return self.y == other.y

    def same_column(self, other):
        return self.x == other.x

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def draw(self, color):
        pygame.draw.rect(self.win.win, color, self.rect)


class Coords:
    def __init__(self, win):
        self.win = win
        self.coords = self.make_coords()
        self.link_coords()
        self.make_walls()

    def make_coords(self):
        coords = []
        for x in range(9):
            for y in range(9):
                coords.append(Coord(x, y, self.win, self))
        return coords

    def link_coords(self):
        for c in self.coords:
            c.link_coord()

    def make_walls(self):
        for c in self.coords:
            c.make_walls()
        for c in self.coords:
            c.make_cross_walls()

    def find_coord(self, x, y):
        return self.coords[x * 9 + y]

    def reset(self):
        for c in self.coords:
            c.is_occuped = False
