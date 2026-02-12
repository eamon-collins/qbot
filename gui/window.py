"""Main window rendering for Quoridor GUI."""
import pygame
from colors import Colors
from widgets import Text, Button
from coord import Coords


def pos_in_rect(rect, pos):
    pos_x, pos_y = pos
    x, y, width, height = rect
    return (x <= pos_x <= x + width
            and y <= pos_y <= y + height)


class Window:
    def __init__(self, width=1000, height=830, case_side=65, wall_width=15,
                 title="Quoridor", bgcolor=Colors.white):
        self.width = width
        self.height = height
        self.case_side = case_side
        self.wall_width = wall_width
        self.bgcolor = bgcolor
        self.top_left = (20, 20)
        self.side_board = 9*self.case_side + 10*self.wall_width
        # Use software rendering to avoid GLX/OpenGL issues
        self.win = pygame.display.set_mode((width, height), pygame.DOUBLEBUF)
        pygame.display.set_caption(title)
        self.button_quit = Button("Quit", self.side_board + 60,
                                  self.side_board - 50, Colors.red)
        self.buttons = [self.button_quit]
        self.title = Text("Quoridor", Colors.black, size=50)
        self.info = Text("Waiting for connection...", Colors.black, size=35)
        self.coords = Coords(self)

    def update_info(self, text, color=None):
        self.info.text = text
        if color is not None:
            self.info.color = color

    def draw_game_board(self, pos):
        for c in self.coords.coords:
            rect = c.rect
            if pos_in_rect(rect, pos):
                color = Colors.grey_dark
            else:
                color = Colors.grey
                wall_east = c.wall_east
                wall_south = c.wall_south
                if wall_east and pos_in_rect(wall_east.rect_small, pos):
                    wall_east.draw(Colors.grey_dark)
                elif wall_south and pos_in_rect(wall_south.rect_small, pos):
                    wall_south.draw(Colors.grey_dark)
            c.draw(color)

    def draw_finish_lines(self, players):
        for p in players.players:
            if p.name != '':
                if p.orient == "north":
                    pygame.draw.line(
                        self.win, p.color,
                        (self.top_left[0], self.top_left[1] + self.side_board),
                        (self.top_left[0] + self.side_board,
                         self.top_left[1] + self.side_board),
                        self.wall_width)
                elif p.orient == "east":
                    pygame.draw.line(
                        self.win, p.color,
                        (self.top_left[0], self.top_left[1]),
                        (self.top_left[0], self.top_left[1] + self.side_board),
                        self.wall_width)
                elif p.orient == "south":
                    pygame.draw.line(
                        self.win, p.color,
                        (self.top_left[0], self.top_left[1]),
                        (self.top_left[0] + self.side_board, self.top_left[1]),
                        self.wall_width)
                elif p.orient == "west":
                    pygame.draw.line(
                        self.win, p.color,
                        (self.top_left[0] + self.side_board, self.top_left[1]),
                        (self.top_left[0] + self.side_board,
                         self.top_left[1] + self.side_board),
                        self.wall_width)

    def draw_right_panel(self, players, current_player=None):
        x, y = self.side_board + 50, 20
        self.title.draw(self.win, (x + 10, y))
        for p in players.players:
            if p.name != '':
                prefix = "> " if current_player == p.num_player else "  "
                text_p = Text(f"{prefix}{p.name}: {p.walls_remain} walls", p.color)
                text_p.draw(self.win, (x, y + 100*p.num_player + 100))

    def draw_buttons(self):
        for b in self.buttons:
            if b.show:
                b.draw(self.win)

    def draw_score_bar(self, score):
        SCORE_BAR_HEIGHT = 20
        SCORE_BAR_WIDTH = 160
        SCORE_BAR_Y = self.side_board - 100
        SCORE_BAR_X = self.side_board + 50
        surface = self.win

        pygame.draw.rect(surface, Colors.grey,
                        (SCORE_BAR_X, SCORE_BAR_Y, SCORE_BAR_WIDTH, SCORE_BAR_HEIGHT))

        fill_width = int(abs(score) * SCORE_BAR_WIDTH//2)
        if score > 0:
            color = Colors.green
            x_pos = SCORE_BAR_X + SCORE_BAR_WIDTH//2
        else:
            color = Colors.red
            x_pos = SCORE_BAR_X + SCORE_BAR_WIDTH//2 - fill_width

        if fill_width > 0:
            pygame.draw.rect(surface, color,
                           (x_pos, SCORE_BAR_Y, fill_width, SCORE_BAR_HEIGHT))

        pygame.draw.line(surface, Colors.black,
                        (SCORE_BAR_X + SCORE_BAR_WIDTH//2, SCORE_BAR_Y),
                        (SCORE_BAR_X + SCORE_BAR_WIDTH//2, SCORE_BAR_Y + SCORE_BAR_HEIGHT))

        font = pygame.font.SysFont(None, 24)
        text = font.render(f"Score: {score:.2f}", True, Colors.black)
        text_rect = text.get_rect(center=(SCORE_BAR_X + SCORE_BAR_WIDTH//2, SCORE_BAR_Y - 15))
        surface.blit(text, text_rect)

    def redraw_window(self, players, walls, pos, score=0.0, current_player=None):
        self.win.fill(self.bgcolor)
        self.draw_game_board(pos)
        self.draw_finish_lines(players)
        self.draw_right_panel(players, current_player)
        self.draw_buttons()
        self.draw_score_bar(score)
        players.draw(self)
        walls.draw()
        self.info.draw(self.win, (self.top_left[0], self.height - 50))
        pygame.display.update()
