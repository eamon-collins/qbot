##This file is for supporting my qbot integration with Quentin's quoridor interface client.
#Eamon

import sys
import os
import traceback
#sorry pygame, it's annoying
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
from quoridor.client.src.network import Network
from quoridor.client.src.window import Window, pos_in_rect
from quoridor.client.src.player import Players, Player
from quoridor.client.src.wall import Walls, Wall
from quoridor.client.src.pathfinder import PathFinder
from quoridor.client.src.sounds import Sounds
from quoridor.client.src.colors import Colors



#wx and wy are list of coordinates that have walls next to them
#p1w is player1 numfences
#p1x and p1y are player 1 coords
def visualize_gamestate(wx, wy, p1w, p1x, p1y, p2w, p2x, p2y):
    # Init pygame
    #place where the pygame window opens
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d, %d" %(1990, 1200)
    #os.environ['SDL_VIDEO_CENTERED'] = "0"
    pygame.init()
    clock = pygame.time.Clock()
    path = os.path.dirname(os.path.abspath(__file__))
    sounds = Sounds(path)

    # Init game
    win = Window()
    coords = win.coords
    players = Players(2, coords)
    p1y = 8-p1y
    p2y = 8-p2y
    # print(f"players {p1y}, {p1x}")
    # print(f"players {p2y}, {p2x}")
    players.players = [Player(0, p1w, "north", Colors.red, coords.find_coord(p1x, p1y)),
                        Player(1, p2w, "south", Colors.green, coords.find_coord(p2x, p2y))]
    players.set_names(["qbot", "Player"])
    walls = Walls()
    pf = PathFinder()

    # wx = [0, 1]
    # wy = [0,0]
    
    # wx = [4,5]
    # wy = [2,2]
    print("#H#$#")

    if len(wx) %2 == 0:
        arr_len = len(wx)
    else:
        arr_len = len(wx) -1
    if wx and wx[0] != -1:
        for i in range(0, arr_len, 2):
            #y,x is row,col from down.
            print(f"{wy[i]},{wx[i]}   {wy[i+1]},{wx[i+1]}")
            w = Wall(coords.find_coord(wx[i],wy[i]), coords.find_coord(wx[i+1],wy[i+1]), win)
            #horizontal walls need to be flipped
            # if wx[i] == wx[i+1]:
            #   print("flipping")
            #   w.make_cross_wall()
            #   w = w.cross_wall
            w.set_color(Colors.black)
            walls.add_wall(w)

    run = True
    while run:
        clock.tick(40)

        pos = pygame.mouse.get_pos()
        try:
            win.redraw_window(None, players, walls, pos)
        

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    pygame.quit()

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    if win.button_quit.click(pos):
                        run = False
                        pygame.quit()
                    else:    # Put a wall
                        for c in coords.coords:
                            wall_east = c.wall_east
                            wall_south = c.wall_south
                            for w in [wall_east, wall_south]:
                                if (w is not None and pos_in_rect(w.rect_small, pos)
                                        and walls.can_add(w)):
                                    if pf.play_wall(w, players):
                                        print(f"cx,y {c.x},{c.y}" )
                                        if w == c.wall_south:
                                            horiz = 1
                                        else:
                                            horiz = 0
                                        pygame.quit()
                                        print(f"c.y {c.y} horiz {horiz}")
                                        print("playermove: "+ f"f{horiz}{(7-c.y)*2+horiz} {c.x}")
                                        return str(f"f{horiz}{(7-c.y)*2+horiz} {c.x}")
                        # mes = player.play_put_wall(
                        #   pos, coords, walls, n, pf, players)
                        # if mes != '':
                        #   win.update_info(mes)

                elif event.type == pygame.KEYDOWN:  # Move pawn
                    # if player.can_play(game):
                    #   player.play_move(walls, n)
                    x = p2x
                    y = 8-p2y
                    if event.key == pygame.K_LEFT and p2x > 0:
                        c = coords.find_coord(p2x - 1, 8-p2y)
                        x-=1
                    elif event.key == pygame.K_RIGHT and p2x < 8:
                        c = coords.find_coord(p2x + 1, 8-p2y)
                        x+=1
                    elif event.key == pygame.K_UP and 8-p2y < 8 :
                        c = coords.find_coord(p2x, 8-p2y + 1)
                        y+=1
                    elif event.key == pygame.K_DOWN and 8-p2y > 0:
                        c = coords.find_coord(p2x, 8-p2y - 1)
                        y-=1    
                    else:
                        continue

                    pygame.quit()
                    # return str(f"p0{c.y} {c.x}")
                    return str(f"p0{y} {x}")

        #hacky but good for debugging
        except Exception as e:
            print(e)
            tb = traceback.extract_tb(e.__traceback__)
            print(tb)
            return "error"
