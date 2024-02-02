from time import sleep

import pygame as pg
import sys

from typing import Dict, List, Tuple

from route_choice_env.problem import Network, Link, Node


WIN_SIZE = (1200, 600)
GRID_SIZES = {
    'OW': (5, 3),
    'SF': (6, 4)
}

RED = (219, 68, 55)
ORANGE = (239, 84, 16)
YELLOW = (244, 180, 0)
GREEN = (15, 157, 88)

BLUE = (0, 121, 191)  # labels

GRAY = (180, 180, 180)  # nodes and links


class EnvViewer(object):
    """A viewer to render a road network environment."""

    def __init__(self, env: 'AbstractEnv', road_network: Network, win_size=WIN_SIZE, grid_size=None) -> None:
         # init pygame modules
        pg.init()
        # self.env = env

        # window size
        self.WIN_SIZE = win_size
        self.GRID_SIZE = GRID_SIZES[road_network.name]

        self.font = pg.font.SysFont(None, 30)
        # set opengl attr
        # pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        # pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        # pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)

        # create opengl context
        self.screen = pg.display.set_mode(self.WIN_SIZE,)  # flags=pg.OPENGL | pg.DOUBLEBUF)
        pg.display.set_caption(f"RouteChoiceEnv - {road_network.name}")

        # detect and use existing opengl context
        # self.ctx = mgl.create_context()

        # create an object to help track time
        self.clock = pg.time.Clock()
        self.time = 0

        self.metric = 'cost'

        # self.scene = None
        self.scene = build_scene(road_network, self.WIN_SIZE, self.GRID_SIZE, self.metric)


    def check_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                self.scene.destroy()
                pg.quit()
                sys.exit()

            if event.type == pg.KEYDOWN and event.key == pg.K_c:
                self.metric = 'cost'
            elif event.type == pg.KEYDOWN and event.key == pg.K_f:
                self.metric = 'flow'

    def render(self, road_network):
        # clear framebuffer
        # self.ctx.clear(color=(0.08, 0.16, 0.18))

        self.check_events()

        # Clear the screen
        self.screen.fill((255, 255, 255))

        for renderer in self.scene.renderers:
            if renderer.element \
                and renderer.element.collidepoint(pg.mouse.get_pos()):

                renderer.display(self.screen, self.font, pg.mouse.get_pos())

        self.scene = build_scene(road_network, self.WIN_SIZE, self.GRID_SIZE, self.metric)

        # render scene
        self.scene.render(self.screen, self.font)

        # swap buffers
        pg.display.flip()

        self.clock.tick(60)

    def close(self) -> None:
        """Close the pygame window."""
        pg.quit()


class LinkRenderer:
    def __init__(self, link, color, metric, width=10, position=(100, 100, 100, 100)):
        self.link: Link = link  # Reference to the Link object
        self.color = color

        if metric == 'cost':
            self.metric_func_callback: function = self.link.get_cost
        elif metric == 'flow':
            self.metric_func_callback: function = self.link.get_flow

        self.width = width

        # Dislocate position a bit to avoid overlapping two way roads
        name = str(self.link)
        s = name[0]
        e = name[-1]
        if s < e:
            self.offset = 4
        else:
            self.offset = -4

        # Get the start and end coordinates of the link (you'll need to adapt this based on your data model)
        self.start_x = position[0] + self.offset
        self.start_y = position[1] + self.offset
        self.end_x = position[2] + self.offset
        self.end_y = position[3] + self.offset

        self.element = None

    def render(self, surface, font):
        self.update()
        # Draw the link as a colored rectangle
        self.element = pg.draw.line(surface, self.color, (self.start_x, self.start_y), (self.end_x, self.end_y), self.width)

    def update(self):
        v = self.metric_func_callback()

        if v > 100:
            self.color = RED
        elif v > 50:
            self.color = ORANGE
        elif v > 20:
            self.color = YELLOW
        elif v > 5:
            self.color = GREEN
        else:
            self.color = GRAY

        # self.color = GRAY
        # print(f'updated link {str(self.link)} with flow {v} to color {self.color}')
        # sleep(1)

    def display(self, surface, font, pos):
        label = str(self.link) + ': ' + str( round( self.metric_func_callback(), 2 ) )
        display_text_box(label, surface, font, pos)

    def destroy(self):
        pass


class NodeRenderer:
    def __init__(self, node, color, radius=30, position=(100, 100)):
        self.node: Node = node  # Reference to the node object
        self.color = color

        self.radius = radius

        # Get the start and end coordinates of the node (you'll need to adapt this based on your data model)
        self.center_x = position[0]
        self.center_y = position[1]

        self.element = None

    def render(self, surface, font):
        # Draw the node as a colored circle
        self.element = pg.draw.circle(surface=surface, color=self.color, center=(self.center_x, self.center_y), radius=self.radius)

        label = font.render(self.node, True, BLUE)
        surface.blit(label, label.get_rect(center = (self.center_x, self.center_y)))

    def update(self):
        pass

    def display(self, surface, font, pos):
        pass

    def destroy(self):
        pass


class MenuRenderer:
    def __init__(self, metric: str, color, position=(100, 100)):
        self.metric = metric
        self.color = color
        self.pos = position

        self.element = None

    def render(self, surface, font):
        display_text_box(self.metric, surface, font, self.pos)

    def update(self):
        pass

    def display(self, surface, font, pos):
        pass

    def destroy(self):
        pass


class Scene:
    def __init__(self) -> None:
        self.renderers = []

    def add_renderer(self, renderer):
        self.renderers.append(renderer)

    def render(self, surface, font):
        for r in self.renderers:
            r.render(surface, font)

    def update(self):
        for r in self.renderers:
            r.update()

    def destroy(self):
        for r in self.renderers:
            r.destroy()


def get_cell_positions(win_size: Tuple[int, int], grid_size: Tuple[int, int] = (5, 3)) -> Tuple[List[Tuple[int, int]], int, int]:
    cell_size_x = win_size[0] / grid_size[0]
    cell_size_y = win_size[1] / grid_size[1]
    x_offsets = [cell_size_x * i for i in range(grid_size[0])]
    y_offsets = [cell_size_y * i for i in range(grid_size[1])]
    cell_positions = [(x, y) for x in x_offsets for y in y_offsets ]
    return cell_positions, cell_size_x, cell_size_y


def build_scene(net: Network, win_size: Tuple[int, int] = WIN_SIZE, grid_size: Tuple[int, int] = None, metric: str = "") -> Scene:
    if not grid_size:
        raise ValueError("Could not build_scene. Grid size is missing.")
    elif not metric:
        raise ValueError("Could not build_scene. Metric is missing.")

    # precisamos de um algoritmo, que divida o tamanho da tela pelo número de nós e distribua os nós na grade
    # por exemplo, OW tem 13 nós precisamos de uma grade 5x3
    cell_positions, cell_size_x, cell_size_y = get_cell_positions(win_size, grid_size)

    n_nodes = len(net.get_N())
    if n_nodes > len(cell_positions):
        raise ValueError(f"Grid size {grid_size} too small for {n_nodes} nodes.")


    node_radius = int( ( min(win_size) / max(grid_size) ) * 0.2 )  # % of the smallest dimension of the window
    link_width = int( node_radius * 0.3 )  # % of the node radius

    node_radius *= 0.2
    nodes: Dict[str, Tuple[int, int]] = {}
    offset = 0
    for i, n in enumerate(net.get_N()):

        # skip
        if i == 2 or i == 12:
            offset += 1

        i += offset
        n = str(n)
        node_position = ( cell_positions[i][0] + (cell_size_x / 2), cell_positions[i][1] + (cell_size_y / 2) )
        nodes[n] = node_position


    links: Dict[str, Tuple[int, int, int, int]] = {}

    # print(net.get_L())
    # sys.exit()

    # print(nodes)

    for i, l in enumerate(net.get_L()):
        l = str(l)
        l_split = l.split('-')

        s_x, s_y = nodes[ l_split[0] ]
        e_x, e_y = nodes[ l_split[-1] ]

        link_position = (s_x, s_y, e_x, e_y)
        links[l] = link_position

    # menu
    menu = MenuRenderer(metric, color=GRAY, position=(0, 0))

    scene = Scene()
    scene.add_renderer(menu)
    for l, link_position in links.items():
        scene.add_renderer( LinkRenderer(net.get_link(l), color=GRAY, metric=metric, width=link_width, position=link_position) )
    for n, node_position in nodes.items():
        scene.add_renderer( NodeRenderer(n, color=GRAY, radius=node_radius, position=node_position) )
    return scene


def display_text_box(text, screen, font, pos):
    # Create a gray rectangle (box)
    box_rect = pg.Rect(pos[0], pos[1], WIN_SIZE[0] * 0.1, WIN_SIZE[1] * 0.1)

    # Render the text with blue color
    text_surface = font.render(text, True, BLUE)

    # Calculate the position to center the text in the box
    text_rect = text_surface.get_rect()
    text_rect.center = box_rect.center

    # Fill the screen with gray
    # screen.fill(GRAY)

    # Draw the gray box
    pg.draw.rect(screen, GRAY, box_rect)

    # Blit the text onto the box
    screen.blit(text_surface, text_rect)


def distance_point_line(pt, l1, l2):
    NV = pg.math.Vector2(l1[1] - l2[1], l2[0] - l1[0])
    LP = pg.math.Vector2(l1)
    P = pg.math.Vector2(pt)
    return abs(NV.normalize().dot(P -LP))
