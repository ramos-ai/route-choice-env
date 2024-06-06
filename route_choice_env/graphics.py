import pygame as pg
import sys

from typing import Dict, List, Tuple

from route_choice_env.problem import Network, Link, Node


WIN_SIZE = (1200, 700)
GRID_SIZES = {
    'BBraess_7_2100_10_c1_900': (3, 8),  # 20
    'Braess_1_4200_10_c1': (2, 2),  # 4
    'Braess_7_4200_10_c1': (5, 8),  # 16
    'OW': (5, 3),  # 13
    'SF': (8, 5),  # 40
    'Anaheim': (26, 16),  # 416
    'Eastern-Massachusetts': (10, 12),  # 74
}

BLUE = (0, 121, 191)  # labels
GRAY = (180, 180, 180)  # nodes and links


class EnvViewer(object):
    """A viewer to render a road network environment."""

    def __init__(self, env: "AbstractEnv", road_network: Network, win_size=WIN_SIZE, grid_size=None) -> None:  # noqa: F821
         # init pygame modules
        pg.init()

        # window size
        self.WIN_SIZE = win_size
        self.GRID_SIZE = GRID_SIZES[road_network.name]

        # create opengl context
        self.screen = pg.display.set_mode(self.WIN_SIZE,)  # flags=pg.OPENGL | pg.DOUBLEBUF)
        pg.display.set_caption(f"RouteChoiceEnv - {road_network.name}")

        # create an object to help track time
        self.clock = pg.time.Clock()
        self.time = 0

        # create a font object
        self.font = pg.font.SysFont(None, 30)

        self.metric = 'cost'
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

        mouse_pos = pg.mouse.get_pos()

        for renderer in self.scene.renderers:
            if (renderer.element and
                renderer.element.collidepoint(mouse_pos)):  # and
                # distance_point_line(mouse_pos, renderer.local_x, renderer.local_y) < 5):

                renderer.display(self.screen, self.font, mouse_pos)

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
    def __init__(self, net, link, color, metric, width=10, position=(100, 100, 100, 100)):
        self.net: Network = net
        self.link: Link = link  # Reference to the Link object
        self.color = color
        self.metric_func_callback: callable = None

        if metric == 'cost':
            self.metric_func_callback = self.link.get_cost
        elif metric == 'flow':
            self.metric_func_callback = self.link.get_flow

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

        self.local_x = (self.start_x, self.end_x)
        self.local_y = (self.start_y, self.end_y)

        self.element = None

    def render(self, surface, font):
        self.update()
        # Draw the link as a colored rectangle
        self.element = pg.draw.line(surface, self.color, (self.start_x, self.start_y), (self.end_x, self.end_y), self.width)

    def update(self):
        v = self.metric_func_callback()
        self.color = color_gradient(v, 0.0, self.net.get_total_flow())

    def display(self, surface, font, pos):
        label = str(self.link) + ': ' + str( round( self.metric_func_callback(), 2 ) )

        pos = (0, 100)

        display_text_box(label, surface, font, pos)

    def destroy(self):
        pass


class NodeRenderer:
    def __init__(self, node, color, radius=10, position=(100, 100)):
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

    Ns = net.render_order if net.render_order else net.get_N().keys()
    for i, n in enumerate(Ns):

        # skip
        if n == '_':
            continue

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
        scene.add_renderer( LinkRenderer(net, net.get_link(l), color=GRAY, metric=metric, width=link_width, position=link_position) )
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


# Para fazer o gradiente de cores, podemos usar a função abaixo, que recebe um valor e retorna uma cor RGB
# O valor é normalizado entre min_value e max_value
# Quanto mais proximo do min_value, mais vermelho.
# Quanto mais proximo do max_value, mais verde.
# Inclua também um valores intermediários para amarelo e laranja.
def color_gradient(value: float, min_value: float, max_value: float) -> Tuple[int, int, int]:
    """
    Maps a value to an RGB color, transitioning from Green to Yellow to Red
    across the specified range of values.

    :param value: The value to map to a color.
    :param min_value: The minimum value, mapped to Green.
    :param max_value: The maximum value, mapped to Red.
    :return: A tuple representing the RGB color.
    """
    if value == 0.0:
        return (180, 180, 180)

    # Ensure the value is within the bounds
    value = max(min(value, max_value), min_value)
    # Normalize the value to a 0-1 scale
    normalized = (value - min_value) / (max_value - min_value)

    if normalized < 0.5:
        # Scale from Green (0,1,0) to Yellow (1,1,0)
        red = int(2 * normalized * 255)  # Increase red to transition to yellow
        green = 255
    else:
        # Scale from Yellow (1,1,0) to Red (1,0,0)
        red = 255
        green = int((1 - (normalized - 0.5) * 2) * 255)  # Decrease green to transition to red
    blue = 0

    return (red, green, blue)


def distance_point_line(pt, l1, l2):
    NV = pg.math.Vector2(l1[1] - l2[1], l2[0] - l1[0])
    LP = pg.math.Vector2(l1)
    P = pg.math.Vector2(pt)

    # print(pt, l1, l2)
    # print(NV, LP, P)

    return abs(NV.normalize().dot(P - LP))
