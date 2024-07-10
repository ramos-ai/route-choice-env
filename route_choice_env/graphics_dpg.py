# basic ref - https://www.youtube.com/watch?v=kkRXLrG5oMA&ab_channel=JonathanHoffstadt

# https://github.com/hoffstadt/DearPyGui
# https://github.com/my1e5/dpg-examples/tree/main

# https://dearpygui.readthedocs.io/en/latest/extra/video-tutorials.html
# https://dearpygui.readthedocs.io/en/latest/reference/dearpygui.html#dearpygui.dearpygui.draw_circle

# https://dearpygui.readthedocs.io/en/latest/documentation/drawing-api.html
# https://dearpygui.readthedocs.io/en/latest/extra/plotting.html

import sys
import math
from typing import Dict, List, Tuple

import dearpygui.dearpygui as dpg

from route_choice_env.problem import Network, Link, Node

WIN_SIZE = (1600, 900)
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

LINK_POSITION = Tuple[int, int, int, int]  # (start_x, start_y, end_x, end_y)
NODE_POSITION = Tuple[int, int]  # (center_x, center_y)

LINK_WIDTH = None
LINK_DISTANCES = {}

SELECTED_LINK = None


def get_cell_positions(
        win_size: Tuple[int, int],
        grid_size: Tuple[int, int] = (5, 3)
        ) -> Tuple[List[Tuple[int, int]], int, int]:

    cell_size_x = win_size[0] / grid_size[0]
    cell_size_y = win_size[1] / grid_size[1]
    x_offsets = [cell_size_x * i for i in range(grid_size[0])]
    y_offsets = [cell_size_y * i for i in range(grid_size[1])]
    cell_positions = [(x, y) for x in x_offsets for y in y_offsets ]
    return cell_positions, cell_size_x, cell_size_y


def build_scene(
        net: Network,
        win_size: Tuple[int, int] = WIN_SIZE,
        grid_size: Tuple[int, int] = None,
        metric: str = ""
        ):

    global LINK_WIDTH

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
    LINK_WIDTH = link_width
    node_radius *= 0.2


    nodes: Dict[str, Tuple[int, int]] = {}
    Ns = net.render_order if net.render_order else net.get_N().keys()
    for i, n in enumerate(Ns):
        if n == '_':   # skip
            continue

        n = str(n)
        node_position = ( cell_positions[i][0] + (cell_size_x / 2), cell_positions[i][1] + (cell_size_y / 2) )
        nodes[n] = node_position


    links: Dict[str, Tuple[int, int, int, int]] = {}
    for i, l in enumerate(net.get_L()):
        l = str(l)
        l_split = l.split('-')

        s_x, s_y = nodes[ l_split[0] ]
        e_x, e_y = nodes[ l_split[-1] ]

        link_position = (s_x, s_y, e_x, e_y)
        links[l] = link_position



    # we need a way to render info window only for link I clicked on

    # A.
    # 1. keep a data structure with mouse click and distance point line to links
    # 2. every time I click on the screen, I calculate the distance to all links
    # 3. if the distance is less than the thickness, I show the info window
    # 4. I need to keep the link data to show in the info window

    # B.
    # 1. use the tag of the link to identify it
    # 2. when I click on the link, I pass the link tag to the handler
    # 3. I use the tag to identify the link and show the info window


    with dpg.window(label="info_window", tag="info_window", show=False, \
                    no_move=True, no_close=True, no_resize=True, no_collapse=True, no_title_bar=True) as info_window:
        dpg.add_text("none", tag="info_text", parent=info_window)

    def show_info(sender, app_data, user_data):
        _LINK = user_data['link']
        _P1 = user_data['p1']
        _P2 = user_data['p2']
        _THICKNESS = user_data['thickness']

        _mouse_pos = dpg.get_mouse_pos()
        if distance_point_line(_mouse_pos, _P1, _P2) < _THICKNESS:
            dpg.configure_item("info_window", show=True, pos=_mouse_pos)
            global SELECTED_LINK
            SELECTED_LINK = _LINK


    def calculate_distances(sender, app_data, user_data):
        _links = user_data['links']

        _mouse_pos = dpg.get_mouse_pos()
        for l, link_position in _links.items():
            s = l[0]  # e.g. A
            e = l[-1]  # e.g. B
            _offset = 8 if s < e else -8
            p1 = ( link_position[0] + _offset, link_position[1] + _offset )
            p2 = ( link_position[2] + _offset, link_position[3] + _offset)

            LINK_DISTANCES[l] = distance_point_line(_mouse_pos, p1, p2)


    for l, link_position in links.items():
        s = l[0]  # e.g. A
        e = l[-1]  # e.g. B
        _offset = 8 if s < e else -8
        p1 = ( link_position[0] + _offset, link_position[1] + _offset )
        p2 = ( link_position[2] + _offset, link_position[3] + _offset)
        _v = net.get_link(l).get_flow()
        _color = color_gradient(_v, 0.0, net.get_total_flow())  # GRAY

        dpg.draw_line(p1, p2, color=_color, thickness=link_width, tag=l)

        with dpg.handler_registry():
            dpg.add_mouse_click_handler(
                callback=show_info,
                user_data={'link': l, 'p1': p1, 'p2': p2, 'thickness': link_width}
            )

    for n, node_position in nodes.items():
        dpg.draw_circle(node_position, node_radius, color=GRAY, thickness=link_width*4, tag=n)
        dpg.add_text(parent=n, color=BLUE)


    # with dpg.handler_registry():
    #     dpg.add_mouse_click_handler(
    #         callback=calculate_distances,
    #         user_data={'links': links}
    #     )


def update_scene(net: Network, metric, win_size=WIN_SIZE):
    # for i, n in enumerate(net.get_N().keys()):
    #     dpg.configure_item(n, color=dpg.mvColor(255, 0, 0, 255))(n)

    for i, l in enumerate(net.get_L()):
        _v = net.get_link(l).get_flow()
        _color = color_gradient(_v, 0.0, net.get_total_flow()/2)
        dpg.configure_item(l, color=_color)


    _width = win_size[0]
    _height = win_size[1]
    _mouse_pos = dpg.get_mouse_pos(local=False)

    global SELECTED_LINK
    if SELECTED_LINK:
        _link: Link = net.get_link(SELECTED_LINK)
        dpg.set_value("info_text",
f"""
Link: {SELECTED_LINK} \n
Flow: { round(_link.get_flow(), 1) } \n
Cost: { round(_link.get_cost(), 2) } \n
"""
        )
    else:
        dpg.set_value("info_text", f"Mouse Position: {_mouse_pos}")

    if _mouse_pos[0] < 0 or \
        _mouse_pos[0] > _width or \
        _mouse_pos[1] < 0 or \
        _mouse_pos[1] > _height+20:  # adds 20 to height to account for window title bar
        dpg.configure_item("info_window", show=False)

        SELECTED_LINK = None


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

    return  (red, green, blue)  # dpg.mvColor(red, green, blue, 255)


def distance_point_line(pt, l1, l2):
    nx, ny = l1[1] - l2[1], l2[0] - l1[0]
    nlen = math.hypot(nx, ny)
    nx /= nlen
    ny /= nlen
    vx, vy = pt[0] - l1[0],  pt[1] - l1[1]
    dist = abs(nx*vx + ny*vy)
    return dist


if __name__ == '__main__':
    env = None
    road_network = Network('OW', 8)
    # run(env)
