import os
import sys
import math
from typing import Dict, List, Tuple

import dearpygui.dearpygui as dpg

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

LINK_POSITION = Tuple[int, int, int, int]  # (start_x, start_y, end_x, end_y)
NODE_POSITION = Tuple[int, int]  # (center_x, center_y)

LINK_WIDTH = None
LINK_DISTANCES = None

SELECTED_LINK = None  # str

class EnvViewer(object):
    __metric = "flow"
    __pause = False
    __step = False

    def __init__(self, env: "AbstractEnv", win_size=WIN_SIZE):
        self.__win_size = win_size
        self.__grid_size = GRID_SIZES[env.road_network.name]
        self.__win_width = win_size[0]
        self.__win_height = win_size[1]
        self.__drawlist_winsize = (self.__win_width * 0.75, self.__win_height - 56)

        # CONTROLS SETUP
        # --------------
        def pause(sender, app_data):
            self.__pause = True

        def resume(sender, app_data):
            self.__pause = False

        def step(sender, app_data):
            self.__step = True

        def exit(sender, app_data):
            dpg.destroy_context()
            sys.exit(0)

        def update_metric(sender, app_data, user_data):
            self.__metric = user_data['metric']

        dpg.create_context()

        # font
        with dpg.font_registry():
            default_font = dpg.add_font(f"{os.getcwd()}\\assets\\Roboto-Light.ttf", 16)
            primary_font = dpg.add_font(f"{os.getcwd()}\\assets\\Roboto-Light.ttf", 20)

        # draw area
        _drawlist_winsize = self.__drawlist_winsize  # subtract 56 to account for window title bar
        with dpg.window(label="Main Window", tag="main_window") as main_window:
            dpg.set_primary_window(main_window, True)


            # LEFT SIDE OF THE SCREEN
            # -----------------------
            with dpg.drawlist(width=self.__win_width * 0.8, height=self.__win_height-56, tag="drawlist") as drawlist:
                build_scene(env.road_network, _drawlist_winsize, self.__grid_size)

            # RIGHT SIDE OF THE SCREEN
            # ------------------------
            _lateral_pos = (self.__win_width * 0.75, 0)
            _lateral_width = self.__win_width * 0.25
            _lateral_height = self.__win_height
            with dpg.window(label="lateral_menu", tag="lateral_menu", pos=_lateral_pos, width=_lateral_width, height=_lateral_height, \
                        no_move=True, no_close=True, no_resize=True, no_collapse=True, no_title_bar=True) as lateral_menu:

                # INFORMATION
                with dpg.table(header_row=True, parent=lateral_menu, width=_lateral_width*0.9, tag="routechoiceenv_tab"):
                    dpg.add_table_column()
                    dpg.add_table_column(label="RouteChoiceEnv", tag="title", width_fixed=True)
                    dpg.add_table_column()

                dpg.add_text(f"Network: {env.road_network.name}", tag="network", parent=lateral_menu)
                dpg.add_text(f"Cost funciton: {env.road_network.get_expr()}", tag="expr", parent=lateral_menu)
                dpg.add_text(f"Total flow: {env.max_num_agents}", tag="flow", parent=lateral_menu)
                dpg.add_text(f"Episode: 0 / {env.episodes}", tag="episode", parent=lateral_menu)
                dpg.add_text(f"Average cost over routes", tag="metric", parent=lateral_menu)  # {_metric}

                dpg.add_spacer()
                dpg.add_separator()
                dpg.add_spacer()

                # CONTROLS
                with dpg.table(header_row=True, parent=lateral_menu, width=_lateral_width*0.9, tag="controls_tab"):
                    dpg.add_table_column()
                    dpg.add_table_column(label="Controls", tag="controls_col", width_fixed=True)
                    dpg.add_table_column()

                # dpg.add_text("Controls", tag="controls", parent=lateral_menu)
                dpg.add_text("  (P) - Pause", parent=lateral_menu)
                dpg.add_text("  (R) - Resume", parent=lateral_menu)
                dpg.add_text("  (S) - Step", parent=lateral_menu)
                dpg.add_text("  (Esc) - Exit", parent=lateral_menu)

                dpg.add_spacer()
                dpg.add_separator()
                dpg.add_spacer()

                with dpg.table(header_row=True, parent=lateral_menu, width=_lateral_width*0.9, tag="metricselection_tab"):
                    dpg.add_table_column()
                    dpg.add_table_column(label="Metric Selection", tag="metricselection_col", width_fixed=True)
                    dpg.add_table_column()

                # dpg.add_text("Metric Selection", tag="metricselection", parent=lateral_menu)
                dpg.add_text("  (F) - Flow", parent=lateral_menu)
                dpg.add_text("  (C) - Cost", parent=lateral_menu)

                with dpg.handler_registry():
                    dpg.add_key_press_handler(key=dpg.mvKey_P, callback=pause)
                    dpg.add_key_press_handler(key=dpg.mvKey_R, callback=resume)
                    dpg.add_key_press_handler(key=dpg.mvKey_S, callback=step)
                    dpg.add_key_press_handler(key=dpg.mvKey_Escape, callback=exit)

                    dpg.add_key_press_handler(key=dpg.mvKey_F, callback=update_metric, user_data={'metric': 'flow'})
                    dpg.add_key_press_handler(key=dpg.mvKey_C, callback=update_metric, user_data={'metric': 'cost'})

                # PLOT
                _plot_pos = (0, _lateral_height*0.6)
                _plot_width = _lateral_width * 0.94
                _plot_height = _lateral_height * 0.34
                with dpg.plot(label=f"Average Reward - {env.algorithm}", pos=_plot_pos, width=_plot_width, height=_plot_height, tag="avg_travel_time", parent=lateral_menu):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Episodes", tag="x_axis")
                    dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis")

                    dpg.set_axis_limits("x_axis", 0, 1000)  # episodes

                    dpg.add_line_series(plotdatax, plotdatay, parent="y_axis", tag="series")

            dpg.bind_font(default_font)
            dpg.bind_item_font("routechoiceenv_tab", primary_font)
            dpg.bind_item_font("controls_tab", primary_font)
            dpg.bind_item_font("metricselection_tab", primary_font)

        # WINDOW SETUP
        # --------------
        dpg.create_viewport(title='RouteChoiceEnv', width=self.__win_width, height=self.__win_height)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)


    def render(self, env: "AbstractEnv"):
        dpg.set_value("episode", f"Episode: {env.iteration} / {env.episodes}")
        if self.__metric == 'flow':
            dpg.set_value("metric", f"Average flow over routes: { round(env.avg_flow, 2) }")
        elif self.__metric == 'cost':
            dpg.set_value("metric", f"Average cost over routes: { round(env.avg_travel_time, 2) }")
        update_series(env)
        update_scene(env.road_network, self.__drawlist_winsize)
        dpg.render_dearpygui_frame()

        while self.__pause:
            self.__step = False
            update_info(env.road_network, self.__drawlist_winsize)
            dpg.render_dearpygui_frame()
            if self.__step:
                break


# PLOT SETUP
# ----------
plotdatax = []
plotdatay = []

def update_series(env):
    if not plotdatax:
        plotdatax.append(0)
    else:
        plotdatax.append(plotdatax[-1] + 1)
    plotdatay.append(env.avg_travel_time)
    dpg.set_value("series", [plotdatax, plotdatay])
    dpg.fit_axis_data("y_axis")


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
        ):

    global LINK_WIDTH

    if not grid_size:
        raise ValueError("Could not build_scene. Grid size is missing.")

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


    with dpg.window(label="info_window", tag="info_window", show=False, \
                    no_move=True, no_close=True, no_resize=True, no_collapse=True, no_title_bar=True) as info_window:
        dpg.add_text("none", tag="info_text", parent=info_window)

    def show_info(sender, app_data, user_data):
        global SELECTED_LINK

        _LINK = user_data['link']
        _P1 = user_data['p1']
        _P2 = user_data['p2']
        _THICKNESS = user_data['thickness']

        _mouse_pos = dpg.get_mouse_pos()
        if _mouse_pos[0] <= 20.0 or _mouse_pos[1] <= 20.0:  # 20 as heuristic
            return

        if distance_point_line_segment(_mouse_pos, _P1, _P2) < _THICKNESS / 2:
            dpg.configure_item("info_window", show=True, pos=_mouse_pos)
            SELECTED_LINK = _LINK


    for l, link_position in links.items():
        s = l[0]  # e.g. A
        e = l[-1]  # e.g. B
        _offset = 6 if s < e else -6  # spacing for two way roads

        if link_position[0] == link_position[2]:  # they match horizontally
            p1 = ( link_position[0] - _offset, link_position[1] - _offset )
            p2 = ( link_position[2] - _offset, link_position[3] - _offset )
        elif link_position[1] == link_position[3]:  # they match vertically
            p1 = ( link_position[0] + _offset, link_position[1] + _offset )
            p2 = ( link_position[2] + _offset, link_position[3] + _offset )
        else:
            _offset += _offset if _offset > 0 else 0  # if diagonal, we double spacing
            p1 = ( link_position[0] - _offset, link_position[1] )
            p2 = ( link_position[2] - _offset, link_position[3] )

        # print(l, p1, p2)

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


def update_scene(net: Network, win_size=WIN_SIZE):
    for i, l in enumerate(net.get_L()):
        _v = net.get_link(l).get_flow()
        _color = color_gradient(_v, 0.0, net.get_total_flow()/2)
        dpg.configure_item(l, color=_color)

    update_info(net, win_size)


def update_info(net: Network, win_size=WIN_SIZE):
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
Marginal cost: { round(_link.get_marginal_cost(), 2) }
"""
        )

    if _mouse_pos[0] < 0 or \
        _mouse_pos[0] > _width or \
        _mouse_pos[1] < 0 or \
        _mouse_pos[1] > _height+20:  # adds 20 to height to account for window title bar

        dpg.configure_item("info_window", show=False)
        SELECTED_LINK = None


plotdatax = []
plotdatay = []

def update_series(env):
    if not plotdatax:
        plotdatax.append(0)
    else:
        plotdatax.append(plotdatax[-1] + 1)
    plotdatay.append(env.avg_travel_time)
    dpg.set_value("series", [plotdatax, plotdatay])
    dpg.fit_axis_data("y_axis")
    # dpg.set_axis_limits("y_axis", 0, max(plotdatay))


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


def distance_point_line_segment(pt, l1, l2):
    # Vector from l1 to l2
    dx, dy = l2[0] - l1[0], l2[1] - l1[1]
    # Vector from l1 to the point
    px, py = pt[0] - l1[0], pt[1] - l1[1]

    # Project point onto the line (using dot product)
    dot_product = px * dx + py * dy
    line_len_sq = dx * dx + dy * dy
    param = dot_product / line_len_sq

    if param < 0:  # Closest to l1
        closest_x, closest_y = l1[0], l1[1]
    elif param > 1:  # Closest to l2
        closest_x, closest_y = l2[0], l2[1]
    else:  # Projection falls on the segment
        closest_x = l1[0] + param * dx
        closest_y = l1[1] + param * dy

    # Distance from the point to the closest point on the segment
    dist_x = pt[0] - closest_x
    dist_y = pt[1] - closest_y
    dist = math.hypot(dist_x, dist_y)

    return dist
