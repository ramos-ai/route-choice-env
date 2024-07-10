from time import sleep

import numpy as np
import dearpygui.dearpygui as dpg
from typing import Dict
from graphics_dpg import build_scene, update_scene, WIN_SIZE, GRID_SIZES
from pettingzoo.utils.conversions import AgentID

from route_choice_env.core import Policy
from route_choice_env.route_choice import RouteChoicePZ
from route_choice_env.policy import EpsilonGreedy

from route_choice_env.agents.simple_driver import SimpleDriver
from route_choice_env.agents.rmq_learning import RMQLearning
from route_choice_env.agents.tq_learning import TQLearning
from route_choice_env.agents.gtq_learning import GTQLearning

from route_choice_env.services import simulate


def get_simple_driver_agents(env: RouteChoicePZ, policy: Policy) -> Dict[AgentID, SimpleDriver]:
    return {
        d_id: SimpleDriver(
            d_id=d_id,
            actions=list(range(env.action_space(d_id).n)),
            policy=policy
        )
        for d_id in env.agents
    }


def get_rmq_learning_agents(env: RouteChoicePZ, policy: Policy) -> Dict[AgentID, RMQLearning]:
    obs_n, info_n = env.reset(return_info=True)
    return {
        d_id: RMQLearning(
            d_id=d_id,
            actions=list(range(env.action_space(d_id).n)),
            initial_costs=info_n[d_id]['free_flow_travel_times'],
            extrapolate_costs=True,
            policy=policy
        )
        for d_id in env.agents
    }


def get_tq_learning_agents(env: RouteChoicePZ, policy: Policy) -> Dict[AgentID, TQLearning]:
    obs_n, info_n = env.reset(return_info=True)
    return {
        d_id: TQLearning(
            d_id=d_id,
            actions=list(range(env.action_space(d_id).n)),
            extrapolate_costs=False,
            policy=policy
        )
        for d_id in env.agents
    }


def get_gtq_learning_agents(env: RouteChoicePZ, policy: Policy) -> Dict[AgentID, GTQLearning]:
    obs_n, info_n = env.reset(return_info=True)
    return {
        d_id: GTQLearning(
            d_id=d_id,
            actions=list(range(env.action_space(d_id).n)),
            extrapolate_costs=False,
            # preference_money_over_time=env.get_driver_preference_money_over_time(d_id),
            policy=policy
        )
        for d_id in env.agents
    }


def simulate_dpg(
        alg,
        net,
        k,
        alpha_decay,
        min_alpha,
        epsilon_decay,
        min_epsilon,
        agent_vehicles_factor,
        revenue_redistribution_rate,
        preference_dist_name,
        episodes,
        seed,
        render
):

    # SIMULATION SETUP
    # ----------------
    if seed:
        np.random.seed(seed)

    route_filename = None
    if net in ['BBraess_1_2100_10_c1_2100', 'BBraess_3_2100_10_c1_900', 'BBraess_5_2100_10_c1_900', 'BBraess_7_2100_10_c1_900']:
        route_filename = f"{net}.TRC.routes"

    # learning rate
    alpha = 1.0

    # initiate environment
    env = RouteChoicePZ(
        net,
        k,
        agent_vehicles_factor,
        revenue_redistribution_rate=revenue_redistribution_rate,
        preference_dist_name=preference_dist_name,
        route_filename=route_filename
        )

    # instantiate global policy
    epsilon = 1.0
    policy = EpsilonGreedy(epsilon, min_epsilon)

    # instantiate learning agents as drivers
    if alg == 'RMQLearning':
        drivers = get_rmq_learning_agents(env, policy)
    elif alg == 'TQLearning':
        drivers = get_tq_learning_agents(env, policy)
    elif alg == 'GTQLearning':
        drivers = get_gtq_learning_agents(env, policy)

    # define metric to observe
    global _metric
    _metric = 'flow'


    # CONTROLS SETUP
    # --------------
    global _pause
    _pause = False

    def pause(sender, app_data):
        global _pause
        _pause = True

    def resume(sender, app_data):
        global _pause
        _pause = False

    def step(sender, app_data):
        global _pause
        _pause = True

    def update_metric(sender, app_data, user_data):
        global _metric
        _metric = user_data['metric']
        print(_metric)


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
        # dpg.set_axis_limits("y_axis", 0, max(plotdatay))


    # GRAPHICS SETUP
    # -------------
    _win_width = WIN_SIZE[0]
    _win_height = WIN_SIZE[1]
    GRID_SIZE = GRID_SIZES[env.road_network.name]

    dpg.create_context()

    _drawlist_winsize = (_win_width * 0.8, _win_height - 56)  # subtract 56 to account for window title bar
    with dpg.window(label="Main Window", tag="main_window") as main_window:
        dpg.set_primary_window(main_window, True)


        # LEFT SIDE OF THE SCREEN
        # -----------------------
        with dpg.drawlist(width=_win_width * 0.8, height=_win_height-56, tag="drawlist") as drawlist:
            build_scene(env.road_network, _drawlist_winsize, GRID_SIZE, _metric)

        # RIGHT SIDE OF THE SCREEN
        # ------------------------
        _lateral_pos = (_win_width * 0.8, 0)
        _lateral_width = _win_width * 0.2
        _lateral_height = _win_height
        with dpg.window(label="lateral_menu", tag="lateral_menu", pos=_lateral_pos, width=_lateral_width, height=_lateral_height, \
                    no_move=True, no_close=True, no_resize=True, no_collapse=True, no_title_bar=True) as lateral_menu:

            # INFORMATION
            dpg.add_text("RouteChoiceEnv", tag="title", parent=lateral_menu)
            dpg.add_text(f"Network: {net}", tag="network", parent=lateral_menu)
            dpg.add_text(f"Total flow: {env.max_num_agents}", tag="flow", parent=lateral_menu)
            dpg.add_text(f"Episode: 0 / {episodes}", tag="episode", parent=lateral_menu)
            dpg.add_text(f"Average {_metric}", tag="metric", parent=lateral_menu)


            # CONTROLS
            dpg.add_text("(P) - Pause", parent=lateral_menu)
            dpg.add_text("(R) - Resume", parent=lateral_menu)
            dpg.add_text("(S) - Step", parent=lateral_menu)

            dpg.add_text("Metric Selection", parent=lateral_menu)
            dpg.add_text("(F) - Flow", parent=lateral_menu)
            dpg.add_text("(C) - Cost", parent=lateral_menu)
            dpg.add_text("(T) - Travel Time", parent=lateral_menu)

            with dpg.handler_registry():
                dpg.add_key_press_handler(key=dpg.mvKey_P, callback=pause)
                dpg.add_key_press_handler(key=dpg.mvKey_R, callback=resume)
                dpg.add_key_press_handler(key=dpg.mvKey_S, callback=step)

                dpg.add_key_press_handler(key=dpg.mvKey_F, callback=update_metric, user_data={'metric': 'flow'})
                dpg.add_key_press_handler(key=dpg.mvKey_C, callback=update_metric, user_data={'metric': 'cost'})
                dpg.add_key_press_handler(key=dpg.mvKey_T, callback=update_metric, user_data={'metric': 'travel_time'})


            # PLOT
            _plot_pos = (0, _lateral_height*0.6)
            _plot_width = _lateral_width * 0.94
            _plot_height = _lateral_height * 0.34
            with dpg.plot(label=f"Average Reward - {alg}", pos=_plot_pos, width=_plot_width, height=_plot_height, tag="avg_travel_time", parent=lateral_menu):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="Episodes", tag="x_axis")
                dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis")

                dpg.set_axis_limits("x_axis", 0, episodes)

                dpg.add_line_series(plotdatax, plotdatay, parent="y_axis", tag="series")


    # WINDOW SETUP
    # --------------
    dpg.create_viewport(title='RouteChoiceEnv', width=_win_width, height=_win_height)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)


    # START SIMULATION
    # ----------------
    done = False
    while dpg.is_dearpygui_running():  # dpg.start_dearpygui()

        best = float('inf')
        if not done:
            for _ in range(episodes):
                while _pause:
                    continue

                # query for action from each agent's policy
                act_n = {d_id: drivers[d_id].choose_action() for d_id in env.agents}

                # update global policy
                policy.update(epsilon_decay)

                # step environment
                obs_n, reward_n, terminal_n, truncated_n, info_n = env.step(act_n)

                # if render:
                dpg.set_value("episode", f"Episode: {_} / {episodes}")
                dpg.set_value("metric", f"Average {_metric}: { round(env.avg_travel_time, 2) }")
                update_series(env)
                update_scene(env.road_network, _metric, _drawlist_winsize)

                # test for best avg travel time
                if env.avg_travel_time < best:
                    best = env.avg_travel_time

                # update strategy (Q table)
                for d_id in drivers.keys():
                    drivers[d_id].update_strategy(obs_n[d_id], reward_n[d_id], info_n[d_id], alpha=alpha)

                # update global learning rate (alpha)
                if alpha > min_alpha:
                    alpha = alpha * alpha_decay
                else:
                    alpha = min_alpha

                solution = env.road_network_flow_distribution
                env.reset()

                dpg.render_dearpygui_frame()

        done = True
        env.close()
        dpg.render_dearpygui_frame()


    dpg.destroy_context()
