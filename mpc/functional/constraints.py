import numpy as np
from casadi import *
from .obstacle import Obstacle as OBS
from .dynamics import Dynamics


class CSConstraints:
    def __init__(self,
                 obstacle: OBS,
                 dynamics: Dynamics,
                 state,
                 initial_state,
                 control_input,
                 horizon=10,
                 is_horizon=5,
                 dt=0.1,
                 max_rotation=np.deg2rad(90.0),
                 max_velocity=0.2,
                 jackknife=np.deg2rad(45.0),
                 ):

        self.obstacle = obstacle
        self.dynamics = dynamics

        # Variables and parameters
        self.x_sym = state
        self.x0 = initial_state
        self.u_sym = control_input
        self.T = horizon
        self.T_is = is_horizon
        self.DT = dt

        # Control limits
        self.MAX_OMEGA = max_rotation  # maximum rotation speed [rad/s]
        self.MAX_SPEED = max_velocity  # 55.0 / 3.6  # maximum speed [m/s]
        self.JACKKNIFE_CON = jackknife  # [degrees]

        self.w, self.w0, self.lbw, self.ubw, self.g, self.lbg, self.ubg = self.__define_constraints_variables()

    def __define_constraints_variables(self):
        optimizing_variables = []
        initial_guessing = []
        lower_boundary_variables = []
        upper_boundary_variables = []
        constraints = []
        lower_boundary_constraints = []
        upper_boundary_constraints = []

        # initial constraints
        optimizing_variables += [self.x_sym[:, 0]]
        lower_boundary_variables += [self.x0[0], self.x0[1], self.x0[2], self.x0[3]]
        upper_boundary_variables += [self.x0[0], self.x0[1], self.x0[2], self.x0[3]]
        initial_guessing += [self.x0[0], self.x0[1], self.x0[2], self.x0[3]]

        for t in range(self.T):
            # input constraints
            optimizing_variables += [self.u_sym[:, t]]
            lower_boundary_variables += [-self.MAX_SPEED, -self.MAX_OMEGA]
            upper_boundary_variables += [self.MAX_SPEED, self.MAX_OMEGA]
            initial_guessing += [0, 0]

            # states constraints
            optimizing_variables += [self.x_sym[:, t+1]]
            lower_boundary_variables += [-inf, -inf, -inf, -inf]
            upper_boundary_variables += [inf, inf, inf, inf]
            initial_guessing += [0, 0, 0, 0]

            # dynamics constraints
            constraints += [self.x_sym[3, t + 1] - self.x_sym[2, t + 1]]
            lower_boundary_constraints += [-self.JACKKNIFE_CON]
            upper_boundary_constraints += [self.JACKKNIFE_CON]

            constraints += [self.x_sym[:, t+1] -
                            self.dynamics.dynamics_mapping(x0=self.x_sym[:, t], p=self.u_sym[:, t])['xf']]
            lower_boundary_constraints += [0, 0, 0, 0]
            upper_boundary_constraints += [0, 0, 0, 0]

            # cbf constraints
            """
            for i in range(len(self.obstacle.obstacle_list)):
                if t < self.T - self.T_is:
                    if self.obstacle.obstacle_list[i]["shape"] == "polygon":
                        _, last_expand_rate, _, _ = self.obstacle.expand_polygon_as_circle(self.x_sym[:, t])
                        _, new_expand_rate, _, _ = self.obstacle.expand_polygon_as_circle(self.x_sym[:, t + 1])
                        constraints += [new_expand_rate - (1 - self.obstacle.lam) * (last_expand_rate - 1) + 1]
                    elif self.obstacle.obstacle_list[i]["shape"] == "circle":
                        last_expanded_radius = self.obstacle.cbf_calculate_obstacle_expansion(
                            robot_position=self.x_sym[:2, t])
                        new_expanded_radius = self.obstacle.cbf_calculate_obstacle_expansion(
                            robot_position=self.x_sym[:2, t + 1])
                        constraints += [(new_expanded_radius - self.obstacle.obstacle_list[i]["radius"]) -
                                        (1 - self.obstacle.lam) *
                                        (last_expanded_radius - self.obstacle.obstacle_list[i]["radius"])]
                    lower_boundary_constraints += [0]
                    upper_boundary_constraints += [inf]
            """
            if t < self.T - self.T_is:
                last_expanded_radius = self.obstacle.cbf_calculate_obstacle_expansion(
                    robot_position=self.x_sym[:2, t])
                new_expanded_radius = self.obstacle.cbf_calculate_obstacle_expansion(
                    robot_position=self.x_sym[:2, t + 1])
                for i in range(len(self.obstacle.obstacle_list)):
                    constraints += [(new_expanded_radius[i] - self.obstacle.obstacle_list[i]["radius"]) -
                                    (1 - self.obstacle.lam) *
                                    (last_expanded_radius[i] - self.obstacle.obstacle_list[i]["radius"])]
                    lower_boundary_constraints += [0]
                    upper_boundary_constraints += [inf]

        return (optimizing_variables,
                initial_guessing,
                lower_boundary_variables,
                upper_boundary_variables,
                constraints,
                lower_boundary_constraints,
                upper_boundary_constraints)







