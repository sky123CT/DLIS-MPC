import numpy as np
from casadi import *
from . import obstacle
from .obstacle import Obstacle as OBS
from .dynamics import Dynamics
from .car import Car


class CSConstraints:
    def __init__(self,
                 obstacle: OBS,
                 dynamics: Dynamics,
                 car: Car,
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
        self.car = car

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

        self.w, self.w0, self.lbw, self.ubw, self.g, self.lbg, self.ubg = self.__define_constraints_2trailers()

    def __define_constraints_2trailers(self):
        optimizing_variables = []
        initial_guessing = []
        lower_boundary_variables = []
        upper_boundary_variables = []
        constraints = []
        lower_boundary_constraints = []
        upper_boundary_constraints = []

        optimizing_variables += [self.x_sym[:, 0]]
        lower_boundary_variables += [self.x0[0], self.x0[1], self.x0[2], self.x0[3], self.x0[4]]
        upper_boundary_variables += [self.x0[0], self.x0[1], self.x0[2], self.x0[3], self.x0[4]]
        initial_guessing += [self.x0[0], self.x0[1], self.x0[2], self.x0[3], self.x0[4]]

        for t in range(self.T):
            optimizing_variables += [self.u_sym[:, t]]
            lower_boundary_variables += [-self.MAX_SPEED, -self.MAX_OMEGA]
            upper_boundary_variables += [-self.MAX_SPEED, self.MAX_OMEGA]
            initial_guessing += [0, 0]

            optimizing_variables += [self.x_sym[:, t + 1]]
            lower_boundary_variables += [-inf, -inf, -inf, -inf, -inf]
            upper_boundary_variables += [inf, inf, inf, inf, inf]
            initial_guessing += [0, 0, 0, 0, 0]

            constraints += [self.x_sym[3, t + 1] - self.x_sym[2, t + 1]]
            lower_boundary_constraints += [-self.JACKKNIFE_CON]
            upper_boundary_constraints += [self.JACKKNIFE_CON]
            constraints += [self.x_sym[4, t + 1] - self.x_sym[3, t + 1]]
            lower_boundary_constraints += [-self.JACKKNIFE_CON]
            upper_boundary_constraints += [self.JACKKNIFE_CON]

            constraints += [self.x_sym[:, t + 1] -
                            self.dynamics.dynamics_mapping(x0=self.x_sym[:, t], p=self.u_sym[:, t])['xf']]
            lower_boundary_constraints += [0, 0, 0, 0, 0]
            upper_boundary_constraints += [0, 0, 0, 0, 0]

            cbf_deactivate = []
            car_data_t0, _ = self.car.get_full_data()
            for i in range(len(self.obstacle.obstacle_list)):
                cbf_deactivate1 = self.obstacle.cbf_deactivate(
                    obstacle_center=self.obstacle.obstacle_list[i]["center"],
                    robot_center=car_data_t0['trailer1'][:2],
                    robot_orientation=car_data_t0['trailer1'][2])
                cbf_deactivate2 = self.obstacle.cbf_deactivate(
                    obstacle_center=self.obstacle.obstacle_list[i]["center"],
                    robot_center=car_data_t0['trailer2'][:2],
                    robot_orientation=car_data_t0['trailer2'][2])
                cbf_deactivate.append([cbf_deactivate1, cbf_deactivate2])
                if t==0:
                    print(cbf_deactivate)

                car_data_symbolic_t, _ = self.car.get_full_data_symbolic(self.x_sym[:, t])
                car_data_symbolic_t_next, _ = self.car.get_full_data_symbolic(self.x_sym[:, t + 1])
                if not cbf_deactivate[i][0]:
                    last_expanded_radius1 = self.obstacle.cbf_calculate_obstacle_expansion(
                        robot_position=car_data_symbolic_t['trailer1'][:2])
                    new_expanded_radius1 = self.obstacle.cbf_calculate_obstacle_expansion(
                        robot_position=car_data_symbolic_t_next['trailer1'][:2])
                    constraints += [(new_expanded_radius1[i] - self.obstacle.obstacle_list[i]["radius"]) -
                                    (1 - self.obstacle.lam) *
                                    (last_expanded_radius1[i] - self.obstacle.obstacle_list[i]["radius"])]
                    lower_boundary_constraints += [0]
                    upper_boundary_constraints += [inf]
                else:
                    pass

                if not cbf_deactivate[i][1]:
                    last_expanded_radius2 = self.obstacle.cbf_calculate_obstacle_expansion(
                        robot_position=car_data_symbolic_t['trailer2'][:2])
                    new_expanded_radius2 = self.obstacle.cbf_calculate_obstacle_expansion(
                        robot_position=car_data_symbolic_t_next['trailer2'][:2])
                    constraints += [(new_expanded_radius2[i] - self.obstacle.obstacle_list[i]["radius"]) -
                                    (1 - self.obstacle.lam) *
                                    (last_expanded_radius2[i] - self.obstacle.obstacle_list[i]["radius"])]
                    lower_boundary_constraints += [0]
                    upper_boundary_constraints += [inf]
                else:
                    pass

        return (optimizing_variables,
                initial_guessing,
                lower_boundary_variables,
                upper_boundary_variables,
                constraints,
                lower_boundary_constraints,
                upper_boundary_constraints)
