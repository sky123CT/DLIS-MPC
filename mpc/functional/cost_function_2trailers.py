from casadi import *
from .car import Car
from .obstacle import Obstacle as OBS
from .CasadiDLIS import CasadiDLIS2Trailers as CSDLIS
from ..utility import *


class CSCostFunction(Callback):
    def __init__(self, name,
                 obstacle: OBS,
                 car: Car,
                 state,
                 state_ref,
                 initial_state,
                 control_input,
                 cbf_slack=None,
                 horizon=10,
                 is_horizon=5,
                 opts={}):
        Callback.__init__(self)
        self.construct(name, opts)

        self.DLIS = CSDLIS()
        self.obstacle = obstacle
        self.car = car

        # Variables and parameters
        self.x_sym = state
        self.x_ref = state_ref
        self.x0 = initial_state
        self.u_sym = control_input
        self.cbf_slack_sym = cbf_slack
        self.T = horizon
        self.T_is = is_horizon

        # cost function parameters
        self.W_is = np.array([100, 200]).reshape(-1, 1)  # intersection weight
        # self.W_is = np.array([0, 0]).reshape(-1, 1)
        self.R = np.diag([0.0001, 0.0001])
        self.W_angle_diff = np.diag([100, 1])
        self.W_stage = np.diag([20, 50, 0.1, 0.1, 1])
        # self.W_stage = np.diag([5, 30, 0.01, 0.01, 1])  # stage weight
        self.W_terminal = np.diag([1000, 3500, 10, 10, 1000])  # terminal weight
        # self.W_stage = np.diag([50, 50, 50, 10, 20, 5, 0, 0, 0.01])  # stage weight
        # self.W_terminal = np.diag([500, 500, 100, 10000, 10000, 2000, 0, 0, 0.1])  # terminal weight

        is_cost_deactivate = []
        car_data_t0, _ = self.car.get_full_data()
        for i in range(len(self.obstacle.obstacle_list)):
            is_cost_deactivate1 = self.obstacle.cbf_deactivate(
                obstacle_center=self.obstacle.obstacle_list[i]["center"],
                robot_center=car_data_t0['trailer1'][:2],
                robot_orientation=car_data_t0['trailer1'][2])
            is_cost_deactivate2 = self.obstacle.cbf_deactivate(
                obstacle_center=self.obstacle.obstacle_list[i]["center"],
                robot_center=car_data_t0['trailer2'][:2],
                robot_orientation=car_data_t0['trailer2'][2])
            is_cost_deactivate.append([is_cost_deactivate1, is_cost_deactivate2])

            if is_cost_deactivate[i][1]:
                is_deact_multiplier1 = 0
                is_deact_multiplier2 = 0.0001

            else:
                is_deact_multiplier1 = 1
                is_deact_multiplier2 = 1
                if is_cost_deactivate[i][0]:
                    is_deact_multiplier1 = 0

        is_deact_multiplier = np.array([is_deact_multiplier1, is_deact_multiplier2]).reshape(-1, 1)
        print(is_deact_multiplier)
        # define symbolic cost functions

        self.stage_cost_sym, self.stage_cost_function = self.__define_stage_cost_2trailer()
        self.terminal_cost_sym, self.terminal_cost_function = self.__define_terminal_cost_2trailer()
        self.angle_cost_sym, self.angle_cost_function = self.__define_angle_cost_2trailer()
        self.velocity_cost_sym, self.velocity_cost_function = self.__define_velocity_cost_2trailer()
        self.intersection_cost_sym, self.intersection_cost_function = self.__define_intersection_cost_2trailer(
            is_multiplier=is_deact_multiplier)

        self.cost_sym_2trailer = (self.intersection_cost_sym[0] +
                                  self.intersection_cost_sym[1] +
                                  self.stage_cost_sym +
                                  self.terminal_cost_sym +
                                  self.angle_cost_sym +
                                  self.velocity_cost_sym)

    def __define_stage_cost_2trailer(self):
        stage_cost_sym = 0
        for t in range(self.T):
            """
            car_reference_data, _ = self.car.get_full_data_symbolic(self.x_ref[:, t])
            car_current_data, _ = self.car.get_full_data_symbolic(self.x_sym[:, t])
            x_ref1 = car_reference_data['trailer1']
            x_ref2 = car_reference_data['trailer2']
            x_ref_t = car_reference_data['tractor']
            x_sym1 = car_current_data['trailer1']
            x_sym2 = car_current_data['trailer2']
            x_sym_t = car_current_data['tractor']
            x_ref = vertcat(x_ref1, x_ref2, x_ref_t)
            x_sym = vertcat(x_sym1, x_sym2, x_sym_t)

            stage_cost_sym += (x_ref - x_sym).T @ self.W_stage @ (x_ref - x_sym)
            """
            stage_cost_sym += ((self.x_ref[:, t] - self.x_sym[:, t]).T @
                               self.W_stage @
                               (self.x_ref[:, t] - self.x_sym[:, t]))

        stage_cost_function = Function('stage_cost', [self.x_sym], [stage_cost_sym])
        return stage_cost_sym, stage_cost_function

    def __define_terminal_cost_2trailer(self):
        """
        car_reference_data, _ = self.car.get_full_data_symbolic(self.x_ref[:, self.T])
        car_terminal_data, _ = self.car.get_full_data_symbolic(self.x_sym[:, self.T])
        x_ref1 = car_reference_data['trailer1']
        x_ref2 = car_reference_data['trailer2']
        x_ref_t = car_reference_data['tractor']
        x_sym1 = car_terminal_data['trailer1']
        x_sym2 = car_terminal_data['trailer2']
        x_sym_t = car_terminal_data['tractor']
        x_ref = vertcat(x_ref1, x_ref2, x_ref_t)
        x_sym = vertcat(x_sym1, x_sym2, x_sym_t)
        """

        terminal_cost_sym = ((self.x_ref[:, self.T] - self.x_sym[:, self.T]).T @
                             self.W_terminal @
                             (self.x_ref[:, self.T] - self.x_sym[:, self.T]))
        terminal_cost_function = Function('terminal_cost', [self.x_sym], [terminal_cost_sym])
        return terminal_cost_sym, terminal_cost_function

    def __define_angle_cost_2trailer(self):
        angle_cost_sym = 0
        for t in range(self.T):
            angle_cost_sym += (vertcat((self.x_sym[4, t]-self.x_sym[3, t]), (self.x_sym[3, t]-self.x_sym[2, t])).T @
                               self.W_angle_diff @
                               vertcat((self.x_sym[4, t]-self.x_sym[3, t]), (self.x_sym[3, t]-self.x_sym[2, t])))
        angle_cost_function = Function('terminal_cost', [self.x_sym], [angle_cost_sym])
        return angle_cost_sym, angle_cost_function

    def __define_velocity_cost_2trailer(self):
        velocity_cost_sym = 0
        for t in range(self.T):
            velocity_cost_sym += self.u_sym[:, t].T @ self.R @ self.u_sym[:, t]
        velocity_cost_function = Function('velocity_cost', [self.u_sym], [velocity_cost_sym])
        return velocity_cost_sym, velocity_cost_function

    def __define_intersection_cost_2trailer(self, is_multiplier):
        intersection_cost = 0
        for i in range(len(self.obstacle.obstacle_list)):
            if self.obstacle.obstacle_list[i]["shape"] == "polygon":
                pass
            elif self.obstacle.obstacle_list[i]["shape"] == "circle":
                for t in range(self.T - self.T_is):
                    car_data_symbolic_t4, _ = self.car.get_full_data_symbolic(self.x_sym[:, t + self.T_is - 1])
                    car_data_symbolic_t0, _ = self.car.get_full_data_symbolic(self.x_sym[:, t])
                    last_expanded_radius1 = self.obstacle.cbf_calculate_obstacle_expansion(
                            robot_position=car_data_symbolic_t4['trailer1'][:2])[i]
                    last_expanded_radius2 = self.obstacle.cbf_calculate_obstacle_expansion(
                            robot_position=car_data_symbolic_t4['trailer2'][:2])[i]

                    new_expanded_radius1 = ((1 - self.obstacle.lam) *
                                            (last_expanded_radius1 - self.obstacle.obstacle_list[i]["radius"]) +
                                            self.obstacle.obstacle_list[i]["radius"])
                    new_expanded_radius2 = ((1 - self.obstacle.lam) *
                                            (last_expanded_radius2 - self.obstacle.obstacle_list[i]["radius"]) +
                                            self.obstacle.obstacle_list[i]["radius"])
                    relative_position_sym1 = (self.obstacle.obstacle_list[i]["center"] -
                                              car_data_symbolic_t0['trailer1'][:2])
                    relative_position_sym2 = (self.obstacle.obstacle_list[i]["center"] -
                                              car_data_symbolic_t0['trailer2'][:2])

                    input_sym = vertcat(relative_position_sym1,
                                        relative_position_sym2,
                                        new_expanded_radius1**2,
                                        new_expanded_radius2**2,
                                        self.x_sym[3, t],
                                        self.x_sym[4, t],
                                        self.x_sym[2, t]
                                        )

                    """
                    input_sym = vertcat(relative_position_sym,
                                        new_expanded_radius ** 2,
                                        -(self.x_sym[2, t] + self.x_sym[3, t])
                                        )
                    """
                    # is_area = (is_multiplier * self.W_is) * self.DLIS.model_approx_1(input_sym, self.DLIS.model_approx_param_1)
                    is_area = (is_multiplier * self.W_is) * self.DLIS.model(input_sym)
                    intersection_cost += is_area
            else:
                raise ValueError("Obstacle shape not given or not fit!")
        intersection_cost_function = Function('intersection_cost', [self.x_sym], [intersection_cost])
        return intersection_cost, intersection_cost_function

    def eval_intersection_cost_2trailer(self, x):
        intersection_cost = self.intersection_cost_function(x)
        return intersection_cost
