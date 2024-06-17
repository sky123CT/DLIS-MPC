from casadi import *
from .obstacle import Obstacle as OBS
from .CasadiDLIS import CasadiDLIS as CSDLIS
from ..utility import *


class CSCostFunction(Callback):
    def __init__(self, name,
                 obstacle: OBS,
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

        # Variables and parameters
        self.x_sym = state
        self.x_ref = state_ref
        self.x0 = initial_state
        self.u_sym = control_input
        self.cbf_slack_sym = cbf_slack
        self.T = horizon
        self.T_is = is_horizon

        # cost function parameters
        self.W_is = 100  # intersection weight
        self.Q = np.diag([1, 1, 0.01, 0.01])
        self.R = np.diag([0.0001, 0.0001])
        self.Qf = np.diag([100, 100, 0.01, 0.1])

        self.GOAL_DIS = 1.1  # goal distance
        self.STOP_SPEED = 0.5 / 3.6  # stop speed
        self.MAX_TIME = 500.0  # max simulation time

        # define symbolic cost functions
        self.stage_cost_sym, self.stage_cost_sym_function = self.__define_stage_cost()
        self.terminal_cost_sym, self.terminal_cost_function = self.__define_terminal_cost()
        self.velocity_cost_sym, self.velocity_cost_function = self.__define_velocity_cost()
        self.intersection_cost_sym, self.intersection_cost_function = self.__define_intersection_cost_test()

        is_cost_deactivate = []
        for i in range(len(self.obstacle.obstacle_list)):
            is_cost_deactivate.append(
                self.obstacle.cbf_deactivate(obstacle_center=self.obstacle.obstacle_list[i]["center"],
                                             robot_center=[self.x0[0], self.x0[1]],
                                             robot_orientation=self.x0[3]))
            if is_cost_deactivate[i]:
                is_deact_multiplier = 0
            else:
                is_deact_multiplier = 1

        #print(is_deact_multiplier)
        self.cost_sym_test = (self.intersection_cost_sym * is_deact_multiplier +
                              self.stage_cost_sym +
                              self.terminal_cost_sym +
                              self.velocity_cost_sym)

    def __define_stage_cost(self):
        stage_cost_sym = 0
        stage_cost_sym += ((self.x_ref[:, self.T] - self.x_sym[:, self.T]).T @
                           self.Q @
                           (self.x_ref[:, self.T] - self.x_sym[:, self.T]))
        stage_cost_function = Function('stage_cost', [self.x_sym], [stage_cost_sym])
        return stage_cost_sym, stage_cost_function

    def __define_terminal_cost(self):
        terminal_cost_sym = ((self.x_ref[:, self.T] - self.x_sym[:, self.T]).T @
                             self.Qf @
                             (self.x_ref[:, self.T] - self.x_sym[:, self.T]))
        terminal_cost_function = Function('terminal_cost', [self.x_sym], [terminal_cost_sym])
        return terminal_cost_sym, terminal_cost_function

    def __define_velocity_cost(self):
        velocity_cost_sym = 0
        for t in range(self.T):
            velocity_cost_sym += self.u_sym[:, t].T @ self.R @ self.u_sym[:, t]
        velocity_cost_function = Function('velocity_cost', [self.u_sym], [velocity_cost_sym])
        return velocity_cost_sym, velocity_cost_function

    def __define_intersection_cost_test(self):
        intersection_cost = 0
        for i in range(len(self.obstacle.obstacle_list)):
            if self.obstacle.obstacle_list[i]["shape"] == "polygon":
                pass
                """
                for t in range(self.T - self.T_is):
                    last_expanded_vertices, last_expand_rate, obstacle_init_radius, i_vertices = (
                        self.obstacle.expand_polygon_as_circle(self.x_sym[:, t + self.T_is - 1]))
                    new_expand_rate = (1 - self.obstacle.lam) * (last_expand_rate - 1) + 1
                    new_expand_vertices = (mtimes(new_expand_rate, (self.obstacle.vertices - self.obstacle.center_point)) +
                                        repmat(self.obstacle.center_point, 1, 4).T)
                    relative_position_sym = new_expand_vertices.reshape((-1, 1)) - repmat(self.x_sym[:2, t], 4, 1)
                    input_sym = vertcat(relative_position_sym, self.x_sym[-2:, t])
                    is_area = self.DLIS.model(input_sym)
                    intersection_cost += is_area
                """
            elif self.obstacle.obstacle_list[i]["shape"] == "circle":
                for t in range(self.T - self.T_is):
                    last_expanded_radius = (
                        self.obstacle.cbf_calculate_obstacle_expansion(robot_position=self.x_sym[:2, t + self.T_is - 1])[i]
                    )
                    new_expanded_radius = ((1 - self.obstacle.lam) *
                                           (last_expanded_radius - self.obstacle.obstacle_list[i]["radius"]) +
                                           self.obstacle.obstacle_list[i]["radius"])
                    relative_position_sym = self.obstacle.obstacle_list[i]["center"] - self.x_sym[:2, t]

                    input_sym = vertcat(relative_position_sym,
                                        new_expanded_radius**2,
                                        -self.x_sym[2, t],
                                        -self.x_sym[3, t]
                                        )

                    """
                    input_sym = vertcat(relative_position_sym,
                                        new_expanded_radius ** 2,
                                        -(self.x_sym[2, t] + self.x_sym[3, t])
                                        )
                    """

                    is_area = self.DLIS.model(input_sym) * self.W_is
                    intersection_cost += is_area


            else:
                raise ValueError("Obstacle shape not given or not fit!")
        intersection_cost_function = Function('intersection_cost', [self.x_sym], [intersection_cost])
        return intersection_cost, intersection_cost_function

    def eval_intersection_cost(self, x):
        intersection_cost = self.intersection_cost_function(x)
        return intersection_cost
