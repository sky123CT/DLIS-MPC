from mpc.functional.car import Car
from mpc.functional.cost_function_test import CSCostFunction
from mpc.functional.dynamics import Dynamics
from mpc.functional.obstacle import Obstacle
from mpc.functional.constraints_test import CSConstraints
from casadi import *
import numpy as np


class MPC:
    def __init__(self,
                 car: Car,
                 obstacle: Obstacle,
                 dynamics: Dynamics,
                 ref_state,
                 init_state,
                 nx=4,
                 nu=2,
                 dt=0.1,
                 horizon=10,
                 horizon_is=5,
                 define_mx=True,
                 show_evaluation=True,
                 cbf_slack_activate=False):

        self.car = car
        self.obstacle = obstacle
        self.dynamics = dynamics

        self.define_mx = define_mx
        self.cbf_slack_activate = cbf_slack_activate
        self.show_evaluation = show_evaluation

        # iterative parameter
        self.MAX_ITER = 5  # Max iteration
        self.DU_TH = 0.1  # 0.1  # iteration finish param
        self.DT = dt  # [s] time tick

        # Model parameters
        self.NX = nx  # x = x, y, yaw, yawt
        self.NU = nu  # u = [v, w]
        self.T = horizon  # horizon length 5
        self.T_is = horizon_is

        # Variables
        if self.define_mx:
            states = []
            for t in range(self.T+1):
                states_t = []
                for i in range(self.NX):
                    states_t.append(MX.sym('x'+str(i)+'_'+str(t)))
                states.append(vertcat(*states_t))
            self.x = horzcat(*states)
            inputs = []
            for t in range(self.T):
                inputs_t = []
                for i in range(self.NU):
                    inputs_t.append(MX.sym('u' + str(i) + '_' + str(t)))
                inputs.append(vertcat(*inputs_t))
            self.u = horzcat(*inputs)
            if self.cbf_slack_activate:
                cbf_slacks = []
                for t in range(self.T):
                    cbf_slack_t = []
                    for i in range(len(self.obstacle.obstacle_list)):
                        cbf_slack_t.append(MX.sym('cbf_slack' + str(i) + '_' + str(t)))
                    cbf_slacks.append(vertcat(*cbf_slack_t))
                self.cbf_slacks = horzcat(*cbf_slacks)
        else:
            self.x = SX.sym('x', self.NX, self.T+1)
            self.u = SX.sym('u', self.NU, self.T)
            if self.cbf_slack_activate:
                self.cbf_slacks = SX.sym('cbf_slack', self.T-self.T_is)

        self.x_ref = ref_state
        self.x0 = init_state

        if self.cbf_slack_activate:
            self.cost_function = CSCostFunction('cost_function',
                                                obstacle=self.obstacle,
                                                state=self.x,
                                                state_ref=self.x_ref,
                                                initial_state=self.x0,
                                                control_input=self.u,
                                                cbf_slack=self.cbf_slacks,
                                                horizon=self.T,
                                                is_horizon=self.T_is)
            self.constraints = CSConstraints(obstacle=self.obstacle,
                                             dynamics=self.dynamics,
                                             state=self.x,
                                             initial_state=self.x0,
                                             control_input=self.u,
                                             cbf_slack=self.cbf_slacks,
                                             horizon=self.T,
                                             is_horizon=self.T_is,
                                             dt=self.DT)
        else:
            self.cost_function = CSCostFunction('cost_function',
                                                obstacle=self.obstacle,
                                                state=self.x,
                                                state_ref=self.x_ref,
                                                initial_state=self.x0,
                                                control_input=self.u,
                                                horizon=self.T,
                                                is_horizon=self.T_is)
            self.constraints = CSConstraints(obstacle=self.obstacle,
                                             dynamics=self.dynamics,
                                             state=self.x,
                                             initial_state=self.x0,
                                             control_input=self.u,
                                             horizon=self.T,
                                             is_horizon=self.T_is,
                                             dt=self.DT)

    def one_iter_mpc_control(self):
        j = self.cost_function.cost_sym_test
        w = self.constraints.w
        lbw = self.constraints.lbw
        ubw = self.constraints.ubw
        w0 = self.constraints.w0
        g = self.constraints.g
        lbg = self.constraints.lbg
        ubg = self.constraints.ubg

        # Create an NLP solver
        prob = {'f': j, 'x': vertcat(*w), 'g': vertcat(*g)}
        opti_setting = {
            'ipopt.max_iter': 100,
            'ipopt.print_level': 0,
            'print_time': 1
        }
        solver = nlpsol('solver', 'ipopt', prob, opti_setting)

        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        w_opt = sol['x'].full().flatten()

        x1_opt = w_opt[0::6]
        x2_opt = w_opt[1::6]
        x3_opt = w_opt[2::6]
        x4_opt = w_opt[3::6]
        u1_opt = w_opt[4::6]
        u2_opt = w_opt[5::6]

        x = vertcat(DM(x1_opt).T, DM(x2_opt).T, DM(x3_opt).T, DM(x4_opt).T)
        print(self.cost_function.eval_intersection_cost(x))

        return x1_opt, x2_opt, x3_opt, x4_opt, u1_opt, u2_opt



