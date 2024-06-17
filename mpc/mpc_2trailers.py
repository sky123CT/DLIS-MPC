from mpc.functional.car import Car
from mpc.functional.cost_function_2trailers import CSCostFunction
from mpc.functional.dynamics import Dynamics
from mpc.functional.obstacle import Obstacle
from mpc.functional.constraints_2trailers import CSConstraints
# from mpc.functional.constraints_ODS import CSConstraints
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
                 show_evaluation=True):

        self.car = car
        self.obstacle = obstacle
        self.dynamics = dynamics

        self.define_mx = define_mx
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
        else:
            self.x = SX.sym('x', self.NX, self.T+1)
            self.u = SX.sym('u', self.NU, self.T)

        self.x_ref = ref_state
        self.x0 = init_state
        self.cost_function = CSCostFunction('cost_function',
                                            obstacle=self.obstacle,
                                            car=self.car,
                                            state=self.x,
                                            state_ref=self.x_ref,
                                            initial_state=self.x0,
                                            control_input=self.u,
                                            horizon=self.T,
                                            is_horizon=self.T_is)
        self.constraints = CSConstraints(obstacle=self.obstacle,
                                         dynamics=self.dynamics,
                                         car=self.car,
                                         state=self.x,
                                         initial_state=self.x0,
                                         control_input=self.u,
                                         horizon=self.T,
                                         is_horizon=self.T_is,
                                         dt=self.DT)

    def one_iter_mpc_control(self):
        j = self.cost_function.cost_sym_2trailer
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
            'ipopt.max_iter': 50,
            'ipopt.print_level': 0,
            'print_time': 1,
            'ipopt.acceptable_tol': 1,
            'ipopt.acceptable_obj_change_tol': 1,
        }
        solver = nlpsol('solver', 'ipopt', prob, opti_setting)

        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        w_opt = sol['x'].full().flatten()

        x1_opt = w_opt[0::7]
        x2_opt = w_opt[1::7]
        x3_opt = w_opt[2::7]
        x4_opt = w_opt[3::7]
        x5_opt = w_opt[4::7]
        u1_opt = w_opt[5::7]
        u2_opt = w_opt[6::7]

        x = vertcat(DM(x1_opt).T, DM(x2_opt).T, DM(x3_opt).T, DM(x4_opt).T, DM(x5_opt).T)

        print("intersection_cost: ", self.cost_function.eval_intersection_cost_2trailer(x))
        print("stage_cost: ", self.cost_function.stage_cost_function(x))
        print("terminal_cost: ", self.cost_function.terminal_cost_function(x))

        return x1_opt, x2_opt, x3_opt, x4_opt, x5_opt, u1_opt, u2_opt
