from mpc.functional.car import Car
from mpc.functional.cost_function import CSCostFunction
from mpc.functional.dynamics import Dynamics
from mpc.functional.obstacle import Obstacle
from mpc.functional.constraints import CSConstraints
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
        if define_mx:
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

        self.car = car
        self.obstacle = obstacle
        self.dynamics = dynamics
        self.show_evaluation = show_evaluation
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
        j = self.cost_function.cost_sym
        w = self.constraints.w
        lbw = self.constraints.lbw
        ubw = self.constraints.ubw
        w0 = self.constraints.w0
        g = self.constraints.g
        lbg = self.constraints.lbg
        ubg = self.constraints.ubg

        # Create an NLP solver
        prob = {'f': j, 'x': vertcat(*w), 'g': vertcat(*g)}
        solver = nlpsol('solver', 'ipopt', prob)

        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        w_opt = sol['x'].full().flatten()

        x1_opt = w_opt[0::6]
        x2_opt = w_opt[1::6]
        x3_opt = w_opt[2::6]
        x4_opt = w_opt[3::6]
        u1_opt = w_opt[4::6]
        u2_opt = w_opt[5::6]

        if self.show_evaluation:
            u = vertcat(DM(u1_opt).T, DM(u2_opt).T)
            x = vertcat(DM(x1_opt).T, DM(x2_opt).T, DM(x3_opt).T, DM(x4_opt).T)
            print(self.cost_function.eval_velocity_cost(u))
            print(self.cost_function.eval_terminal_cost(x))
            # print(self.cost_function.eval_intersection_cost(x))

        return x1_opt, x2_opt, x3_opt, x4_opt, u1_opt, u2_opt

    def iterative_mpc_control(self):
        pass
