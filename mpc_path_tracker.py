"""

Path tracking simulation with iterative linear model predictive control

Robot model: differential driven mobile robot with passive trailer, reference point on the trailer

State variable: (x2, y2, theta1, theta2)
Control input variable: (v1, w1)

author: Yucheng Tang (@Yucheng-Tang)

Citation: Atsushi Sakai (@Atsushi_twi)
"""
import pandas as pd
import matplotlib.pyplot as plt
import math
import time
from sympy import *
from casadi import *
from mpc.path_planning import cubic_spline_planner as PathPlanner
from mpc.utility.utility import *

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                ".utility/path_planning/")

# Model parameters
NX = 4  # x = x, y, yaw, yawt
NU = 2  # u = [v, w]
T = 10  # horizon length 5

# control barrier function parameters
lam = 0.03  # barrier function param
r_f = 100  # reachable set param
circle_robot_limit = 55  # determine if the robot is tangent to a circle
T_CAL = []
MIN_DIST = []

# mpc parameters
R = np.diag([0.01, 0.01])  # input cost matrix
Rd = np.diag([0.01, 1.0])  # ([0.01, 1.0])  # input difference cost matrix
Q = np.diag([1.0, 1.0, 0.001, 0.1])
# np.diag([10.0, 10.0, 0.01, 0.01])  # state cost matrix
# on axis
# 1 for pure backwards, 10 for switch during the path
# 10, 10, 0.01, 0.1 for backwards circular movement
Qf = np.diag([100.0, 100.0, 0.01, 1])  # state final matrix
GOAL_DIS = 1.1  # goal distance
STOP_SPEED = 0.5 / 3.6  # stop speed
MAX_TIME = 500.0  # max simulation time

# iterative parameter
MAX_ITER = 5  # Max iteration
DU_TH = 0.1  # 0.1  # iteration finish param

TARGET_SPEED = 0.25  # [m/s] target speed
N_IND_SEARCH = 10  # Search index number

DT = 0.1  # [s] time tick

# Vehicle parameters
LENGTH = 0.72  # [m]
LENGTH_T = 0.36  # [m]
WIDTH = 0.48  # [m]
BACKTOWHEEL= 0.36  # [m]
WHEEL_LEN = 0.1  # [m]
WHEEL_WIDTH = 0.05  # [m]
TREAD = 0.2  # [m]
WB = 0.3  # [m]
ROD_LEN = 0.5  # [m]
CP_OFFSET = 0.0  # [m]

# Obstacle parameter
O_X = [-3.5]  # [-3]# , -4.5] # -2
O_Y = [-0.25]  # [4.2]# , -2.2] # 0.1
O_R = [1.0]  # , 0.5] # 0.5
O_D = [100000000]

# Control limits
MAX_OMEGA = np.deg2rad(90.0)  # maximum rotation speed [rad/s]

MAX_SPEED = 0.2  # 55.0 / 3.6  # maximum speed [m/s]
MIN_SPEED = - 0.2  # minimum speed [m/s]
JACKKNIFE_CON = 45.0  # [degrees]

CIR_RAD = 5  # radius of circular path [m]

TIME_LIST = []

PX_SET = []
PY_SET = []
X_SET = []
Y_SET = []

BARRIER_LIST = []

# TODO: add MODE parameter for path selection
MODE = ""

show_animation = True


# TODO: add control horizon variable
# TODO: Notebook and Git

class State:
    """
    vehicle state class
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, yawt=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.yawt = yawt
        self.predelta = None


def pi_2_pi(angle):
    """
    Normalize an angle to be within the range [-pi, pi].
    """


    while (angle > math.pi):
        angle = angle - 2.0 * math.pi

    while (angle < -math.pi):
        angle = angle + 2.0 * math.pi

    return angle


def get_linear_model_matrix(xref_t, xref_t1, x_t):
    """
    Calculate the linearized model matrices A, B, and C for vehicle dynamics.

    Args:
    xref_t: State at time t.
    xref_t1: State at time t+1.
    x_t: Current state.

    Returns:
    A: State transition matrix.
    B: Input matrix.
    C: Constant matrix.
    """

    # dyawt = (xref_t1[3]-xref_t[3])
    # dyawt is not the velocity theta2 at operation point!!!
    rp_t = np.array([xref_t[0], xref_t[1]])
    rp_t1 = np.array([xref_t1[0], xref_t1[1]])
    v_r = np.linalg.norm(rp_t1 - rp_t)
    w_r = xref_t1[2] - xref_t[2]

    A = np.zeros((NX, NX))
    A[0, 0] = 1.0
    A[1, 1] = 1.0
    A[2, 2] = 1.0
    # A[3, 3] = 1.0

    # model from pythonrobotic
    A[0, 2] = v_r * math.sin(x_t[3] - x_t[2]) * math.cos(x_t[3])
    A[0, 3] = - v_r * math.sin(2 * x_t[3] - x_t[2])
    A[1, 2] = v_r * math.sin(x_t[3] - x_t[2]) * math.sin(x_t[3])
    A[1, 3] = v_r * math.cos(2 * x_t[3] - x_t[2])
    A[3, 2] = (v_r * math.cos(x_t[3] - x_t[2]) - CP_OFFSET * w_r * math.sin(x_t[3] - x_t[2])) / ROD_LEN
    A[3, 3] = 1.0 - (v_r * math.cos(x_t[3] - x_t[2]) - CP_OFFSET * w_r * math.sin(x_t[3] - x_t[2])) / ROD_LEN

    # my model
    # A[0, 1] = DT * dyawt
    # A[1, 0] = - DT * dyawt
    # A[1, 3] = DT * v_r * math.cos(xref_t[3] - xref_t[2])
    # mdoel from paper
    # A[0, 1] = dyawt
    # A[1, 0] = - dyawt
    # A[1, 3] = v_r * math.cos(xref_t[2] - xref_t[3])

    B = np.zeros((NX, NU))

    # model from pythonrobotics
    B[0, 0] = DT * math.cos(x_t[3] - x_t[2]) * math.cos(x_t[3])
    B[1, 0] = DT * math.cos(x_t[3] - x_t[2]) * math.sin(x_t[3])
    B[2, 1] = DT
    B[3, 0] = - DT * math.sin(x_t[3] - x_t[2]) / ROD_LEN
    B[3, 1] = - DT * CP_OFFSET * math.cos(x_t[3] - x_t[2]) / ROD_LEN

    # my model
    # B[0, 0] = DT * math.cos(x_t[3] - x_t[2])
    # B[2, 1] = - DT
    # B[3, 0] = DT * math.sin(x_t[3] - x_t[2]) / ROD_LEN

    # model from paper
    # B[0, 0] = DT * math.cos(x_t[2] - x_t[3])
    # B[2, 1] = DT
    # B[3, 0] = DT * math.sin(x_t[2] - x_t[3]) / ROD_LEN
    # print("B", B[0, 0], B[3, 0])

    C = np.zeros(NX)

    # model from pythonrobotics
    C[0] = - v_r * (x_t[2] * math.sin(x_t[3] - x_t[2]) * math.cos(x_t[3]) - x_t[3] * math.sin(2 * x_t[3] - x_t[2]))
    C[1] = - v_r * (x_t[2] * math.sin(x_t[3] - x_t[2]) * math.sin(x_t[3]) + x_t[3] * math.cos(2 * x_t[3] - x_t[2]))
    C[3] = (x_t[3] - x_t[2]) * (v_r * math.cos(x_t[3] - x_t[2]) - CP_OFFSET * w_r * math.sin(x_t[3] - x_t[2])) / ROD_LEN

    # my model
    # C[0] = DT * v_r * math.cos(xref_t[3] - xref_t[2])
    # C[2] = DT * (xref_t1[2] - xref_t[2])
    # C[3] = DT * (xref_t1[3] - xref_t[3])

    # model from paper
    # C[0] = - v_r * math.cos(xref_t[2] - xref_t[3])
    # C[2] = (xref_t1[2] - xref_t[2])
    # C[3] = (xref_t1[3] - xref_t[3])
    # print("C", C[2], C[3])

    return A, B, C


# TODO: modify the plot_car to use x2, y2, yaw and yawt, plot tractor-trailer system in one function?
def plot_car(x, y, yaw, length, truckcolor="-k"):  # pragma: no cover
    """
    Plot a car's position and orientation in a 2D plot.

    Args:
    x: x-coordinate of the car's center.
    y: y-coordinate of the car's center.
    yaw: yaw angle (orientation) of the car.
    length: length of the car.
    truckcolor: color of the car's truck (default is black).
    """

    # using BACKTOWHEEL parameter
    # outline = np.array([[-BACKTOWHEEL, (length - BACKTOWHEEL), (length - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
    #                    [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    outline = np.array([[-length / 2, (length - length / 2), (length - length / 2), -length / 2, -length / 2],
                        [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    # fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
    #                      [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])

    rr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                         [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD,
                          -WHEEL_WIDTH - TREAD]])

    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                     [-math.sin(yaw), math.cos(yaw)]])

    outline = (outline.T.dot(Rot1)).T
    rr_wheel = (rr_wheel.T.dot(Rot1)).T
    rl_wheel = (rl_wheel.T.dot(Rot1)).T

    outline[0, :] += x
    outline[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    # Plot the car's outline and wheels
    plt.plot(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten(), truckcolor)
    # plt.plot(np.array(fr_wheel[0, :]).flatten(),
    #          np.array(fr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rr_wheel[0, :]).flatten(),
             np.array(rr_wheel[1, :]).flatten(), truckcolor)
    # plt.plot(np.array(fl_wheel[0, :]).flatten(),
    #          np.array(fl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rl_wheel[0, :]).flatten(),
             np.array(rl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(x, y, "*")


def update_state(state, v, dyaw):
    """
    Update the vehicle's state based on control inputs.

    Args:
    state: Current vehicle state.
    v: Velocity.
    dyaw: Change in yaw rate.

    Returns:
    state_new: Updated vehicle state.
    """

    # print("(x,y, yaw, yawt): ", state.x, state.y, state.yaw, state.yawt, " control input (v, dyaw):", v, dyaw)

    # input check
    if dyaw >= MAX_OMEGA:
        dyaw = MAX_OMEGA
    elif dyaw <= -MAX_OMEGA:
        dyaw = -MAX_OMEGA

    # print("update model: ", v, dyaw, state.yaw, state.yawt)

    # Update the vehicle's state using a kinematic model
    # my model
    # state.x = state.x + v * math.cos(state.yawt - state.yaw) * math.cos(state.yawt) * DT
    # state.y = state.y + v * math.cos(state.yawt - state.yaw) * math.sin(state.yawt) * DT
    # state.yaw = state.yaw + dyaw * DT
    # state.yawt = state.yawt - v / ROD_LEN * math.sin(state.yawt - state.yaw) * DT

    # model from paper
    state_new = State(x=state.x, y=state.y, yaw=state.yaw, yawt=state.yawt)

    state_new.x = state.x + v * math.cos(state.yaw - state.yawt) * math.cos(state.yawt) * DT
    state_new.y = state.y + v * math.cos(state.yaw - state.yawt) * math.sin(state.yawt) * DT
    state_new.yaw = state.yaw + dyaw * DT
    state_new.yawt = state.yawt + v / ROD_LEN * math.sin(state.yaw - state.yawt) * DT \
                     - CP_OFFSET * dyaw * math.cos(state.yaw - state.yawt) / ROD_LEN * DT

    return state_new


def get_nparray_from_matrix(x):
    """
    Convert a matrix to a flattened NumPy array.

    Args:
    x: Input matrix.

    Returns:
    flattened_array: Flattened NumPy array.
    """

    return np.array(x).flatten()


def calc_nearest_index(state, cx, cy, cyaw, pind):
    """
    Calculate the index and distance of the nearest point on a path.

    Args:
    state: Current vehicle state.
    cx: List of x-coordinates of path points.
    cy: List of y-coordinates of path points.
    cyaw: List of yaw angles of path points.
    pind: Current index to start searching from.

    Returns:
    ind: Index of the nearest point on the path.
    mind: Distance from the current state to the nearest point.
    """

    dx = [state.x - icx for icx in cx[pind:(pind + N_IND_SEARCH)]]
    dy = [state.y - icy for icy in cy[pind:(pind + N_IND_SEARCH)]]

    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

    mind = min(d)

    ind = d.index(mind) + pind

    mind = math.sqrt(mind)

    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y

    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1

    return ind, mind


def predict_motion(x0, ov, odyaw, xref):
    """
    Predict the vehicle's motion over a horizon using control inputs and a reference trajectory.

    Args:
    x0: Initial vehicle state (x, y, yaw, yawt).
    ov: List of velocity values over the prediction horizon.
    odyaw: List of yaw rate values over the prediction horizon.
    xref: Reference trajectory matrix.

    Returns:
    xbar: Predicted motion of the vehicle over the horizon (matrix).
    """

    xbar = xref * 0.0
    # print("x_bar size, ", xbar.shape, "x_ref size, ", xref.shape)
    for i, _ in enumerate(x0):
        xbar[i, 0] = x0[i]

    state = State(x=x0[0], y=x0[1], yaw=x0[2], yawt=x0[3])
    for (vi, dyawi, i) in zip(ov, odyaw, range(1, T + 1)):
        state = update_state(state, vi, dyawi)
        xbar[0, i] = state.x
        xbar[1, i] = state.y
        xbar[2, i] = state.yaw
        xbar[3, i] = state.yawt

    return xbar


def predict_motion_mpc(x, u):
    """
    Predict the vehicle's motion over a single time step using control inputs and the current state.

    Args:
    x: Current vehicle state (x, y, yaw, yawt).
    u: Control inputs (velocity, yaw rate).

    Returns:
    x1: Predicted vehicle state at the next time step.
    """

    x1 = x.copy()

    x1[0] = x[0] + u[0] * cos(x[2] - x[3]) * cos(x[3]) * DT
    x1[1] = x[1] + u[0] * cos(x[2] - x[3]) * sin(x[3]) * DT
    x1[2] = x[2] + u[1] * DT
    x1[3] = x[3] + u[0] / ROD_LEN * sin(x[2] - x[3]) * DT - CP_OFFSET * u[1] * cos(
        x[2] - x[3]) / ROD_LEN * DT

    return x1


def cal_line2circle(Ax, Ay, Bx, By, Cx, Cy, R2):
    """
    Calculate the squared distance between a line segment and a circle.

    Args:
    Ax, Ay: Coordinates of the first endpoint of the line segment.
    Bx, By: Coordinates of the second endpoint of the line segment.
    Cx, Cy: Coordinates of the center of the circle.
    R2: Square of the radius of the circle.

    Returns:
    res: Squared distance between the line segment and the circle.
    """

    # Calculate the length of the line segment (LAB)
    LAB = casadi.sqrt((Bx - Ax) ** 2 + (By - Ay) ** 2)

    # Calculate the direction vector of the line segment (Dx, Dy)
    Dx = (Bx - Ax) / LAB
    Dy = (By - Ay) / LAB

    # Calculate the projection of the center onto the line segment (Ex, Ey)
    t = Dx * (Cx - Ax) + Dy * (Cy - Ay)

    Ex = t * Dx + Ax
    Ey = t * Dy + Ay

    # Calculate the distance from the center to the projection (LEC)
    LEC = casadi.sqrt((Ex - Cx) ** 2 + (Ey - Cy) ** 2)

    test_var = (t - casadi.sqrt(R2 - LEC ** 2)) * Dx + Ax

    # dt =
    # dt = casadi.if_else(casadi.sqrt(R ** 2 - LEC ** 2)<-casadi.inf, 1,0)
    # Px1 = casadi.if_else(R < LEC, casadi.sqrt(LEC**2 - R**2), 0)
    # Px1 = LEC

    # Calculate the distance from the projection to the circle (dt)
    dt = casadi.if_else(LEC < casadi.sqrt(R2), casadi.sqrt(fabs(R2 - LEC ** 2)), 0)

    # return dt

    # Px1 = casadi.if_else(LEC < R, (t - casadi.sqrt(fabs(R ** 2 - LEC ** 2))) * Dx + Ax, 0)
    # Py1 = casadi.if_else(LEC < R, (t - casadi.sqrt(fabs(R ** 2 - LEC ** 2))) * Dy + Ay, 0)
    #
    # Px2 = casadi.if_else(LEC < R, (t + casadi.sqrt(fabs(R ** 2 - LEC ** 2))) * Dx + Ax, 0)
    # Py2 = casadi.if_else(LEC < R, (t + casadi.sqrt(fabs(R ** 2 - LEC ** 2))) * Dy + Ay, 0)

    # Calculate the coordinates of the two intersection points (Px1, Py1) and (Px2, Py2)
    Px1 = casadi.if_else(t > dt, (t - dt) * Dx + Ax, Ax)
    Py1 = casadi.if_else(t > dt, (t - dt) * Dy + Ay, Ay)

    Px2 = casadi.if_else(LAB - t < dt, Bx, (t + dt) * Dx + Ax)
    Py2 = casadi.if_else(LAB - t < dt, By, (t + dt) * Dy + Ay)

    # Calculate the squared distance between the line segment and the circle (res)
    # res = casadi.if_else(t < -dt, casadi.power(Px2 - Px1, 2) + casadi.power(Py2 - Py1, 2), 0)  # + 0.0000001
    res = casadi.power(Px2 - Px1, 2) + casadi.power(Py2 - Py1, 2)
    # res = Px2
    # res = casadi.sqrt(res)

    # return casadi.sqrt(casadi.power(Px2 - Px1, 2) + casadi.power(Py2 - Py1, 2))
    return res
    # return Px1


def test_line2circle(Ax, Ay, Bx, By, Cx, Cy, R):
    """
    Test the calculation of the squared distance between a line segment and a circle.

    Args:
    Ax, Ay: Coordinates of the first endpoint of the line segment.
    Bx, By: Coordinates of the second endpoint of the line segment.
    Cx, Cy: Coordinates of the center of the circle.
    R: Radius of the circle.

    Returns:
    res: Squared distance between the line segment and the circle.
    """

    LAB = sqrt((Bx - Ax) ** 2 + (By - Ay) ** 2)

    Dx = (Bx - Ax) / LAB
    Dy = (By - Ay) / LAB

    t = Dx * (Cx - Ax) + Dy * (Cy - Ay)

    Ex = t * Dx + Ax
    Ey = t * Dy + Ay

    LEC = sqrt((Ex - Cx) ** 2 + (Ey - Cy) ** 2)

    test_var = (t - casadi.sqrt(R ** 2 - LEC ** 2)) * Dx + Ax

    if LEC < R:
        dt = sqrt(R ** 2 - LEC ** 2)
        # print(LAB, t, dt)
        # if t < -dt:
        #     return 0
        if t > dt:
            Px1 = (t - dt) * Dx + Ax
            Py1 = (t - dt) * Dy + Ay
        else:
            Px1 = Ax
            Py1 = Ay
        if LAB - t < dt:
            Px2 = Bx
            Py2 = By
        else:
            Px2 = (t + dt) * Dx + Ax
            Py2 = (t + dt) * Dy + Ay
        res = (Px2 - Px1)** 2 + (Py2 - Py1) **2
        print("point 1", Px1, Py1, "point 2", Px2, Py2)
    else:
        dt = 0
        res = 0

    # return casadi.sqrt(casadi.power(Px2 - Px1, 2) + casadi.power(Py2 - Py1, 2))
    return res

def iterative_linear_mpc_control(xref, x0, ov, odyaw):
    """
    Perform iterative MPC control with updated operational points.

    Args:
    xref: Reference trajectory.
    x0: Initial state.
    ov: Initial control input (velocity).
    odyaw: Initial control input (angular velocity).

    Returns:
    ov: Updated control input (velocity).
    odyaw: Updated control input (angular velocity).
    ox: Predicted x positions.
    oy: Predicted y positions.
    oyaw: Predicted yaw angles.
    oyawt: Predicted yaw angular velocities.
    """
    # print("xref, ", xref, "x0, ", x0)
    if ov is None or odyaw is None:
        ov = [0.0] * T
        odyaw = [0.0] * T

    for i in range(MAX_ITER):
        xbar = predict_motion(x0, ov, odyaw, xref)
        pov, podyaw = ov[:], odyaw[:]
        ov, odyaw, ox, oy, oyaw, oyawt = linear_mpc_control(xref, xbar, x0)
        # print("ov_size: ", ov.shape)
        du = sum(abs(ov - pov)) + sum(abs(odyaw - podyaw))  # calc u change value
        # print("U change value: ", sum(abs(ov - pov)), sum(abs(odyaw - podyaw)))
        if du <= DU_TH:  # iteration finish param
            break
    else:
        print("Iterative is max iter")

    return ov, odyaw, ox, oy, oyaw, oyawt


# TODO: check control model
def linear_mpc_control(xref, xbar, x0):
    """
    linear mpc control

    Args:
    xref: reference point
    xbar: operational point
    x0: initial state

    Returns:
    ov: Calculated control input (velocity).
    odyaw: Calculated control input (angular velocity).
    ox: Calculated x positions.
    oy: Calculated y positions.
    oyaw: Calculated yaw angles.
    oyawt: Calculated yaw angular velocities.
    """

    global O_D

    opti = casadi.Opti();

    slack = opti.variable(3)

    x = opti.variable(NX, T + 1)
    u = opti.variable(NU, T)

    y = opti.variable(NX, 1000)

    p = opti.parameter(NX, T + 1)

    opti.set_value(p, xref)
    obj = 0
    # constraint = []

    stage_cost = 0

    for t in range(T):
        # st = x[:, t]
        # ref = xref[:, t]
        # con = u[:, t]

        obj += u[:, t].T @ R @ u[:, t]  # control(velocity) cost^

        if t < (T - 1):
            obj += (u[:, t + 1] - u[:, t]).T @ Rd @ (u[:, t + 1] - u[:, t])  # acceleration cost

        if t != 0:
            obj += (p[:, t] - x[:, t]).T @ Q @ (p[:, t] - x[:, t])  # stage cost
            stage_cost += (p[:, t] - x[:, t]).T @ Q @ (p[:, t] - x[:, t])

    obj += (p[:, T] - x[:, T]).T @ Qf @ (p[:, T] - x[:, T])  # terminal cost

    obj += 0.1 * slack.T@slack

    # v * cos(yaw), v * sin(yaw), v / WB * tan(delta), a
    A, B, C = get_linear_model_matrix(
        xref[:, t], xref[:, t + 1], xbar[:, t])
    #     constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]

    t_ref = 5

    # # sample based reachable set in cost function
    # U_min = np.array([MIN_SPEED, -MAX_OMEGA])
    # U_max = np.array([MAX_SPEED, MAX_OMEGA])
    #
    # M = 1000
    # ys = np.zeros((M, 4))
    # x_init = x[:, t_ref-1]
    # us = np.random.uniform(low=U_max, high=U_min, size=(M, 2))
    # for t in range(T-t_ref+1):
    #     if t == 0:
    #         for i in range(len(us)):
    #             ys[i, :] = predict_motion_mpc(x, us[i])
    #     else:
    #         for i in range(len(us)):
    #             ys[i, :] = predict_motion_mpc(ys[i], us[i])
    # print(ys.shape)

    # x1 = x[0, t_ref - 1]
    # y1 = x[1, t_ref - 1]
    # yaw1 = x[2, t_ref - 1]
    # yawt1 = x[3, t_ref - 1]
    #
    # x2 = x[0, t_ref - 1]
    # y2 = x[1, t_ref - 1]
    # yaw2 = x[2, t_ref - 1]
    # yawt2 = x[3, t_ref - 1]
    #
    # x4 = x[0, t_ref - 1]
    # y4 = x[1, t_ref - 1]
    # yaw4 = x[2, t_ref - 1]
    # yawt4 = x[3, t_ref - 1]

    # opti.subject_to(slack[1] == stage_cost)
    # opti.subject_to(slack[2] == (p[:, T] - x[:, T]).T @ Qf @ (p[:, T] - x[:, T]) )

    # Reachable set
    for t in range(T):
        # Nonlinear MPC
        opti.subject_to(x[0, t + 1] == x[0, t] + u[0, t] * cos(x[2, t] - x[3, t]) * cos(x[3, t]) * DT)
        opti.subject_to(x[1, t + 1] == x[1, t] + u[0, t] * cos(x[2, t] - x[3, t]) * sin(x[3, t]) * DT)
        opti.subject_to(x[2, t + 1] == x[2, t] + u[1, t] * DT)
        opti.subject_to(x[3, t + 1] == x[3, t] + u[0, t] / ROD_LEN * sin(x[2, t] - x[3, t]) * DT - CP_OFFSET * u[1, t] *
                        cos(x[2, t] - x[3, t]) / ROD_LEN * DT)
        # Linear MPC
        # opti.subject_to(x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C)

        # slack[:] = 0
        print("distance: ", np.rad2deg(atan2((x0[1] - O_Y[0]), (x0[0] - O_X[0]))), np.rad2deg(x0[3]),abs(np.rad2deg(x0[3])-np.rad2deg(atan2((x0[1] - O_Y[0]), (x0[0] - O_X[0])))), circle_robot_limit)

        if t < T - t_ref and abs(np.rad2deg(x0[3])-np.rad2deg(atan2((x0[1] - O_Y[0]), (x0[0] - O_X[0])))) < circle_robot_limit: #  and sqrt((x0[1] - O_Y[0])**2 + (x0[0] - O_X[0])**2) > 0.1 + O_R[0]:# and x0[0] > O_X[0]:  # sqrt((x0[1] - O_Y[0])**2 + (x0[0] - O_X[0])**2) < 2+ O_R[0]:  # and x0[1] - O_Y[0] < O_R[0]:
            print("t", t)

            # symbolic regression
            angle = - (x[2, t] - x[3, t])
            x1_ab = tan(
                (cos(sin(angle) + (cos(angle / -0.20138893) * -0.09728945))) ** 3 * -0.09880204)
            # ((math.cos(math.sin(x[i]) + (math.cos((0.5052427 / x[i]) + -0.3219844) * 0.13560449)))**3 * -0.10234546)
            y1_ab = (sin(sin(sin(angle + (angle - 0.029763347)))) * (
                    -0.21348037 / ((sin(angle / 0.51343226)) ** 3 + 3.8204532)))

            x2_ab = tan(tan(
                (cos(sin((angle - (
                        cos(angle * 1.1766043) * angle) ** 2) + 0.09091717))) ** 2) * -0.06572991)
            y2_ab = sin(sin(sin(sin(angle))) * -0.116112776)
            # ((math.sin((x[ind, 3] - (-0.013659489)**2) / 0.46065488) * -0.052274257) / math.cos(math.sin(x[ind, 3])))
            # (math.sin(x[ind, 3] / 0.6428589) * -0.07370442)

            x4_ab = ((cos(angle) + -0.3333654) * -0.16466537)
            y4_ab = (angle / ((-5.2116513 / cos(
                tan(sin(angle)))) - 3.598078))  # (math.sin(math.sin(math.sin(x[0]))) / -8.648622)

            rel_x1 = x1_ab * cos(x[2, t]) + y1_ab * -sin(
                x[2, t])  # @ [[cos(x[2, t]), -sin(x[2, t])],[sin(x[2, t]), cos(x[2, t])]]
            rel_y1 = x1_ab * sin(x[2, t]) + y1_ab * cos(x[2, t])
            x1 = x[0, t] + 2*rel_x1
            y1 = x[1, t] + 2*rel_y1

            rel_x2 = x2_ab * cos(x[2, t]) + y2_ab * -sin(x[2, t])
            rel_y2 = x2_ab * sin(x[2, t]) + y2_ab * cos(x[2, t])
            x2 = x[0, t] + 2*rel_x2
            y2 = x[1, t] + 2*rel_y2

            rel_x4 = x4_ab * cos(x[2, t]) + y4_ab * -sin(x[2, t])
            rel_y4 = x4_ab * sin(x[2, t]) + y4_ab * cos(x[2, t])
            x4 = x[0, t] + 2*rel_x4
            y4 = x[1, t] + 2*rel_y4

            rad_cbf = (1 - lam) * (
                    (x[0, t + 4] - O_X[0]) * (x[0, t + 4] - O_X[0]) + (x[1, t + 4] - O_Y[0]) * (x[1, t + 4] - O_Y[0]) -
                    O_R[0] ** 2)
            # rad_cbf = ((x[0, t + 5] - O_X[0]) * (x[0, t + 5] - O_X[0]) + (x[1, t + 5] - O_Y[0]) * (
            #         x[1, t + 5] - O_Y[0]) - O_R[0] ** 2)
            # rad_cbf = (1 - lam) * (
            #         (x[0, T] - O_X[0]) * (x[0, T] - O_X[0]) + (x[1, T] - O_Y[0]) * (x[1, T] - O_Y[0]) - O_R[0] ** 2)
            # obj += -0.5 * ((x1-x2) ** 2 + (y1-y2) ** 2)
            # obj += r_f * (cal_line2circle(x1, y1, x[0, t], x[1, t], O_X[0], O_Y[0], sqrt(rad_cbf + O_R[0] ** 2)) / sqrt(
            #     (x1 - x[0, t]) ** 2 + (y1 - x[1, t]) ** 2)) ** 2
            # obj += r_f * (cal_line2circle(x2, y2, x[0, t], x[1, t], O_X[0], O_Y[0], sqrt(rad_cbf + O_R[0] ** 2)) / sqrt(
            #     (x2 - x[0, t]) ** 2 + (y2 - x[1, t]) ** 2)) ** 2
            # opti.subject_to(cal_line2circle(x1, y1, x[0, t], x[1, t], O_X[0], O_Y[0], (rad_cbf + O_R[0] ** 2)) <= 0.02)
            # opti.subject_to(cal_line2circle(x2, y2, x[0, t], x[1, t], O_X[0], O_Y[0], (rad_cbf + O_R[0] ** 2)) <= 0.02)
            # opti.subject_to(cal_line2circle(x4, y4, x[0, t], x[1, t], O_X[0], O_Y[0], (rad_cbf + O_R[0] ** 2)) <= 0.02) # check what happened if else in casadi!!!!
            # opti.subject_to((x4-x[0,t])**2 + (y4-x[1,t])**2 <= 0.01)
            #cost = r_f * cal_line2circle(x1, y1, x[0, t], x[1, t], O_X[0], O_Y[0], (rad_cbf + O_R[0] ** 2))
            obj += r_f * cal_line2circle(x1, y1, x[0, t], x[1, t], O_X[0], O_Y[0], (rad_cbf + O_R[0] ** 2))
            obj += r_f * cal_line2circle(x2, y2, x[0, t], x[1, t], O_X[0], O_Y[0], (rad_cbf + O_R[0] ** 2))
            obj += r_f * cal_line2circle(x4, y4, x[0, t], x[1, t], O_X[0], O_Y[0], (rad_cbf + O_R[0] ** 2))
            # obj += r_f * cal_line2circle(x1, y1, x[0, t], x[1, t], O_X[0], O_Y[0], O_R[0]) ** 2
            # obj += r_f * cal_line2circle(x2, y2, x[0, t], x[1, t], O_X[0], O_Y[0], O_R[0]) ** 2
            # obj += r_f * cal_line2circle(x4, y4, x[0, t], x[1, t], O_X[0], O_Y[0], O_R[0]) ** 2
            # obj += r_f * cal_line2circle(x1, y1, x4, y4, O_X[0], O_Y[0], sqrt(rad_cbf + O_R[0] ** 2)) ** 2
            # obj += r_f * cal_line2circle(x2, y2, x4, y4, O_X[0], O_Y[0], sqrt(rad_cbf + O_R[0] ** 2)) ** 2

            # opti.subject_to(slack[0] == r_f * cal_line2circle(x4, y4, x[0, t], x[1, t], O_X[0], O_Y[0], (rad_cbf + O_R[0] ** 2)))


        # obj += slack[0] + slack[1] + slack[2]

            # obj += 1/((x[0, t] - O_X[0])**2 - O_R[0]**2)

        # jackknife constraint
        opti.subject_to(fabs(x[3, t + 1] - x[2, t + 1]) <= np.deg2rad(JACKKNIFE_CON))

        # control barrier function (CBF)
        # lam = 1  # 0-0.25   # 0.1-0.5 infeasible 0.05 ok 1, 0.8 (-5 0 1)

        for i in range(len(O_X)):
            # print(abs(x0[0] - O_X[i]), abs(x0[1] - O_Y[i]))
            # print("comparison!", (x0[0] - O_X[i]) ** 2 + (x0[1] - O_Y[i]) ** 2, O_D[i])

            if x0[0] - O_X[i] > 0:
                print("CBF activated!!!!!!!!!!!!!!!!!!")
                opti.subject_to(
                    (x[0, t + 1] - O_X[i]) * (x[0, t + 1] - O_X[i]) + (x[1, t + 1] - O_Y[i]) * (x[1, t + 1] - O_Y[i]) -
                    O_R[i] ** 2 >=
                    (1 - lam) * (
                            (x[0, t] - O_X[i]) * (x[0, t] - O_X[i]) + (x[1, t] - O_Y[i]) * (x[1, t] - O_Y[i]) - O_R[
                        i] ** 2))

            # opti.subject_to(
            #     (x[0, t + 1] + cos(x[3, t + 1]) * ROD_LEN + cos(x[2, t + 1]) * CP_OFFSET - O_X[i]) * (
            #                 x[0, t + 1] + cos(x[3, t + 1]) * ROD_LEN + cos(x[2, t + 1]) * CP_OFFSET - O_X[i]) + (
            #                 x[1, t + 1] + sin(x[3, t + 1] * ROD_LEN) + sin(x[2, t + 1]) * CP_OFFSET - O_Y[i]) * (
            #                 x[1, t + 1] + sin(x[3, t + 1] * ROD_LEN) + sin(x[2, t + 1]) * CP_OFFSET - O_Y[i]) - O_R[
            #         i] ** 2 >= (1 - lam)**(t+1) * (
            #                 (x[0, t] + cos(x[3, t]) * ROD_LEN + cos(x[2, t]) * CP_OFFSET - O_X[i]) * (
            #                     x[0, t] + cos(x[3, t]) * ROD_LEN + cos(x[2, t]) * CP_OFFSET - O_X[i]) + (
            #                             x[1, t] + sin(x[3, t] * ROD_LEN) + sin(x[2, t]) * CP_OFFSET - O_Y[i]) * (
            #                             x[1, t] + sin(x[3, t] * ROD_LEN) + sin(x[2, t]) * CP_OFFSET - O_Y[i]) - O_R[
            #                     i] ** 2))
            # opti.subject_to(
            #     fabs(x[0, t + 1] - O_X[i]) ** 3 + fabs(x[1, t + 1] - O_Y[i]) ** 3 - O_R[i] ** 3 >= (1 - lam) * (
            #                 fabs(x[0, t] - O_X[i]) ** 3 + fabs(x[1, t] - O_Y[i]) ** 3 - O_R[i] ** 3))
            # ** (t+1)

        # hard constraint
        # opti.subject_to((x[0, t + 1] - O_X[0]) * (x[0, t + 1] - O_X[0]) + (x[1, t + 1] - O_Y[0]) * (x[1, t + 1] - O_Y[0]) >= O_R[0]**2)
        # 0.5 * ((x[0, t] - 20) * (x[0, t] - 20) + (x[1, t + 1]) * (x[1, t + 1]) - 4)
        # (1 - lam) * ((x[0, t] - O_X)**2 + (x[1, t] - O_Y)**2 - O_R ** 2))

        # rotation acceleration constraint
        # if t < (T - 1):
        #     opti.subject_to(fabs(u[1, t + 1] - u[1, t]) <= MAX_DSTEER * DT)

    # if x0[1] - O_Y[i] > 0:
    # outside the last barrier function constraint
    # opti.subject_to((x1 - O_X[0]) * (x1 - O_X[0]) + (y1 - O_Y[0]) * (y1 - O_Y[0]) - O_R[0] ** 2 >= (1 - lam) * (
    #             (x[0, T - 2] - O_X[i]) * (x[0, T - 2] - O_X[i]) + (x[1, T - 2] - O_Y[i]) * (x[1, T - 2] - O_Y[i]) -
    #             O_R[i] ** 2))
    # opti.subject_to((x11 - O_X[0]) * (x11 - O_X[0]) + (y11 - O_Y[0]) * (y11 - O_Y[0]) - O_R[0] ** 2 >= (1 - lam) * (
    #         (x[0, T - 2] - O_X[i]) * (x[0, T - 2] - O_X[i]) + (x[1, T - 2] - O_Y[i]) * (x[1, T - 2] - O_Y[i]) -
    #         O_R[i] ** 2))

    # outside the hard constraints
    # opti.subject_to((x1 - O_X[0]) * (x1 - O_X[0]) + (y1 - O_Y[0]) * (y1 - O_Y[0]) - O_R[0] ** 2 >= 0)
    # opti.subject_to((x11 - O_X[0]) * (x11 - O_X[0]) + (y11 - O_Y[0]) * (y11 - O_Y[0]) - O_R[0] ** 2 >= 0)

    # initial pose constraint
    opti.subject_to(x[:, 0] == x0)

    # control input constraint
    opti.subject_to(fabs(u[0, :]) <= MAX_SPEED)
    opti.subject_to(fabs(u[1, :]) <= MAX_OMEGA)

    opti.minimize(obj)

    p_opts = dict(print_time=False, verbose=False)
    s_opts = dict(print_level=1)
    opti.solver("ipopt", p_opts, s_opts)
    print(opti.debug.value)
    # opti.solver("ipopt")  # set numerical backend
    st = time.time()
    sol = opti.solve()

    # p_opts = dict(print_time=False, verbose=False)
    # s_opts = dict(print_level=0, linear_solver='ma57')
    # opti.solver("ipopt", p_opts, s_opts)
    # # opti.solver("ipopt")  # set numerical backend
    # st = time.time()
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", opti.debug.value)
    # sol = opti.solve()

    elapsed_time = (time.time() - st) * 1000
    print('Execution time:', elapsed_time, 'milliseconds')
    T_CAL.append(elapsed_time)

    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", opti.debug, sol.value, sol.value(x), sol.value(u))
    # print(obj)
    ox = get_nparray_from_matrix(sol.value(x)[0, :])
    oy = get_nparray_from_matrix(sol.value(x)[1, :])
    oyaw = get_nparray_from_matrix(sol.value(x)[2, :])
    oyawt = get_nparray_from_matrix(sol.value(x)[3, :])
    ov = get_nparray_from_matrix(sol.value(u)[0, :])
    odyaw = get_nparray_from_matrix(sol.value(u)[1, :])
    oslack = get_nparray_from_matrix(sol.value(slack))
    print("slack!!!!!!!!", oslack[0], oslack[1], oslack[2])
    BARRIER_LIST.clear()
    for j in range(len(O_X)):
        O_D[j] = sqrt((x0[0] - O_X[j]) ** 2 + (x0[1] - O_Y[j]) ** 2)
        MIN_DIST.append(O_D[j]) # should be rewrite for multiple obstacle
    t_ref=6
    print("t_ref", t_ref-1)
    x1 = ox[t_ref - 1]
    y1 = oy[t_ref - 1]
    yaw1 = oyaw[t_ref - 1]
    yawt1 = oyawt[t_ref - 1]
    x11 = ox[t_ref - 1]
    y11 = oy[t_ref - 1]
    yaw11 = oyaw[t_ref - 1]
    yawt11 = oyawt[t_ref - 1]
    x12 = ox[t_ref - 1]
    y12 = oy[t_ref - 1]
    yaw12 = oyaw[t_ref - 1]
    yawt12 = oyawt[t_ref - 1]
    # for t1 in range(T - t_ref):
    oangle = -(oyaw[t_ref - 1] - oyawt[t_ref - 1])
    print("oangle", np.rad2deg(oyaw[t_ref - 1]), np.rad2deg(oyawt[t_ref - 1]), oangle, ",", np.rad2deg(oangle))
    x1_ab = tan(
        (cos(sin(oangle) + (cos(oangle / -0.20138893) * -0.09728945))) ** 3 * -0.09880204)
    # ((math.cos(math.sin(x[i]) + (math.cos((0.5052427 / x[i]) + -0.3219844) * 0.13560449)))**3 * -0.10234546)
    y1_ab = (sin(sin(sin(oangle + (oangle - 0.029763347)))) * (
            -0.21348037 / ((sin(oangle / 0.51343226)) ** 3 + 3.8204532)))

    x2_ab = tan(tan(
        (cos(sin((oangle - (
                cos(oangle * 1.1766043) * oangle) ** 2) + 0.09091717))) ** 2) * -0.06572991)
    y2_ab = sin(sin(sin(sin(oangle))) * -0.116112776)
    # ((math.sin((x[ind, 3] - (-0.013659489)**2) / 0.46065488) * -0.052274257) / math.cos(math.sin(x[ind, 3])))
    # (math.sin(x[ind, 3] / 0.6428589) * -0.07370442)

    x4_ab = ((cos(oangle) + -0.3333654) * -0.16466537)
    y4_ab = (oangle / ((-5.2116513 / cos(
        tan(sin(oangle)))) - 3.598078))  # (math.sin(math.sin(math.sin(x[0]))) / -8.648622)

    rel_x1 = x1_ab * cos(oyaw[t_ref - 1]) + y1_ab * -sin(
        oyaw[t_ref - 1])  # @ [[cos(x[2, t]), -sin(x[2, t])],[sin(x[2, t]), cos(x[2, t])]]
    rel_y1 = x1_ab * sin(oyaw[t_ref - 1]) + y1_ab * cos(oyaw[t_ref - 1])
    x1 = ox[t_ref - 1] + 2*rel_x1
    y1 = oy[t_ref - 1] + 2*rel_y1

    rel_x2 = x2_ab * cos(oyaw[t_ref - 1]) + y2_ab * -sin(oyaw[t_ref - 1])
    rel_y2 = x2_ab * sin(oyaw[t_ref - 1]) + y2_ab * cos(oyaw[t_ref - 1])
    x2 = ox[t_ref - 1] + 2*rel_x2
    y2 = oy[t_ref - 1] + 2*rel_y2

    rel_x4 = x4_ab * cos(oyaw[t_ref - 1]) + y4_ab * -sin(oyaw[t_ref - 1])
    rel_y4 = x4_ab * sin(oyaw[t_ref - 1]) + y4_ab * cos(oyaw[t_ref - 1])
    x4 = ox[t_ref - 1] + 2*rel_x4
    y4 = oy[t_ref - 1] + 2*rel_y4

    rad_cbf = ((ox[t_ref + 1] - O_X[0]) * (ox[t_ref + 1] - O_X[0]) + (oy[t_ref + 1] - O_Y[0]) * (
            oy[t_ref + 1] - O_Y[0]) - O_R[0] ** 2)

    BARRIER_LIST.append(sqrt(rad_cbf + O_R[0] ** 2))

    print("reachable cost!!!!!!!!!!", rad_cbf + O_R[0] ** 2, "x1: (", x1, y1, ")" , test_line2circle(x1, y1, ox[t_ref - 1], oy[t_ref - 1], O_X[0], O_Y[0], sqrt(rad_cbf + O_R[0] ** 2)),
          "x2: (", x2, y2, ")", test_line2circle(x2, y2, ox[t_ref - 1], oy[t_ref - 1], O_X[0], O_Y[0], sqrt(rad_cbf + O_R[0] ** 2)),
          "x4: (", x4, y4, ")", test_line2circle(x4, y4, ox[t_ref - 1], oy[t_ref - 1], O_X[0], O_Y[0], sqrt(rad_cbf + O_R[0] ** 2)),
          "Distance:", sqrt((x2-ox[t_ref - 1])**2+(y2-oy[t_ref - 1])**2), sqrt((x1-ox[t_ref - 1])**2+(y1-oy[t_ref - 1])**2), sqrt((ox[t_ref - 1]-x4)**2+(oy[t_ref - 1]-y4)**2),
          test_line2circle(x2, y2, x4, y4, O_X[0], O_Y[0], sqrt(rad_cbf + O_R[0] ** 2)),
          test_line2circle(x1, y1, x4, y4, O_X[0], O_Y[0], sqrt(rad_cbf + O_R[0] ** 2)),
          "Distance:", sqrt((x2-x4)**2+(y2-y4)**2), sqrt((x1-x4)**2+(y1-y4)**2))

    X_SET.clear()
    Y_SET.clear()

    X_SET.append(ox[t_ref - 1])
    X_SET.append(x1)
    X_SET.append(x2)
    X_SET.append(x4)

    Y_SET.append(oy[t_ref - 1])
    Y_SET.append(y1)
    Y_SET.append(y2)
    Y_SET.append(y4)

    return ov, odyaw, ox, oy, oyaw, oyawt


def calc_ref_trajectory(state, cx, cy, cyaw, cyawt, ck, sp, dl, pind):
    """
    Calculate a reference trajectory.

    Args:
    state: Current state information
    cx: List of reference path points' x-coordinates.
    cy: List of reference path points' y-coordinates.
    cyaw: List of reference path's yaw angles.
    cyawt: List of trailer's yaw angles.
    pind (int): Index of the reference point on the path.

    Returns:
    xref: Reference trajectory array with shape (NX, T + 1), where NX is the number of states, and T is the prediction time steps.
    ind: Index of the nearest path point.
    """

    xref = np.zeros((NX, T + 1))
    dref = np.zeros((1, T + 1))
    ncourse = len(cx)

    ind, _ = calc_nearest_index(state, cx, cy, cyawt, pind)

    if pind >= ind:
        ind = pind

    xref[0, 0] = cx[ind]
    xref[1, 0] = cy[ind]
    # xref[2, 0] = sp[ind]
    # TODO: if the pose of tractor could be calculate in this case, when yes, define the function cal_tractor_pose
    xref[2, 0] = cyaw[ind]  # cal_tractor_pose(cyawt[ind])
    xref[3, 0] = cyawt[ind]

    for i in range(T + 1):
        # TODO: what will happen when v is not state variable and should be negative?
        if (ind + i) < ncourse:
            xref[0, i] = cx[ind + i]
            xref[1, i] = cy[ind + i]
            xref[2, i] = cyaw[ind + i]  # cal_tractor_pose(cyawt[ind + dind])
            xref[3, i] = cyawt[ind + i]
        else:
            xref[0, i] = cx[ncourse - 1]
            xref[1, i] = cy[ncourse - 1]
            # xref[2, i] = sp[ncourse - 1]
            # TODO: traget yaw position should be pre-defined!!!
            xref[2, i] = cyaw[ncourse - 1]  # cal_tractor_pose(cyawt[ncourse - 1])
            xref[3, i] = cyawt[ncourse - 1]

    return xref, ind


def check_goal(state, goal, tind, nind):
    # check goal
    dx = state.x - goal[0]
    dy = state.y - goal[1]
    d = math.hypot(dx, dy)

    print(d, GOAL_DIS)

    isgoal = (d <= GOAL_DIS)

    if abs(tind - nind) >= 10:
        isgoal = False

    if isgoal:
        return True

    return False


def do_simulation(cx, cy, cyaw, cyawt, ck, sp, dl, initial_state):
    """
    Simulation

    cx: course x position list
    cy: course y position list
    cyawt: course yaw position of the trailer list
    ck: course curvature list
    sp: speed profile
    dl: course tick [m]

    """

    goal = [cx[-1], cy[-1]]

    state = initial_state

    # initial yaw compensation
    if state.yawt - cyawt[0] >= math.pi:
        state.yawt -= math.pi * 2.0
    elif state.yawt - cyawt[0] <= -math.pi:
        state.yawt += math.pi * 2.0

    Time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    yawt = [state.yawt]
    t = [0.0]
    dyaw = [0.0]
    v = [0.0]
    diff_tar_act = []
    target_ind, _ = calc_nearest_index(state, cx, cy, cyawt, 0)

    txt_list = [state.x + math.cos(state.yawt) * ROD_LEN]
    tyt_list = [state.y + math.sin(state.yawt) * ROD_LEN]

    O_DIST = 100000000000000

    ov, odyaw = None, None

    # TODO: check the function smooth_yaw
    cyawt = smooth_yaw(cyawt)

    start = time.time()

    while MAX_TIME >= Time:
        xref, target_ind = calc_ref_trajectory(state, cx, cy, cyaw, cyawt, ck, sp, dl, target_ind)
        # print(target_ind)
        # target_ind += 1

        x0 = [state.x, state.y, state.yaw, state.yawt]  # current state

        diff_tar_act.append([abs(xref[0, 0] - state.x), abs(xref[1, 0] - state.y)])

        ov, odyaw, ox, oy, oyaw, oyawt = iterative_linear_mpc_control(
            xref, x0, ov, odyaw)

        print("xO", x0, "x ref: ", xref[0, :], "state: ", ox, "y ref: ", xref[1, :], "state: ", oy, oyaw, oyawt,
              "input: ", ov, odyaw)
        if abs(ov[0]) <= 0.0001:
            print("WARNING!!!!!!", ov[0])

        if odyaw is not None:
            dyawi, vi = odyaw[0], ov[0]

        state = update_state(state, vi, dyawi)

        # visualization reachable set
        PX_SET.clear()
        PY_SET.clear()
        next_state1 = state
        next_state2 = state
        next_state3 = state
        for i in range(T):
            next_state1 = update_state(next_state1, MIN_SPEED, MAX_OMEGA)
            next_state2 = update_state(next_state2, MIN_SPEED, -MAX_OMEGA)
            next_state3 = update_state(next_state3, MIN_SPEED, 0)
            PX_SET.append(next_state1.x)
            PX_SET.append(next_state2.x)
            PX_SET.append(next_state3.x)
            PY_SET.append(next_state1.y)
            PY_SET.append(next_state2.y)
            PY_SET.append(next_state3.y)
        # print("reachibility: ", PX_SET, PY_SET)
        Time = Time + DT

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        yawt.append(state.yawt)
        t.append(Time)
        dyaw.append(dyawi)
        v.append(vi)

        if Time == 0.2:
            phi_tractrix = 1
            phi_model = 0
        # dyaw = yaw[-1] - yaw[-2]
        # TODO: Tractrix curve for kinematic model?

        if check_goal(state, goal, target_ind, len(cx)):
            print("Goal")
            end = time.time()
            print("total time used", end - start)
            break

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event: [exit(0) if event.key == 'escape' else None])
            if ox is not None:
                plt.plot(ox, oy, "xr", label="MPC")
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(x, y, "ob", label="trajectory")
            plt.plot(xref[0, :], xref[1, :], "xk", label="xref")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            # TODO: the whole tractor-trailer system will be plot in one function
            plot_car(state.x, state.y, state.yawt, LENGTH_T)

            # for tractor
            plot_car(state.x + np.cos(state.yawt) * ROD_LEN + np.cos(state.yaw) * CP_OFFSET,
                     state.y + np.sin(state.yawt) * ROD_LEN + np.sin(state.yaw) * CP_OFFSET, state.yaw, LENGTH)
            plt.plot([state.x, state.x + np.cos(state.yawt) * ROD_LEN],
                     [state.y, state.y + np.sin(state.yawt) * ROD_LEN],
                     color='black', linewidth=2, linestyle='-')
            plt.plot([state.x + np.cos(state.yawt) * ROD_LEN,
                      state.x + np.cos(state.yawt) * ROD_LEN + np.cos(state.yaw) * CP_OFFSET],
                     [state.y + np.sin(state.yawt) * ROD_LEN,
                      state.y + np.sin(state.yawt) * ROD_LEN + np.sin(state.yaw) * CP_OFFSET], color='black',
                     linewidth=2, linestyle='--')
            txt_list.append(state.x + np.cos(state.yawt) * ROD_LEN)
            tyt_list.append(state.y + np.sin(state.yawt) * ROD_LEN)

            # reachable set
            plt.plot(X_SET, Y_SET, "or", label='reachability')
            # plt.plot(PX_SET, PY_SET, "or", label='reachability')
            for i in range(len(BARRIER_LIST)):
                theta = np.linspace(0, 2 * np.pi, 100)
                circle_x = O_X[0] + (BARRIER_LIST[i]) * np.cos(theta)
                circle_y = O_Y[0] + (BARRIER_LIST[i]) * np.sin(theta)
                # plt.plot(circle_x, circle_y)

            for i in range(len(O_X)):
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", (state.x - O_X[i]) ** 2 + (state.y - O_Y[i]) ** 2 - O_R[i] ** 2)

                theta = np.linspace(0, 2 * np.pi, 100)
                circle_x = O_X[i] + O_R[i] * np.cos(theta)
                circle_y = O_Y[i] + O_R[i] * np.sin(theta)
                plt.plot(circle_x, circle_y)

            plt.axis("equal")
            plt.grid(True)

            plt.pause(0.0001)
    return t, x, y, yaw, yawt, dyaw, v, txt_list, tyt_list, diff_tar_act

def calc_speed_profile(cx, cy, cyaw, target_speed):
    # Clculate the moving direction of each time step

    speed_profile = [target_speed] * len(cx)
    direction = 1.0  # forward

    # Set stop point
    for i in range(len(cx) - 1):
        dx = cx[i + 1] - cx[i]
        dy = cy[i + 1] - cy[i]

        move_direction = math.atan2(dy, dx)
        # print(cyaw[i], move_direction, dy, dx)

        if dx != 0.0 and dy != 0.0:
            dangle = abs(pi_2_pi(move_direction - cyaw[i]))
            if dangle >= math.pi / 4.0:
                direction = -1.0
            else:
                direction = 1.0

        if direction != 1.0:
            speed_profile[i] = - target_speed
        else:
            speed_profile[i] = target_speed
        # speed_profile[i] = -target_speed

    speed_profile[-1] = 0.0

    return speed_profile


def smooth_yaw(yaw):
    for i in range(len(yaw) - 1):
        dyaw = yaw[i + 1] - yaw[i]

        while dyaw >= math.pi / 2.0:
            yaw[i + 1] -= math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

        while dyaw <= -math.pi / 2.0:
            yaw[i + 1] += math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

    return yaw


def get_straight_course(dl):
    ax = [0.0, -2.0, -4.0, -6.0, -8.0]
    ay = [0.0, 0.0, 0.0, 0.0, 0.0]
    cx, cy, cyaw, ck, s = PathPlanner.calc_spline_course(
        ax, ay, ds=dl)

    return cx, cy, cyaw, ck


def get_straight_course2(dl):
    ax = [0.0, -2.0, -4.0, -8.0, -10.0, -12.0, -14.0]
    ay = [0.0, -0.2, 0.2, 0.0, -0.2, 0.2, 0.0]
    cx, cy, cyaw, ck, s = PathPlanner.calc_spline_course(
        ax, ay, ds=dl)

    return cx, cy, cyaw, ck


def get_straight_course3(dl):
    ax = [0.0, -10.0, -20.0, -40.0, -50.0, -60.0, -70.0]
    ay = [0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0]
    cx, cy, cyaw, ck, s = PathPlanner.calc_spline_course(
        ax, ay, ds=dl)

    cyaw = [pi_2_pi(i - math.pi) for i in cyaw]

    return cx, cy, cyaw, ck


def get_forward_course(dl):
    ax = [0.0, 60.0, 125.0, 50.0, 75.0, 30.0, -10.0]
    ay = [0.0, 0.0, 50.0, 65.0, 30.0, 50.0, -20.0]
    cx, cy, cyaw, ck, s = PathPlanner.calc_spline_course(
        ax, ay, ds=dl)

    return cx, cy, cyaw, ck


def get_switch_back_course(dl):
    ax = [0.0, 30.0, 6.0, 20.0, 35.0]
    ay = [0.0, 0.0, 20.0, 35.0, 20.0]
    cx, cy, cyaw, ck, s = PathPlanner.calc_spline_course(
        ax, ay, ds=dl)
    ax = [35.0, 10.0, 0.0, 0.0]
    ay = [20.0, 30.0, 5.0, 0.0]
    cx2, cy2, cyaw2, ck2, s2 = PathPlanner.calc_spline_course(
        ax, ay, ds=dl)
    cyaw2 = [i - math.pi for i in cyaw2]
    cx.extend(cx2)
    cy.extend(cy2)
    cyaw.extend(cyaw2)
    ck.extend(ck2)

    return cx, cy, cyaw, ck


def get_reverse_parking_course(dl):
    # ax = [0.0, -5.0, -10.0, -20.0, -30.0, -30.0, -30.0]
    # ay = [0.0, 0.0, 0.0, 0.0, -10.0, -20.0, -30.0]
    ax = [0.0, -1.0, -2.0, -4.0, -6.0, -6.0, -6.0]
    ay = [0.0, 0.0, 0.0, 0.0, -2.0, -4.0, -6.0]
    cx, cy, cyaw, ck, s = PathPlanner.calc_spline_course(
        ax, ay, ds=dl)

    cyaw = [pi_2_pi(i - math.pi) for i in cyaw]

    return cx, cy, cyaw, ck


def get_circle_course_forward(r):
    t = np.linspace(0, 2 * math.pi, num=200)
    ax = [r * math.sin(i) for i in t]
    ay = [r * math.cos(i) for i in t]
    ck = np.zeros(200)

    return ax, ay, -t, ck


def get_circle_course_backward(r):
    t = np.linspace(0, 0.5 * math.pi, num=50)
    ax = [- r * math.sin(i) for i in t]
    ay = [r * math.cos(i) for i in t]
    ck = np.zeros(200)

    return ax, ay, t, ck


def main():
    print(__file__ + " start!!")

    dl = 0.05  # course tick
    # cx, cy, cyawt, ck = get_straight_course(dl)
    # cyawt = np.zeros(len(cyawt))
    # cx, cy, cyawt, ck = get_straight_course2(dl)
    # cyawt = [pi_2_pi(i-math.pi) for i in cyawt]
    # cx, cy, cyawt, ck = get_straight_course3(dl)
    # cx, cy, cyawt, ck = get_forward_course(dl)
    # print(cyawt)
    # cyawt = [abs(i) for i in cyawt]
    # cx, cy, cyawt, ck = get_switch_back_course(dl)
    # cx, cy, cyawt, ck = get_circle_course_forward(CIR_RAD)
    # cx, cy, cyawt, ck = get_circle_course_backward(CIR_RAD)
    # cx, cy, cyawt, ck = get_reverse_parking_course(dl)
    cx, cy, cyawt, ck = line_cross_2_obstacle()
    # print(cyawt)

    sp = calc_speed_profile(cx, cy, cyawt, TARGET_SPEED)
    # sp = [i*-1 for i in sp]
    # TODO: get cyaw from high-level planning part
    cyaw = np.copy(cyawt)
    # print(sp)

    initial_state = State(x=cx[0], y=cy[0], yaw=cyaw[0], yawt=cyawt[0])
    # cyaw[0], cyawt[0]

    t, x, y, yaw, yawt, dyawt, v, txt, tyt, diff = do_simulation(
        cx, cy, cyaw, cyawt, ck, sp, dl, initial_state)
    # print("control input:", dyawt, v)

    diff = np.array(diff)
    # print("error: ", sum([math.sqrt(i[0] ** 2 + i[1] ** 2) for i in diff]))

    res = [x, y]

    print("min distance to obstacle: ", min(MIN_DIST))

    print("save data into csv!", res)
    pd.DataFrame(res).to_csv("res_cbf_02_20.csv")

    if show_animation:  # pragma: no cover
        print("max iteration time", max(T_CAL))

        plt.close("all")
        plt.subplots()
        plt.plot(cx, cy, "-r", label="spline")
        plt.plot(x, y, "-g", label="trailer")
        plt.plot(txt, tyt, "-b", label="tractor")
        theta = np.linspace(0, 2 * np.pi, 100)
        circle_x = O_X + O_R * np.cos(theta)
        circle_y = O_Y + O_R * np.sin(theta)
        plt.plot(circle_x, circle_y)
        # plt.plot(txm, tym, "-y", label="tracking")
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend()

        plt.subplots()
        plt.plot(t, v, "-r", label="speed")
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel("Speed [kmh]")

        plt.subplots()
        t = np.linspace(0, diff.shape[0], num=diff.shape[0])
        plt.plot(t, diff.T[0], "-g", label="x_axis")
        plt.plot(t, diff.T[1], "-b", label="y_axis")
        plt.plot(t, [math.sqrt(i[0] ** 2 + i[1] ** 2) for i in diff], "-r", label="distance")
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel("difference [m]")

        plt.show()


def main2():
    print(__file__ + " start!!")

    dl = 1.0  # course tick
    cx, cy, cyaw, ck = get_straight_course3(dl)

    sp = calc_speed_profile(cx, cy, cyaw, TARGET_SPEED)

    initial_state = State(x=cx[0], y=cy[0], yaw=0.0, yawt=0.0)
    print(initial_state.yaw, initial_state.yawt)

    t, x, y, yaw, v, d, a = do_simulation(cx, cy, cyaw, ck, sp, dl, initial_state)

    if show_animation:  # pragma: no cover
        plt.close("all")
        plt.subplots()
        plt.plot(cx, cy, "-r", label="spline")
        plt.plot(x, y, "-g", label="tracking")
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend()

        plt.subplots()
        plt.plot(t, v, "-r", label="speed")
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel("Speed [kmh]")

        plt.show()


def main3():
    x1 = 1
    y1 = 1
    x2 = -3
    y2 = -3
    print(test_line2circle(x1, y1, x2, y2, 0, 0, 3))
    plt.subplots()
    plt.scatter(x1, y1)
    plt.scatter(x2, y2)
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = 0 + 3 * np.cos(theta)
    circle_y = 0 + 3 * np.sin(theta)
    plt.plot(circle_x, circle_y)
    plt.show()


if __name__ == '__main__':
    main()
    # main2()
    # main3()
