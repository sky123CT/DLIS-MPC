import numpy as np
import math
from casadi import *


def casadi_distance(point1_sym, point2_sym):
    if point1_sym.shape[0] != point2_sym.shape[0]:
        raise ValueError("two points must have the same dimension!")
    else:
        square_distance = 0
        for i in range(point1_sym.shape[0]):
            square_distance += (point1_sym[i] - point2_sym[i]) ** 2
        return sqrt(square_distance)

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


def calc_nearest_index(state, cx, cy, cyaw, pind, N_IND_SEARCH):
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


def calc_ref_trajectory(state, cx, cy, cyaw, cyawt, pind, NX, T, N_IND_SEARCH):
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
    ncourse = len(cx)

    ind, _ = calc_nearest_index(state, cx, cy, cyawt, pind, N_IND_SEARCH)

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


def line_cross_2_obstacle():
    t = np.linspace(0, 50, num=50)
    ax = [-0.2*i for i in t]
    ay = [0*i for i in t]
    yaw = np.zeros(50)
    ck = np.zeros(200)

    return ax, ay, yaw, ck


def get_circle_course_backward(r):
  t = np.linspace(0, 0.5 * math.pi, num=50)
  ax = [- r * math.sin(i) for i in t]
  ay = [r * math.cos(i) for i in t]
  ck = np.zeros(200)

  return ax, ay, t, ck


def pi_2_pi(angle):
    while angle > math.pi:
        angle = angle - 2.0 * math.pi

    while angle < -math.pi:
        angle = angle + 2.0 * math.pi

    return angle


def calc_speed_profile(cx, cy, cyaw, target_speed):
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
