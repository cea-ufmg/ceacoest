"""Symbolic quaterion operations, dynamics and rotations."""


import numpy as np
import sympy


def derivative(quat, omega, renorm_gain=0):
    [q0, q1, q2, q3] = quat
    [p, q, r] = omega
    err = 1 - (q0**2 + q1**2 + q2**2 + q3**2)
    q0dot = -0.5 * (p*q1 + q*q2 + r*q3) + renorm_gain*err*q0
    q1dot = -0.5 * (-p*q0 - r*q2 + q*q3) + renorm_gain*err*q1
    q2dot = -0.5 * (-q*q0 + r*q1 - p*q3) + renorm_gain*err*q2
    q3dot = -0.5 * (-r*q0 - q*q1 + p*q2) + renorm_gain*err*q3
    return np.array([q0dot, q1dot, q2dot, q3dot])


def rotmat(quat):
    """Quaternion rotation matrix."""
    q0, q1, q2, q3 = quat
    return np.array(
        [[q0**2 + q1**2 - q2**2 - q3**2, 2*(q1*q2 + q0*q3), 2*(q1*q3 - q0*q2)],
         [2*(q1*q2 - q0*q3), q0**2 - q1**2 + q2**2 - q3**2, 2*(q2*q3 + q0*q1)],
         [2*(q1*q3 + q0*q2), 2*(q2*q3 - q0*q1), q0**2 - q1**2 - q2**2 + q3**2]]
    )


def toeuler(quat):
    """Convert quaternion rotation to roll-pitch-yaw Euler angles."""
    q0, q1, q2, q3 = quat
    roll = sympy.atan2(2*(q2*q3 + q0*q1), q0**2 - q1**2 - q2**2 + q3**2)
    pitch = -sympy.asin(2*(q1*q3 - q0*q2))
    yaw = sympy.atan2(2*(q1*q2 + q0*q3), q0**2 + q1**2 - q2**2 - q3**2)
    return np.array([roll, pitch, yaw])

