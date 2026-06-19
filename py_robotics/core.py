"""Robotics kinematics and dynamics utilities translated from MATLAB to NumPy.

Conventions
-----------
* Twists are ordered as [omega_x, omega_y, omega_z, v_x, v_y, v_z].
* Most functions accept 1-D vectors and 2-D/3-D arrays. Homogeneous transform
  stacks may be either (4,4,n) as in MATLAB or (n,4,4); a single transform is (4,4).
* Robot data can be a Robot dataclass, dict, or any object with matching attributes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Tuple
import json
import numpy as np

_EPS = np.finfo(float).eps


@dataclass
class Robot:
    dof: int
    A: np.ndarray
    M: np.ndarray
    ME: np.ndarray = field(default_factory=lambda: np.eye(4))
    TCP: np.ndarray = field(default_factory=lambda: np.eye(4))
    mass: np.ndarray | None = None
    inertia: np.ndarray | None = None
    com: np.ndarray | None = None
    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -9.8]))


def _arr(x, dtype=float):
    return np.asarray(x, dtype=dtype)


def _vec(x) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def _row6(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if v.ndim == 1:
        return v.reshape(1, -1)
    if v.ndim == 2 and v.shape[1] == 1:
        return v.reshape(1, -1)
    return v


def _get(robot: Any, name: str):
    if isinstance(robot, dict):
        return robot[name]
    return getattr(robot, name)


def _set(robot: Any, name: str, value):
    if isinstance(robot, dict):
        robot[name] = value
    else:
        setattr(robot, name, value)


def _dof(robot: Any) -> int:
    return int(_get(robot, 'dof'))


def _A(robot: Any) -> np.ndarray:
    return _arr(_get(robot, 'A'))


def _M_stack(robot: Any) -> np.ndarray:
    return _as_stack(_arr(_get(robot, 'M')), 4, 4)


def _inertia_stack(robot: Any) -> np.ndarray:
    return _as_stack(_arr(_get(robot, 'inertia')), 3, 3)


def _ME(robot: Any) -> np.ndarray:
    ME = _arr(_get(robot, 'ME'))
    TCP = _arr(_get(robot, 'TCP')) if hasattr(robot, 'TCP') or (isinstance(robot, dict) and 'TCP' in robot) else np.eye(4)
    return ME @ TCP


def _gravity(robot: Any) -> np.ndarray:
    return _vec(_get(robot, 'gravity'))


def _as_stack(X: np.ndarray, r: int, c: int) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 2:
        if X.shape != (r, c):
            raise ValueError(f'Expected shape {(r, c)}, got {X.shape}')
        return X.reshape(1, r, c)
    if X.ndim == 3:
        if X.shape[:2] == (r, c):      # MATLAB layout: r x c x n
            return np.moveaxis(X, 2, 0).copy()
        if X.shape[1:] == (r, c):      # Python layout: n x r x c
            return X.copy()
    raise ValueError(f'Expected transform stack with ({r},{c},n) or (n,{r},{c}), got {X.shape}')


def _maybe_single(stack: np.ndarray):
    return stack[0] if stack.shape[0] == 1 else stack


def _block2(A, B, C, D):
    return np.block([[A, B], [C, D]])


def _solve(A, B):
    return np.linalg.solve(np.asarray(A, dtype=float), np.asarray(B, dtype=float))


def so_w(w):
    """Skew matrix [w] from a 3-vector."""
    w = _vec(w)
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def vec2skew_mat(x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1 or (x.ndim == 2 and 1 in x.shape):
        return so_w(x)
    return np.stack([so_w(row) for row in x], axis=0)


def skew_mat2vec(X):
    Xs = _as_stack(np.asarray(X, dtype=float), 3, 3)
    out = np.zeros((Xs.shape[0], 3))
    for i, X in enumerate(Xs):
        out[i] = [-X[1, 2], X[0, 2], -X[0, 1]]
    return out[0] if len(out) == 1 else out


def exp_w(w):
    wv = _row6(w)
    if wv.shape[1] != 3:
        wv = np.asarray(w, dtype=float).reshape(-1, 3)
    Rs = []
    for wi in wv:
        theta = np.linalg.norm(wi)
        if theta <= _EPS:
            Rs.append(np.eye(3))
        else:
            wh = wi / theta
            W = so_w(wh)
            Rs.append(np.eye(3) + np.sin(theta) * W + (1.0 - np.cos(theta)) * (W @ W))
    R = np.stack(Rs, axis=0)
    return R[0] if len(R) == 1 else R


def logR(R):
    Rs = _as_stack(np.asarray(R, dtype=float), 3, 3)
    out = np.zeros((Rs.shape[0], 3))
    I = np.eye(3)
    for i, Ri in enumerate(Rs):
        _, _, Vt = np.linalg.svd(Ri - I)
        v = Vt.T[:, -1]
        v_hat = np.array([Ri[2, 1] - Ri[1, 2], Ri[0, 2] - Ri[2, 0], Ri[1, 0] - Ri[0, 1]])
        phi = np.arctan2(float(v @ v_hat), np.trace(Ri) - 1.0)
        out[i] = phi * v
    return out[0] if len(out) == 1 else out


def normalize_twist(v):
    V = _row6(v)
    out = np.zeros((V.shape[0], 7))
    for i, vi in enumerate(V):
        if np.linalg.norm(vi) <= _EPS:
            out[i, 5] = 1.0
            continue
        theta = np.linalg.norm(vi[:3])
        if theta <= _EPS:
            theta = np.linalg.norm(vi[3:6])
            out[i, 6] = theta
            out[i, 3:6] = vi[3:6] / theta
        else:
            out[i, 6] = theta
            out[i, 0:6] = vi[0:6] / theta
    return out[0] if len(out) == 1 else out


def se_twist(v):
    V = _row6(v)
    Ts = np.zeros((V.shape[0], 4, 4))
    for i, vi in enumerate(V):
        Ts[i, :3, :3] = so_w(vi[:3])
        Ts[i, :3, 3] = vi[3:6]
    return Ts[0] if len(Ts) == 1 else Ts


def exp_twist(v):
    V = _row6(v)
    Ts = np.zeros((V.shape[0], 4, 4))
    for i, vi in enumerate(V):
        Ts[i, 3, 3] = 1.0
        sa = normalize_twist(vi)
        s = sa[:6]
        theta = sa[6]
        W = so_w(s[:3])
        Ts[i, :3, :3] = exp_w(vi[:3])
        Ts[i, :3, 3] = (np.eye(3) * theta + (1 - np.cos(theta)) * W + (theta - np.sin(theta)) * (W @ W)) @ s[3:6]
    return Ts[0] if len(Ts) == 1 else Ts


def tform_inv(T):
    Ts = _as_stack(np.asarray(T, dtype=float), 4, 4)
    invs = np.zeros_like(Ts)
    for i, Ti in enumerate(Ts):
        R = Ti[:3, :3]
        t = Ti[:3, 3]
        invs[i] = np.eye(4)
        invs[i, :3, :3] = R.T
        invs[i, :3, 3] = -R.T @ t
    return invs[0] if len(invs) == 1 else invs


def derivative_tform_inv(T, dT):
    T = _arr(T); dT = _arr(dT)
    R = T[:3, :3]; t = T[:3, 3]
    dR = dT[:3, :3]; dt = dT[:3, 3]
    invT = np.eye(4)
    invT[:3, :3] = R.T
    invT[:3, 3] = -R.T @ t
    dInvT = np.zeros((4, 4))
    dInvT[:3, :3] = dR.T
    dInvT[:3, 3] = -dR.T @ t - R.T @ dt
    return dInvT, invT


def adjoint_T(tform):
    Ts = _as_stack(np.asarray(tform, dtype=float), 4, 4)
    Ads = np.zeros((Ts.shape[0], 6, 6))
    for i, T in enumerate(Ts):
        R = T[:3, :3]
        Ads[i, :3, :3] = R
        Ads[i, 3:6, 3:6] = R
        Ads[i, 3:6, :3] = so_w(T[:3, 3]) @ R
    return Ads[0] if len(Ads) == 1 else Ads


def derivative_adjoint_T(T, dT):
    R = T[:3, :3]
    t = T[:3, 3]
    dR = dT[:3, :3]
    dt = dT[:3, 3]
    z = np.zeros((3, 3))
    dAdT = _block2(dR, z, so_w(dt) @ R + so_w(t) @ dR, dR)
    AdT = _block2(R, z, so_w(t) @ R, R)
    return dAdT, AdT


def adjoint_twist(v):
    V = _row6(v)
    Ads = np.zeros((V.shape[0], 6, 6))
    for i, vi in enumerate(V):
        W = so_w(vi[:3])
        Ads[i, :3, :3] = W
        Ads[i, 3:6, 3:6] = W
        Ads[i, 3:6, :3] = so_w(vi[3:6])
    return Ads[0] if len(Ads) == 1 else Ads


def rttr(dh_row):
    alpha, a, d, theta = _vec(dh_row)
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st, 0.0, a], [st * ca, ct * ca, -sa, -d * sa], [st * sa, ct * sa, ca, d * ca], [0, 0, 0, 1.0]])


def rttr_inverse(T):
    T = _arr(T)
    theta = np.arctan2(-T[0, 1], T[0, 0])
    alpha = np.arctan2(-T[1, 2], T[2, 2])
    a = T[0, 3]
    sa, ca = np.sin(alpha), np.cos(alpha)
    d = T[1, 3] / -sa if abs(sa) > abs(ca) else T[2, 3] / ca
    return np.array([alpha, a, d, theta])


def forward_kin_dh(dh_table):
    T = np.eye(4)
    for row in np.asarray(dh_table, dtype=float):
        T = T @ rttr(row)
    return T


def make_tform(R=None, t=None):
    if R is None:
        return np.eye(4)
    Rs = _as_stack(np.asarray(R, dtype=float), 3, 3)
    tv = np.asarray(t, dtype=float)
    if tv.ndim == 1:
        tv = tv.reshape(1, 3)
    Ts = np.zeros((Rs.shape[0], 4, 4))
    for i in range(Rs.shape[0]):
        Ts[i] = np.eye(4)
        Ts[i, :3, :3] = Rs[i]
        Ts[i, :3, 3] = tv[i]
    return Ts[0] if len(Ts) == 1 else Ts


def logT(T):
    Ts = _as_stack(np.asarray(T, dtype=float), 4, 4)
    out = np.zeros((Ts.shape[0], 6))
    I = np.eye(3)
    for i, Ti in enumerate(Ts):
        R = Ti[:3, :3]
        p = Ti[:3, 3]
        out[i, :3] = logR(R)
        theta = np.linalg.norm(out[i, :3])
        if theta <= _EPS:
            out[i, 3:6] = p
            continue
        w = out[i, :3] / theta
        W = so_w(w)
        W2 = W @ W
        cot = 1.0 / np.tan(theta / 2.0)
        out[i, 3:6] = (I - theta / 2.0 * W + (1.0 - theta / 2.0 * cot) * W2) @ p
    return out[0] if len(out) == 1 else out


def sm2twist(sm):
    S = np.asarray(sm, dtype=float)
    if S.ndim == 1:
        S = S.reshape(1, -1)
    V = np.zeros((S.shape[0], 6))
    for i, row in enumerate(S):
        q = row[0:3]
        s = row[3:6] / np.linalg.norm(row[3:6])
        h = row[6]
        theta = row[7]
        if np.isinf(h):
            V[i] = np.r_[np.zeros(3), s] * theta
        else:
            V[i] = np.r_[s, np.cross(-s, q) + h * s] * theta
    return V[0] if len(V) == 1 else V


def twist2sm(v):
    V = _row6(v)
    sm = np.zeros((V.shape[0], 8))
    for i, vi in enumerate(V):
        theta = np.linalg.norm(vi[:3])
        if theta <= _EPS:
            sm[i, 6] = np.inf
            theta = np.linalg.norm(vi[3:6])
            if theta <= _EPS:
                sm[i, 5] = 1.0
            else:
                sm[i, 3:6] = vi[3:6] / theta
                sm[i, 7] = theta
        else:
            s = vi[:3] / theta
            h = float(s @ vi[3:6]) / theta
            q = np.linalg.pinv(so_w(s)) @ (h * s - vi[3:6] / theta)
            sm[i] = np.r_[q, s, h, theta]
    return sm[0] if len(sm) == 1 else sm


def screw_axis(p, dir, h):
    p = np.asarray(p, dtype=float); direction = np.asarray(dir, dtype=float)
    if p.ndim == 1: p = p.reshape(1, 3)
    if direction.ndim == 1: direction = direction.reshape(1, 3)
    n = p.shape[0]
    sm = np.c_[p, direction, np.ones((n, 1)) * h, np.ones((n, 1))]
    return sm2twist(sm)


def rot_screw_axis(p, dir):
    return screw_axis(p, dir, 0.0)


def tran_screw_axis(p, dir):
    return screw_axis(p, dir, np.inf)


def twist_dist(T1, T2):
    return logT(tform_inv(T1) @ T2)


def w_dr_A(r):
    r = _vec(r)
    nr = np.linalg.norm(r)
    if nr != 0:
        sr = so_w(r)
        return np.eye(3) - (1 - np.cos(nr)) / nr**2 * sr + (nr - np.sin(nr)) / nr**3 * (sr @ sr)
    return np.eye(3)


def derivative_Ar(r, dr):
    r = _vec(r); dr = _vec(dr)
    nr = np.linalg.norm(r)
    if nr == 0:
        return -so_w(dr)
    dnr = float(dr @ r) / nr
    skr = so_w(r); sk2r = skr @ skr
    dskr = so_w(dr); dsk2r = so_w(dr) @ so_w(r) + so_w(r) @ so_w(dr)
    a1 = (np.cos(nr) - 1) / nr**2
    a2 = (nr - np.sin(nr)) / nr**3
    da1 = -np.sin(nr) * dnr / nr**2 + 2 * (1 - np.cos(nr)) * dnr / nr**3
    da2 = (dnr - np.cos(nr) * dnr) / nr**3 - 3 * (nr - np.sin(nr)) * dnr / nr**4
    return a1 * dskr + da1 * skr + a2 * dsk2r + da2 * sk2r


def analytic_jacobian_matrix(Jb, T):
    R = T[:3, :3]
    r = logR(R)
    return _block2(np.linalg.inv(w_dr_A(r)), np.zeros((3, 3)), np.zeros((3, 3)), R) @ Jb


def derivative_jacobian_matrix(robot, q, qd):
    A = _A(robot); M = _M_stack(robot); ME = _ME(robot); n = _dof(robot)
    q = _vec(q); qd = _vec(qd)
    T = ME.copy(); Jb = np.zeros((6, n)); dJb = np.zeros((6, n)); dT = np.zeros((4, 4))
    for i in range(n - 1, -1, -1):
        dInvT, invT = derivative_tform_inv(T, dT)
        dAdT, adT = derivative_adjoint_T(invT, dInvT)
        Jb[:, i] = adT @ A[i]
        dJb[:, i] = dAdT @ A[i]
        tform = M[i] @ exp_twist(A[i] * q[i])
        dT = tform @ (se_twist(A[i]) @ T * qd[i] + dT)
        T = tform @ T
    return dJb, Jb, dT, T


def derivative_jacobian_matrix_analytic(robot, q, qd):
    dJb, Jb, dT, T = derivative_jacobian_matrix(robot, q, qd)
    qd = _vec(qd)
    R = T[:3, :3]
    r = logR(R)
    A_mat = w_dr_A(r)
    Vb = Jb @ qd
    wb = Vb[:3]
    dr = _solve(A_mat, wb)
    dA = derivative_Ar(r, dr)
    invA = np.linalg.inv(A_mat)
    z = np.zeros((3, 3))
    Ja = _block2(invA, z, z, R) @ Jb
    dJa = _block2(-invA @ dA @ invA, z, z, R @ so_w(wb)) @ Jb + _block2(invA, z, z, R) @ dJb
    return dJa, Ja, dT, T


def forward_kin_general(robot, q):
    A = _A(robot); M = _M_stack(robot); T = _ME(robot).copy(); q = _vec(q); n = _dof(robot)
    for i in range(n - 1, -1, -1):
        T = M[i] @ exp_twist(A[i] * q[i]) @ T
    return T


def jacobian_matrix(robot, q):
    A = _A(robot); M = _M_stack(robot); T = _ME(robot).copy(); q = _vec(q); n = _dof(robot)
    Jb = np.zeros((6, n))
    for i in range(n - 1, -1, -1):
        Jb[:, i] = adjoint_T(tform_inv(T)) @ A[i]
        T = M[i] @ exp_twist(A[i] * q[i]) @ T
    return Jb, T


def jacobian_matrix_analytic(robot, q):
    Jb, T = jacobian_matrix(robot, q)
    return analytic_jacobian_matrix(Jb, T), T


def jacobian_matrix_all(robot, q):
    A = _A(robot); M = _M_stack(robot); n = _dof(robot); q = _vec(q); ME = _ME(robot)
    J = np.zeros((n, 6, n))
    for i in range(n - 1, -1, -1):
        T = np.eye(4)
        for j in range(i, -1, -1):
            J[i, :, j] = adjoint_T(T) @ A[j]
            T = T @ exp_twist(-A[j] * q[j]) @ tform_inv(M[j])
    J[n - 1] = adjoint_T(tform_inv(ME)) @ J[n - 1]
    return J


def spatial_inertia_matrix(I, m, com):
    I = _arr(I); com = _vec(com)
    G = np.zeros((6, 6))
    sk = so_w(com)
    G[:3, :3] = I - m * ((com @ com) * np.eye(3) - np.outer(com, com) + sk @ sk)
    G[:3, 3:6] = m * sk
    G[3:6, :3] = -G[:3, 3:6]
    G[3:6, 3:6] = m * np.eye(3)
    return G


def transform_inertia_matrix(Ib, m, Tb):
    R = Tb[:3, :3]; t = Tb[:3, 3]
    return R @ Ib @ R.T + m * ((t @ t) * np.eye(3) - np.outer(t, t))


def transform_com_inertia_matrix(I, m, Tb):
    R = Tb[:3, :3]; t = Tb[:3, 3]
    I_rot = I - m * ((t @ t) * np.eye(3) - np.outer(t, t))
    return R.T @ I_rot @ R


def composite_inertia_matrix(I1, m1, I2, m2, T12):
    t = m2 / (m1 + m2) * T12[:3, 3]
    T = make_tform(np.eye(3), t)
    I = transform_inertia_matrix(I1, m1, T) + transform_inertia_matrix(I2, m2, tform_inv(T12) @ T)
    return I, T


def composite_inertias(I1, m1, I2, m2, T1, T2):
    ro = m2 / (m1 + m2)
    t = ro * T2[:3, 3] + (1 - ro) * T1[:3, 3]
    T = make_tform(np.eye(3), t)
    I = transform_inertia_matrix(I1, m1, tform_inv(T1)) + transform_inertia_matrix(I2, m2, tform_inv(T2))
    return I, T


def mass_matrix(robot, q):
    mass = _vec(_get(robot, 'mass')); inertia = _inertia_stack(robot); com = _arr(_get(robot, 'com'))
    A = _A(robot); M = _M_stack(robot); n = _dof(robot); q = _vec(q)
    J = np.zeros((n, 6, n)); Mq = np.zeros((n, n))
    for i in range(n - 1, -1, -1):
        G = spatial_inertia_matrix(inertia[i], mass[i], com[i])
        T = np.eye(4)
        for j in range(i, -1, -1):
            J[i, :, j] = adjoint_T(T) @ A[j]
            T = T @ exp_twist(-A[j] * q[j]) @ tform_inv(M[j])
        Mq += J[i].T @ G @ J[i]
    return Mq, J


def derivative_mass_matrix(robot, q, qd):
    mass = _vec(_get(robot, 'mass')); inertia = _inertia_stack(robot); com = _arr(_get(robot, 'com'))
    A = _A(robot); M = _M_stack(robot); n = _dof(robot); q = _vec(q); qd = _vec(qd)
    J = np.zeros((n, 6, n)); dJ = np.zeros((n, 6, n)); Mq = np.zeros((n, n)); dMq = np.zeros((n, n))
    for i in range(n - 1, -1, -1):
        G = spatial_inertia_matrix(inertia[i], mass[i], com[i])
        T = np.eye(4); dT = np.zeros((4, 4))
        for j in range(i, -1, -1):
            dAdT, AdT = derivative_adjoint_T(T, dT)
            J[i, :, j] = AdT @ A[j]
            dJ[i, :, j] = dAdT @ A[j]
            tform = exp_twist(-A[j] * q[j]) @ tform_inv(M[j])
            dT = (dT + T @ se_twist(-A[j]) * qd[j]) @ tform
            T = T @ tform
        Mq += J[i].T @ G @ J[i]
        dMq += dJ[i].T @ G @ J[i] + J[i].T @ G @ dJ[i]
    return dMq, Mq, dJ, J


def inverse_dynamics(robot, q, qd, qdd, F_ME=None):
    if F_ME is None:
        F_ME = np.zeros(6)
    mass = _vec(_get(robot, 'mass')); inertia = _inertia_stack(robot); com = _arr(_get(robot, 'com'))
    A = _A(robot); M = _M_stack(robot); ME = _ME(robot); n = _dof(robot)
    q = _vec(q); qd = _vec(qd); qdd = _vec(qdd); F_ME = _vec(F_ME)
    nu0 = np.zeros(6); dnu0 = np.r_[np.zeros(3), -_gravity(robot)]
    nu = np.zeros((n, 6)); dnu = np.zeros((n, 6)); tau = np.zeros(n)
    for i in range(n):
        T = exp_twist(-A[i] * q[i]) @ tform_inv(M[i])
        Map = adjoint_T(T)
        nu[i] = Map @ nu0 + A[i] * qd[i]
        dnu[i] = Map @ dnu0 + adjoint_twist(nu0) @ A[i] * qd[i] + A[i] * qdd[i]
        nu0 = nu[i]; dnu0 = dnu[i]
    F = F_ME.copy(); T = tform_inv(ME)
    for i in range(n - 1, -1, -1):
        G = spatial_inertia_matrix(inertia[i], mass[i], com[i])
        F = adjoint_T(T).T @ F + G @ dnu[i] - adjoint_twist(nu[i]).T @ (G @ nu[i])
        tau[i] = F @ A[i]
        T = exp_twist(-A[i] * q[i]) @ tform_inv(M[i])
    return tau


def inverse_dynamics_fext(robot, q, qd, qdd, fext):
    mass = _vec(_get(robot, 'mass')); inertia = _inertia_stack(robot); com = _arr(_get(robot, 'com'))
    A = _A(robot); M = _M_stack(robot); ME = _ME(robot); n = _dof(robot)
    q = _vec(q); qd = _vec(qd); qdd = _vec(qdd); fext = np.asarray(fext, dtype=float)
    if fext.shape[0] != 6 and fext.shape[1] == 6:
        fext = fext.T
    nu0 = np.zeros(6); dnu0 = np.r_[np.zeros(3), -_gravity(robot)]
    nu = np.zeros((n, 6)); dnu = np.zeros((n, 6)); tau = np.zeros(n)
    for i in range(n):
        T = exp_twist(-A[i] * q[i]) @ tform_inv(M[i])
        Map = adjoint_T(T)
        nu[i] = Map @ nu0 + A[i] * qd[i]
        dnu[i] = Map @ dnu0 + adjoint_twist(nu0) @ A[i] * qd[i] + A[i] * qdd[i]
        nu0 = nu[i]; dnu0 = dnu[i]
    T = np.eye(4); F = np.zeros(6)
    for i in range(n - 1, -1, -1):
        extf = fext[:, i] if i < n - 1 else adjoint_T(tform_inv(ME)).T @ fext[:, i]
        G = spatial_inertia_matrix(inertia[i], mass[i], com[i])
        F = adjoint_T(T).T @ F + G @ dnu[i] - adjoint_twist(nu[i]).T @ (G @ nu[i]) - extf
        tau[i] = F @ A[i]
        T = exp_twist(-A[i] * q[i]) @ tform_inv(M[i])
    return tau


def gravity_velocity_torque(robot, q, qd):
    return inverse_dynamics(robot, q, qd, np.zeros(_dof(robot)), np.zeros(6))


def gravity_torque(robot, q):
    return gravity_velocity_torque(robot, q, np.zeros(_dof(robot)))


def velocity_torque(robot, q, qd):
    return gravity_velocity_torque(robot, q, qd) - gravity_torque(robot, q)


def forward_dynamics(robot, q, qd, tao, F_ME=None):
    if F_ME is None: F_ME = np.zeros(6)
    n = _dof(robot); ME = _ME(robot)
    Mq, J = mass_matrix(robot, q)
    Jb = adjoint_T(tform_inv(ME)) @ J[n - 1]
    hqqd = inverse_dynamics(robot, q, qd, np.zeros(n), np.zeros(6))
    b = _vec(tao) - hqqd - Jb.T @ _vec(F_ME)
    return _solve(Mq, b)


def get_ext_torque(robot, q, fext):
    J = jacobian_matrix_all(robot, q)
    n = _dof(robot)
    fext = np.asarray(fext, dtype=float)
    if fext.shape[0] != 6 and fext.shape[1] == 6:
        fext = fext.T
    out = np.zeros(n)
    for i in range(n):
        out += J[i].T @ fext[:, i]
    return out


def manipulator_dynamics(robot, tao, F_ME, t, y):
    n = _dof(robot); y = _vec(y); q = y[:n]; qd = y[n:2*n]
    tau = tao(t, y) if callable(tao) else tao
    wrench = F_ME(t, y) if callable(F_ME) else F_ME
    return np.r_[qd, forward_dynamics(robot, q, qd, tau, wrench)]


def manipulator_dynamics_fext(robot, tao, fext, t, y):
    n = _dof(robot); y = _vec(y); q = y[:n]; qd = y[n:2*n]
    tau = tao(t, y) if callable(tao) else tao
    F = fext(t, y) if callable(fext) else fext
    hqqd = gravity_velocity_torque(robot, q, qd)
    ext_torque = get_ext_torque(robot, q, F)
    Mq, _ = mass_matrix(robot, q)
    return np.r_[qd, _solve(Mq, _vec(tau) - hqqd + ext_torque)]


def manipulator_dynamics_general(robot, controller, fext, t, y):
    n = _dof(robot); y = _vec(y); q = y[:n]; qd = y[n:2*n]
    F = fext(t, y) if callable(fext) else fext
    tau, daux = controller(t, y, F)
    Mq, _ = mass_matrix(robot, q)
    hqqd = gravity_velocity_torque(robot, q, qd)
    ext_torque = get_ext_torque(robot, q, F)
    daux = _vec(daux)
    return np.r_[qd, _solve(Mq, _vec(tau) - hqqd + ext_torque), daux]


def damping_least_square(A, B, lambda_=0.01, thr=np.inf):
    A = _arr(A); B = _arr(B)
    AAt = A @ A.T
    if np.linalg.cond(AAt) > thr:
        return _solve(A.T @ A + lambda_**2 * np.eye(A.shape[1]), A.T @ B)
    return A.T @ _solve(AAt, B)


def inverse_kin_general(robot, Td, ref, tol=(1e-4, 1e-4), max_iter=100):
    flag = 0; alpha = 1.0; lambda_ = 0.01; max_step = 0.1
    q = _vec(ref).copy(); Td = _arr(Td); Rd = Td[:3, :3]; pd = Td[:3, 3]
    for _ in range(max_iter):
        J, T = jacobian_matrix(robot, q)
        R = T[:3, :3]; p = T[:3, 3]
        xe = np.r_[logR(R.T @ Rd), R.T @ (pd - p)]
        if np.linalg.norm(xe[:3]) < tol[0] and np.linalg.norm(xe[3:6]) < tol[1]:
            return q, 1
        qe = damping_least_square(J, xe, lambda_, 0)
        norm_qe = np.linalg.norm(qe)
        if norm_qe > max_step:
            qe = qe / norm_qe * max_step
        q = q + alpha * qe
    return q, flag


def A_v(Z, M): return Z.T @ M @ Z

def A_x(J, M): return np.linalg.inv(J @ _solve(M, J.T))

def A_x_inv(J, M): return J @ _solve(M, J.T)

def A_x_x(J, M, x): return _solve(J @ _solve(M, J.T), x)

def pinv_J(J, M):
    tem = _solve(M, J.T)
    return tem @ np.linalg.inv(J @ tem)

def pinv_J_x(J, M, x): return _solve(M, J.T) @ _solve(J @ _solve(M, J.T), x)

def pinv_JT_x(J, M, x): return _solve(J @ _solve(M, J.T), J @ _solve(M, x))

def pinv_Z(Z, M): return _solve(Z.T @ M @ Z, Z.T @ M)

def null_proj(J, M, x): return x - pinv_J_x(J, M, J @ x)

def null_z(J):
    J = _arr(J); m, n = J.shape
    Jm = J[:m, :m]; Jr = J[:m, m:]
    return np.vstack([-_solve(Jm, Jr), np.eye(n - m)])

def derivative_null_z(J, dJ):
    J = _arr(J); dJ = _arr(dJ); m, n = J.shape
    dJm = dJ[:m, :m]; dJr = dJ[:m, m:]; Jm = J[:m, :m]; Jr = J[:m, m:]
    return np.vstack([_solve(Jm, dJm) @ _solve(Jm, Jr) - _solve(Jm, dJr), np.zeros((n - m, n - m))])

def derivative_pinv_J(J, M, dJ, dM):
    tem = _solve(M, J.T)
    dtem = -_solve(M, dM) @ _solve(M, J.T) + _solve(M, dJ.T)
    tem2 = np.linalg.inv(J @ tem)
    return (dtem - tem @ _solve(J @ tem, dJ @ tem + J @ dtem)) @ tem2

def derivative_pinv_J_x(J, M, dJ, dM, x):
    tem = _solve(M, J.T)
    dtem = -_solve(M, dM) @ _solve(M, J.T) + _solve(M, dJ.T)
    tem2 = J @ tem
    return (dtem - tem @ _solve(tem2, dJ @ tem + J @ dtem)) @ _solve(tem2, x)

def derivative_pinv_Z(Z, M, dZ, dM):
    tem1 = Z.T @ M @ Z
    dtem1 = dZ.T @ M @ Z + Z.T @ dM @ Z + Z.T @ M @ dZ
    tem2 = Z.T @ M
    dtem2 = dZ.T @ M + Z.T @ dM
    return -_solve(tem1, dtem1) @ _solve(tem1, tem2) + _solve(tem1, dtem2)

def Mu_v(Z, M, dZ, dM, C): return (Z.T @ C - A_v(Z, M) @ derivative_pinv_Z(Z, M, dZ, dM)) @ Z

def Mu_vx(J, M, dZ, dM, Z, C): return (Z.T @ C - A_v(Z, M) @ derivative_pinv_Z(Z, M, dZ, dM)) @ pinv_J(J, M)

def Mu_vx_x(J, Z, M, dZ, dM, C, x): return (Z.T @ C - A_v(Z, M) @ derivative_pinv_Z(Z, M, dZ, dM)) @ pinv_J_x(J, M, x)

def Mu_x(J, M, dJ, C): return (pinv_JT_x(J, M, C) - A_x_x(J, M, dJ)) @ pinv_J(J, M)

def Mu_x_x(J, M, dJ, C, x): return (pinv_JT_x(J, M, C) - A_x_x(J, M, dJ)) @ pinv_J_x(J, M, x)

def Mu_xv(J, M, dJ, Z, C): return (pinv_JT_x(J, M, C) - A_x_x(J, M, dJ)) @ Z


def principal_inertia_frame(Ib):
    vals, vecs = np.linalg.eigh(Ib)
    ind = np.argsort(vals)[::-1]
    return vecs[:, ind], np.diag(vals[ind])


def get_dynamics(robot):
    n = _dof(robot); mass = _vec(_get(robot, 'mass')); com = _arr(_get(robot, 'com')); inertia = _inertia_stack(robot)
    param = np.zeros(10 * n)
    for i in range(n):
        param[10*i:10*(i+1)] = np.r_[mass[i], mass[i] * com[i], inertia[i,0,0], inertia[i,1,1], inertia[i,2,2], inertia[i,0,1], inertia[i,0,2], inertia[i,1,2]]
    return param


def update_dynamics(robot, param):
    n = _dof(robot); param = _vec(param)
    mass = np.zeros(n); com = np.zeros((n, 3)); inertia = np.zeros((n, 3, 3))
    for i in range(n):
        pi = param[10*i:10*(i+1)]
        mass[i] = pi[0]
        com[i] = pi[1:4] / pi[0]
        inertia[i, 0, 0] = pi[4]; inertia[i, 1, 1] = pi[5]; inertia[i, 2, 2] = pi[6]
        inertia[i, 0, 1] = inertia[i, 1, 0] = pi[7]
        inertia[i, 0, 2] = inertia[i, 2, 0] = pi[8]
        inertia[i, 1, 2] = inertia[i, 2, 1] = pi[9]
    _set(robot, 'mass', mass); _set(robot, 'com', com); _set(robot, 'inertia', inertia)
    return robot


def read_robot_from_MDH(dh_table):
    dh_table = np.asarray(dh_table, dtype=float); n = dh_table.shape[0]
    M = np.zeros((n, 4, 4)); A = np.zeros((n, 6))
    for i in range(n):
        M[i] = rttr(dh_table[i]); A[i] = [0, 0, 1, 0, 0, 0]
    return Robot(dof=n, M=M, A=A, ME=np.eye(4), mass=np.zeros(n), inertia=np.zeros((n,3,3)), com=np.zeros((n,3)), gravity=np.array([0,0,-9.8]), TCP=np.eye(4))


def read_robot_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    n = int(data['dof'])
    Mraw = np.asarray(data['M'], dtype=float); Iraw = np.asarray(data['inertia'], dtype=float)
    M = np.zeros((n, 4, 4)); inertia = np.zeros((n, 3, 3))
    for i in range(n):
        M[i] = make_tform(exp_w(Mraw[i, :3]), Mraw[i, 3:6])
        inertia[i] = [[Iraw[i,0], Iraw[i,5], Iraw[i,4]], [Iraw[i,5], Iraw[i,1], Iraw[i,3]], [Iraw[i,4], Iraw[i,3], Iraw[i,2]]]
    data['M'] = M; data['inertia'] = inertia
    return Robot(dof=n, A=np.asarray(data['A'], dtype=float), M=M, ME=np.asarray(data.get('ME', np.eye(4)), dtype=float), TCP=np.asarray(data.get('TCP', np.eye(4)), dtype=float), mass=np.asarray(data.get('mass', np.zeros(n)), dtype=float), inertia=inertia, com=np.asarray(data.get('com', np.zeros((n,3))), dtype=float), gravity=np.asarray(data.get('gravity', [0,0,-9.8]), dtype=float))


def save_robot_json(robot, filename):
    n = _dof(robot); inertia = _inertia_stack(robot); M = _M_stack(robot)
    Iraw = np.zeros((n, 6)); Mraw = np.zeros((n, 6))
    for i in range(n):
        Iraw[i] = [inertia[i,0,0], inertia[i,1,1], inertia[i,2,2], inertia[i,1,2], inertia[i,0,2], inertia[i,0,1]]
        Mraw[i] = np.r_[logR(M[i,:3,:3]), M[i,:3,3]]
    data = dict(dof=n, A=_A(robot).tolist(), M=Mraw.tolist(), ME=_get(robot, 'ME').tolist(), TCP=_get(robot, 'TCP').tolist(), mass=_vec(_get(robot, 'mass')).tolist(), inertia=Iraw.tolist(), com=_arr(_get(robot, 'com')).tolist(), gravity=_gravity(robot).tolist())
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def generateRCM(d, a, b, c, p_rcm):
    p_rcm = _vec(p_rcm)
    x_rcm = np.array([1., 0., 0.]); z_rcm = np.array([0., 0., 1.]); y_rcm = np.cross(z_rcm, x_rcm)
    St = sm2twist(np.r_[p_rcm, z_rcm, np.inf, 1.0]) * d
    Sz = sm2twist(np.r_[p_rcm, z_rcm, 0.0, 1.0]) * -a
    Sx = sm2twist(np.r_[p_rcm, x_rcm, 0.0, 1.0]) * b
    Sy = sm2twist(np.r_[p_rcm, y_rcm, 0.0, 1.0]) * -c
    return exp_twist(Sx) @ exp_twist(Sy) @ exp_twist(Sz) @ exp_twist(St)


def m_c_g_matrix(robot, q, qd):
    """Compute M(q), C(q,qd), g(q) and TCP Jacobian quantities.

    This is a direct NumPy translation of the MATLAB implementation.
    """
    n = _dof(robot); q = _vec(q); qd = _vec(qd)
    mass = _vec(_get(robot, 'mass')); com = _arr(_get(robot, 'com')); inertia = _inertia_stack(robot)
    A = _A(robot); M = _M_stack(robot); ME = _ME(robot)
    J = np.zeros((n, 6, n)); dJ = np.zeros((n, 6, n)); pdJ = np.zeros((n, n, 6, n))
    Mq = np.zeros((n, n)); dMq = np.zeros((n, n)); pdMq = np.zeros((n, n, n)); C = np.zeros((n, n)); P = np.zeros((n, n)); g = np.zeros(n)
    Gs = np.zeros((n, 6, 6)); Jb = np.zeros((6, n)); dJb = np.zeros((6, n)); Tcp = np.eye(4); dTcp = np.zeros((4,4))
    for i in range(n - 1, -1, -1):
        Gs[i] = spatial_inertia_matrix(inertia[i], mass[i], com[i])
        T = np.eye(4); dT = np.zeros((4,4)); pdT = np.zeros((n,4,4))
        for j in range(i, -1, -1):
            dAdT, AdT = derivative_adjoint_T(T, dT)
            J[i,:,j] = AdT @ A[j]
            dJ[i,:,j] = dAdT @ A[j]
            tform = exp_twist(-A[j] * q[j]) @ tform_inv(M[j])
            pdtform = se_twist(-A[j]) @ tform
            pdT[j] = T @ pdtform
            for k in range(j + 1, i + 1):
                pdAdT, _ = derivative_adjoint_T(T, pdT[k])
                pdJ[k, i, :, j] = pdAdT @ A[j]
                pdT[k] = pdT[k] @ tform
            dT = (dT + T @ se_twist(-A[j]) * qd[j]) @ tform
            T = T @ tform
        Mq += J[i].T @ Gs[i] @ J[i]
        dMq += dJ[i].T @ Gs[i] @ J[i] + J[i].T @ Gs[i] @ dJ[i]
        for k in range(i + 1):
            drc = -pdT[k,:3,:3].T @ (T[:3,3] - com[i]) - T[:3,:3].T @ pdT[k,:3,3]
            P[i, k] = -mass[i] * (_gravity(robot) @ drc)
        if i == n - 1:
            invT = tform_inv(T)
            Tcp = invT @ ME
            dTcp = -invT @ dT @ invT @ ME
            adTb = adjoint_T(tform_inv(ME))
            Jb = adTb @ J[i]
            dJb = adTb @ dJ[i]
    for i in range(n):
        for j in range(n):
            # MATLAB uses pdJ(:,:,j,i); here pdJ[i, j]
            pdMq[:, :, i] += pdJ[i, j].T @ Gs[j] @ J[j] + J[j].T @ Gs[j] @ pdJ[i, j]
            g[i] += P[j, i]
    for k in range(n):
        for j in range(n):
            for i in range(n):
                C[k, j] += 0.5 * (pdMq[k, j, i] + pdMq[k, i, j] - pdMq[i, j, k]) * qd[i]
    return Mq, C, g, Jb, dJb, dMq, dTcp, Tcp

