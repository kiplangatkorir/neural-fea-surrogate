import numpy as np

from sfepy.discrete import (
    FieldVariable, Material, Integral, Equation, Equations, Problem
)
from sfepy.discrete.fem import Mesh, FEDomain, Field
from sfepy.terms import Term
from sfepy.discrete.conditions import Conditions, EssentialBC
from sfepy.base.base import Struct





def solve_plate_rm(
    q_func,
    E,
    nu,
    h,
    bc_type="clamped",
    n=64,
):
    """
    Linear Reissner–Mindlin plate solver using SfePy.

    Parameters
    ----------
    q_func : callable
        Function q(x, y) -> load value
    E, nu, h : float
        Material parameters
    bc_type : str
        'clamped', 'simply_supported', 'mixed'
    n : int
        Grid resolution

    Returns
    -------
    w_grid : ndarray (n, n)
        Transverse displacement field
    """

    # ------------------------------------------------------------
    # Mesh and domain
    # ------------------------------------------------------------
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    coors = np.column_stack([xx.ravel(), yy.ravel()])
    nx = ny = n

    conn = []
    for j in range(n - 1):
        for i in range(n - 1):
            n0 = j * n + i
            conn.append([n0, n0 + 1, n0 + n + 1, n0 + n])

    mesh = Mesh.from_data(
        "plate",
        coors,
        None,
        [np.array(conn, dtype=np.int32)],
        ["2_4"]
    )

    domain = FEDomain("domain", mesh)
    omega = domain.create_region("Omega", "all")

    # ------------------------------------------------------------
    # Fields
    # ------------------------------------------------------------
    field_w = Field.from_args("w", np.float64, "scalar", omega, approx_order=2)
    field_t = Field.from_args("theta", np.float64, "vector", omega, approx_order=2)

    w = FieldVariable("w", "unknown", field_w)
    v = FieldVariable("v", "test", field_w, primary_var_name="w")

    theta = FieldVariable("theta", "unknown", field_t)
    psi = FieldVariable("psi", "test", field_t, primary_var_name="theta")

    # ------------------------------------------------------------
    # Materials
    # ------------------------------------------------------------
    D = E * h**3 / (12.0 * (1.0 - nu**2))
    G = E / (2.0 * (1.0 + nu))
    kappa = 5.0 / 6.0

    mat = Material(
        "mat",
        D=D,
        G=G,
        h=h,
        kappa=kappa,
    )

    # Load material
    def load_fun(ts, coors, **kwargs):
        q = q_func(coors[:, 0], coors[:, 1])
        return {"val": q.reshape(-1, 1)}

    load = Material("load", function=load_fun)

    integral = Integral("i", order=3)

    # ------------------------------------------------------------
    # Weak form
    # ------------------------------------------------------------
    t_bending = Term.new(
        "dw_laplace.i.Omega(mat.D, psi, theta)",
        integral,
        omega,
        mat=mat,
        psi=psi,
        theta=theta,
    )

    t_shear = Term.new(
        "dw_dot.i.Omega(mat.kappa * mat.G * mat.h, psi, theta)",
        integral,
        omega,
        mat=mat,
        psi=psi,
        theta=theta,
    )

    t_load = Term.new(
        "dw_volume_lvf.i.Omega(load.val, v)",
        integral,
        omega,
        load=load,
        v=v,
    )

    eqs = Equations([
        Equation("bending", t_bending),
        Equation("shear", t_shear),
        Equation("load", t_load),
    ])

    # ------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------
    gamma = domain.create_region("Gamma", "vertices of surface", "facet")

    if bc_type == "clamped":
        bcs = Conditions([
            EssentialBC("w_bc", gamma, {"w.all": 0.0}),
            EssentialBC("t_bc", gamma, {"theta.all": 0.0}),
        ])

    elif bc_type == "simply_supported":
        bcs = Conditions([
            EssentialBC("w_bc", gamma, {"w.all": 0.0}),
        ])

    else:
        raise ValueError(f"Unknown bc_type: {bc_type}")

    # ------------------------------------------------------------
    # Problem definition and solve
    # ------------------------------------------------------------
    pb = Problem("plate_rm", equations=eqs)
    pb.set_bcs(ebcs=bcs)

    pb.set_solver_defaults()
    state = pb.solve()

    w_sol = state.get_parts()["w"].reshape((n, n))

    return w_sol
