import numpy as np
from sfepy.discrete import FieldVariable, Material, Integral, Equation, Equations, Problem
from sfepy.discrete.fem import Mesh, FEDomain, Field
from sfepy.terms import Term
from sfepy.discrete.conditions import Conditions, EssentialBC

def solve_plate_scalar(
    q_func,
    D=1.0,
    bc_type="clamped",
    n=64,
):
    """
    Linear scalar plate surrogate using Poisson equation proxy:
        D * Laplace(w) = q

    Parameters
    ----------
    q_func : callable
        Function q(x, y) -> load value
    D : float
        Bending stiffness
    bc_type : str
        'clamped' (zero displacement)
    n : int
        Grid resolution

    Returns
    -------
    w_grid : ndarray (n, n)
        Transverse displacement field
    """

    # ---------------- Mesh ----------------
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    coors = np.column_stack([xx.ravel(), yy.ravel()])

    # Connectivity for 4-node quads
    conn = []
    for j in range(n-1):
        for i in range(n-1):
            n0 = j*n + i
            conn.append([n0, n0+1, n0+n+1, n0+n])
    conn = np.array(conn, dtype=np.int32)

    descs = ['2_4'] * conn.shape[0]

    mesh = Mesh.from_data('plate', coors, None, [conn], descs)
    domain = FEDomain('domain', mesh)
    omega = domain.create_region('Omega', 'all')

    # ---------------- Field ----------------
    field_w = Field.from_args('w', np.float64, 'scalar', omega, approx_order=2)
    w = FieldVariable('w', 'unknown', field_w)
    v = FieldVariable('v', 'test', field_w, primary_var_name='w')

    # ---------------- Material ----------------
    mat = Material('mat', D=D)

    def load_fun(ts, coors, **kwargs):
        q = q_func(coors[:,0], coors[:,1])
        return {'val': q.reshape(-1,1)}
    load = Material('load', function=load_fun)

    integral = Integral('i', order=3)

    # ---------------- Weak Form ----------------
    t_diff = Term.new('dw_laplace.i.Omega(mat.D, v, w)', integral, omega, mat=mat, v=v, w=w)
    t_load = Term.new('dw_volume_lvf.i.Omega(load.val, v)', integral, omega, load=load, v=v)
    eqs = Equations([Equation('balance', t_diff - t_load)])

    # ---------------- Boundary Conditions ----------------
    gamma = domain.create_region('Gamma', 'vertices of surface', 'facet')

    if bc_type == 'clamped':
        bcs = Conditions([EssentialBC('w_bc', gamma, {'w.all': 0.0})])
    else:
        raise ValueError(f'Unknown bc_type {bc_type}')

    # ---------------- Solve ----------------
    pb = Problem('plate_scalar', equations=eqs)
    pb.set_bcs(ebcs=bcs)
    pb.set_solver_defaults()
    state = pb.solve()

    # Reshape to grid
    w_sol = state.get_parts()['w'].reshape((n,n))
    return w_sol
