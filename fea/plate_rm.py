from fenics import *
import numpy as np

def solve_plate_rm(
    q_expr,
    E,
    nu,
    h,
    bc_type="clamped",
    n=64,
):
    """
    Solve linear Reissner–Mindlin plate bending problem.

    Parameters
    ----------
    q_expr : fenics.Expression or Constant
        Transverse load q(x,y)
    E : float
        Young's modulus
    nu : float
        Poisson ratio
    h : float
        Plate thickness
    bc_type : str
        'clamped', 'simply_supported', or 'mixed'
    n : int
        Grid resolution per direction

    Returns
    -------
    w_grid : np.ndarray, shape (n, n)
        Displacement field on structured grid
    """

    # ------------------------------------------------------------------
    # Mesh
    # ------------------------------------------------------------------
    mesh = UnitSquareMesh(n - 1, n - 1)

    # ------------------------------------------------------------------
    # Function spaces
    # w  : displacement
    # tx : rotation about y
    # ty : rotation about x
    # ------------------------------------------------------------------
    V = FunctionSpace(mesh, "Lagrange", 2)
    W = MixedFunctionSpace([V, V, V])

    (w, tx, ty) = TrialFunctions(W)
    (v, sx, sy) = TestFunctions(W)

    # ------------------------------------------------------------------
    # Material parameters
    # ------------------------------------------------------------------
    D = E * h**3 / (12.0 * (1.0 - nu**2))
    G = E / (2.0 * (1.0 + nu))
    kappa = Constant(5.0 / 6.0)

    # ------------------------------------------------------------------
    # Variational formulation
    # ------------------------------------------------------------------
    a_bending = D * (
        inner(grad(tx), grad(sx)) +
        inner(grad(ty), grad(sy))
    ) * dx

    a_shear = kappa * G * h * (
        (tx - w.dx(0)) * (sx - v.dx(0)) +
        (ty - w.dx(1)) * (sy - v.dx(1))
    ) * dx

    a = a_bending + a_shear
    L = q_expr * v * dx

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------
    bcs = []

    if bc_type == "clamped":
        bcs = [
            DirichletBC(W.sub(0), Constant(0.0), "on_boundary"),
            DirichletBC(W.sub(1), Constant(0.0), "on_boundary"),
            DirichletBC(W.sub(2), Constant(0.0), "on_boundary"),
        ]

    elif bc_type == "simply_supported":
        bcs = [
            DirichletBC(W.sub(0), Constant(0.0), "on_boundary"),
        ]

    elif bc_type == "mixed":
        left = CompiledSubDomain("near(x[0], 0.0)")
        right = CompiledSubDomain("near(x[0], 1.0)")

        bcs = [
            DirichletBC(W.sub(0), Constant(0.0), left),
            DirichletBC(W.sub(1), Constant(0.0), left),
            DirichletBC(W.sub(2), Constant(0.0), left),
            DirichletBC(W.sub(0), Constant(0.0), right),
        ]

    else:
        raise ValueError(f"Unknown bc_type: {bc_type}")

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    solution = Function(W)
    solve(a == L, solution, bcs)

    w_sol, _, _ = solution.split(deepcopy=True)

    # ------------------------------------------------------------------
    # Project to structured grid (for ML)
    # ------------------------------------------------------------------
    grid = np.linspace(0.0, 1.0, n)
    w_grid = np.zeros((n, n))

    for i, x in enumerate(grid):
        for j, y in enumerate(grid):
            w_grid[j, i] = w_sol(Point(x, y))

    return w_grid
# Example usage
if __name__ == "__main__":
    # Define load
    q = Constant(-1.0)

    # Material and geometric properties
    E = 1.0e5      # Young's modulus
    nu = 0.3       # Poisson ratio
    h = 0.01       # Plate thickness

    # Solve plate bending problem
    w_result = solve_plate_rm(q, E, nu, h, bc_type="clamped", n=64)

    # Print maximum displacement
    print("Maximum displacement:", np.min(w_result))