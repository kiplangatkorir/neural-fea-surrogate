# fea/plate_solver.py
from fenics import *
import numpy as np

class ReissnerMindlinPlateSolver:
    """Stable Reissner-Mindlin plate solver using quadrilateral elements and Selective Reduced Integration (SRI)."""
    
    def __init__(self, grid_n=64, deg=2):
        """
        Args:
            grid_n: Number of elements per side (will yield (grid_n+1) nodes per side).
            deg: Polynomial degree of the finite element (recommend 2 for balance).
        """
        self.grid_n = grid_n
        self.deg = deg
        
        # Create a mesh of QUADRILATERALS (crucial for SRI and better bending behavior)
        self.mesh = UnitSquareMesh.create(grid_n, grid_n, CellType.Type.quadrilateral)
        
        # Define MIXED FUNCTION SPACE: Deflection (scalar) + Rotation (vector)
        # Using Continuous Galerkin (CG) elements
        We = FiniteElement("Lagrange", self.mesh.ufl_cell(), deg)   # for deflection w
        Te = VectorElement("Lagrange", self.mesh.ufl_cell(), deg)   # for rotation θ
        self.W = FunctionSpace(self.mesh, MixedElement([We, Te]))
        
        # Separate function space for extracting just 'w' for output
        self.Ww = FunctionSpace(self.mesh, We)
        
        print(f"[Solver] Initialized: {grid_n}x{grid_n} quad mesh, CG{deg} elements.")
    
    def solve(self, load_func, E, nu, h, bc_type='clamped'):
        """
        Solves the Reissner-Mindlin plate bending problem.
        
        Args:
            load_func: A FEniCS Expression, Constant, or Function for the distributed load (pressure).
            E: Young's modulus (Pa).
            nu: Poisson's ratio.
            h: Plate thickness (m).
            bc_type: 'clamped' or 'simply_supported'.
            
        Returns:
            w_nodal: Numpy array of deflection values at mesh vertices, length (grid_n+1)**2.
        """
        # --- 1. Material Parameters ---
        D = E * h**3 / (12.0 * (1.0 - nu**2))  # Bending rigidity
        F = (E * h * 5.0/6.0) / (2.0 * (1.0 + nu))  # Shear rigidity (with shear correction factor κ=5/6)
        
        # --- 2. Define Trial and Test Functions on the Mixed Space ---
        u = TrialFunction(self.W)
        w, theta = split(u)  # split mixed function into deflection and rotation
        
        v = TestFunction(self.W)
        w_test, theta_test = split(v)
        
        # --- 3. Variational Form with SELECTIVE REDUCED INTEGRATION (SRI) ---
        # Bending energy term (full integration - standard dx)
        # Strain: κ(θ) = sym(grad(θ))
        kappa = sym(grad(theta))
        # Constitutive relation: M = D[(1-ν)κ + ν tr(κ)I]
        M = D * ((1.0 - nu) * kappa + nu * tr(kappa) * Identity(2))
        bending_energy = inner(M, sym(grad(theta_test))) * dx
        
        # Shear energy term (REDUCED INTEGRATION - key to avoid locking)
        # Reduced integration quadrature degree
        reduced_deg = 2 * self.deg - 2  # Common choice for SRI
        dx_shear = dx(metadata={'quadrature_degree': reduced_deg})
        
        # Shear strain: γ = θ - ∇w
        shear_strain = theta - grad(w)
        shear_energy = F * dot(shear_strain, theta_test - grad(w_test)) * dx_shear
        
        # External work from load
        L = load_func * w_test * dx
        
        # Complete bilinear and linear forms
        a = bending_energy + shear_energy
        L = L
        
        # --- 4. Boundary Conditions ---
        def boundary(x, on_boundary):
            return on_boundary
        
        # Clamped: w = 0 AND θ = (0,0) on boundary
        if bc_type == 'clamped':
            bc_w = DirichletBC(self.W.sub(0), Constant(0.0), boundary)  # Deflection
            bc_theta = DirichletBC(self.W.sub(1), Constant((0.0, 0.0)), boundary)  # Rotation
            bcs = [bc_w, bc_theta]
        # Simply Supported: w = 0 on boundary (rotation is free)
        elif bc_type == 'simply_supported':
            bcs = [DirichletBC(self.W.sub(0), Constant(0.0), boundary)]
        else:
            raise ValueError(f"Unknown bc_type: {bc_type}")
        
        # --- 5. Solve ---
        u_sol = Function(self.W)
        solve(a == L, u_sol, bcs,
              solver_parameters={'linear_solver': 'mumps', 'preconditioner': 'default'})
        
        # --- 6. Extract Deflection 'w' for Output ---
        w_sol = Function(self.Ww)
        w_sol.assign(u_sol.sub(0))  # Extract the deflection component from mixed solution
        
        # Get values at all vertices (nodes) of the mesh
        w_nodal = w_sol.compute_vertex_values(self.mesh)  # shape: (n_vertices,)
        
        return w_nodal
    
    def get_mesh_coordinates(self):
        """Returns the (x, y) coordinates of all mesh vertices."""
        return self.mesh.coordinates()  # shape: (n_vertices, 2)

# ============ Load Generation Utilities ============
def create_random_patch_load(mesh_coords, grid_n, intensity_range=(0.5, 2.0)):
    """
    Creates a FEniCS UserExpression for a random rectangular patch load.
    """
    class PatchLoad(UserExpression):
        def __init__(self, grid_n, **kwargs):
            super().__init__(**kwargs)
            self.grid_n = grid_n
            # Randomly choose a patch location and size
            patch_center_x = np.random.uniform(0.25, 0.75)
            patch_center_y = np.random.uniform(0.25, 0.75)
            patch_half_width = np.random.uniform(0.05, 0.2)
            self.value = np.random.uniform(*intensity_range)
            
            self.x_min = patch_center_x - patch_half_width
            self.x_max = patch_center_x + patch_half_width
            self.y_min = patch_center_y - patch_half_width
            self.y_max = patch_center_y + patch_half_width
        
        def eval(self, value, x):
            if (self.x_min <= x[0] <= self.x_max) and (self.y_min <= x[1] <= self.y_max):
                value[0] = self.value
            else:
                value[0] = 0.0
        
        def value_shape(self):
            return ()
    
    return PatchLoad(grid_n, degree=1)