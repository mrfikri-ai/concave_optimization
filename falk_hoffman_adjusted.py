import numpy as np
from scipy import optimize
from scipy.linalg import eigh
from typing import Tuple, List, Set, Dict
from dataclasses import dataclass
import cvxpy as cp

###############################################################################
# 1. Updated Problem Setup (QP Calculation) 
###############################################################################
np.random.seed(123)
values = np.random.randint(35, 60, size=15)
cluster_size = np.random.randint(45, 70, size=15)

def qp_calculation(values, clstr_size):
    """
    Constructs Q, p, r for a quadratic objective of the form:
        f(x) = x^T Q x + p^T x + r
    (though we might scale things as needed for a concave objective).
    """
    sensing_range = np.array(values)
    cluster_size = clstr_size
    task_num = len(cluster_size)
    
    # Example parameter construction
    q = np.kron(np.eye(task_num), sensing_range.T) 
    # This q is used to build Q. Note that Q might be indefinite.
    Q = 2 * (q.T @ q)
    
    p = -2 * (cluster_size.T @ q)
    r = cluster_size @ cluster_size
    return Q, p, r

def solve_binary_qp_xpress(Q, p, r, A1, b1, A2, b2):
    """
    Solve a binary quadratic programming problem using CVXPY + XPRESS:
    minimize    0.5 x^T Q x + p^T x + r
    subject to  A1 x = b1
                A2 x >= b2
                x in {0, 1}^n
    """
    n = Q.shape[0]
    x = cp.Variable(n, boolean=True)
    objective = 0.5 * cp.quad_form(x, Q) + p @ x + r
    constraints = [
        A1 @ x == b1.flatten(),
        A2 @ x >= b2.flatten()
    ]
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(
        solver=cp.XPRESS, 
        xpress_options={
            "miprelstop": 1e-9, 
            "mipabsstop": 1e-9,
            "presolve": 2,
        }
    )
    
    if prob.status not in ["infeasible", "unbounded", None]:
        return x.value, prob.value
    else:
        raise ValueError(f"XPRESS failed to find a solution. Status: {prob.status}")

def cvx_qp_calculation(Q):
    """
    Shifts Q so that it becomes positive semidefinite (for a convex objective).
    """
    alpha = 0.2
    eigQ = np.linalg.eigvals(Q)
    q = (Q - (np.min(eigQ.real) - alpha) * np.eye(len(Q)))
    return q

def ccv_qp_calculation(Q):
    """
    Shifts Q so that it becomes negative semidefinite (making x^T Q x concave).
    We subtract (max eigenvalue + alpha)*I from Q if max eigenvalue > 0.
    """
    alpha = 0.2
    eigQ = np.linalg.eigvals(Q)
    # Shift Q downward so that it is more likely to be negative semidefinite
    q = Q - (np.max(eigQ.real) + alpha) * np.eye(len(Q))
    return q

# Generate the Q, p, r for the new objective
Q, p_vec, r_const = qp_calculation(values, cluster_size)

# Convert Q to a (likely) negative semidefinite matrix for concavity
q_concave = ccv_qp_calculation(Q)
q_convex = cvx_qp_calculation(Q)

# Dimensions: if 'values' and 'cluster_size' both have length 6,
# then dimension is n*m = 6*6 = 36 for x.
n = len(values)
m = len(cluster_size)
n_dims = n * m

# Build constraints using Kronecker products
A1 = np.kron(np.eye(n), np.ones((1, m)))  # shape (n, n*m)
b1 = np.ones((n, 1))
A2 = np.kron(np.ones((1, n)), np.eye(m))  # shape (m, n*m)
b2 = np.ones((m, 1))

# Combine them into a single set: A x <= b
A_combined = np.vstack((A1, A2))  # shape (n+m, n*m)
b_combined = np.vstack((b1, b2))  # shape (n+m, 1)

###############################################################################
# 2. Falk-Hoffman Solver (Partition-based Global Minimizer for Concave f)
###############################################################################
@dataclass
class Simplex:
    vertices: np.ndarray  # Each row is a vertex
    values: np.ndarray    # Function values at those vertices
    underestimator: float # Current underestimator (piecewise linear)

class FalkHoffmanSolver:
    def __init__(self, objective_func, constraints, tolerance=1e-6):
        """
        Initialize Falk-Hoffman algorithm solver
        
        Args:
            objective_func: Concave objective function to minimize
            constraints:   Linear constraints Ax ≤ b
            tolerance:     Convergence tolerance
        """
        self.phi = objective_func
        self.constraints = constraints
        self.tolerance = tolerance
        self.lower_bound = float('-inf')
        self.upper_bound = float('inf')
        self.best_vertex = None
        self.simplices: List[Simplex] = []
        
    def construct_initial_approximation(self, initial_vertices: np.ndarray) -> None:
        """
        Step 1: Construct initial piecewise linear underestimator
        """
        # Evaluate function at vertices
        values = np.array([self.phi(v) for v in initial_vertices])
        
        # Create initial simplex
        initial_simplex = Simplex(
            vertices=initial_vertices,
            values=values,
            underestimator=self._compute_underestimator(initial_vertices, values)
        )
        
        self.simplices = [initial_simplex]
        self.upper_bound = np.min(values)  # Because we are minimizing a concave function
        self.lower_bound = initial_simplex.underestimator
        self.best_vertex = initial_vertices[np.argmin(values)]

    def _compute_underestimator(self, vertices: np.ndarray, values: np.ndarray) -> float:
        """
        Solve an LP to find a linear underestimator L(x) = a·x + b with 
        L(v_i) ≤ phi(v_i) for all i. Then take the maximum b subject to those constraints.
        """
        n = vertices.shape[1]
        
        # For each vertex i: a·v_i + b <= values[i]
        A_ub = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
        b_ub = values
        
        # Maximize b => minimize -b
        c = np.zeros(n + 1)
        c[-1] = -1.0
        
        result = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
        if not result.success:
            return float('-inf')
        
        # The LP's objective is c^T x = -b, so b = -result.fun
        return -result.fun

    def identify_refinement_regions(self) -> List[int]:
        """
        Step 2: Identify regions (simplices) needing refinement,
        i.e., those whose gap (upper_bound - underestimator) > tolerance.
        """
        max_gap = float('-inf')
        regions_to_refine = []
        
        for i, simplex in enumerate(self.simplices):
            gap = self.upper_bound - simplex.underestimator
            if gap > self.tolerance:
                regions_to_refine.append(i)
                max_gap = max(max_gap, gap)
                
        return regions_to_refine

    def subdivide_simplex(self, simplex: Simplex) -> List[Simplex]:
        """
        Step 2: Subdivide a simplex into smaller ones; for example,
        find the longest edge and split. Here we keep a simple demonstration.
        """
        vertices = simplex.vertices
        n = vertices.shape[1]
        
        # In an (n)-dim. simplex, we typically have (n+1) vertices
        # We pick the longest edge:
        max_length = 0
        max_edge = (0, 1)
        
        for i in range(n+1):
            for j in range(i+1, n+1):
                length = np.linalg.norm(vertices[i] - vertices[j])
                if length > max_length:
                    max_length = length
                    max_edge = (i, j)
        
        # Midpoint on that edge
        midpoint = (vertices[max_edge[0]] + vertices[max_edge[1]]) / 2
        
        # Create two new simplices by replacing the 'max_edge[1]' vertex
        # in turn with the midpoint
        new_simplices = []
        for _ in range(2):
            new_vertices = vertices.copy()
            new_vertices[max_edge[1]] = midpoint
            new_values = np.array([self.phi(v) for v in new_vertices])
            new_simplices.append(
                Simplex(
                    vertices=new_vertices,
                    values=new_values,
                    underestimator=self._compute_underestimator(new_vertices, new_values)
                )
            )
            # For the second new simplex, we might do another small tweak,
            # but for demonstration, we keep it simple.
        return new_simplices

    def update_bounds(self) -> None:
        """
        Step 3 & 4: Update global lower_bound (max of underestimators)
                    and upper_bound (min function value).
        """
        # Lower bound
        self.lower_bound = max(s.underestimator for s in self.simplices)
        
        # Upper bound & best vertex
        min_value = float('inf')
        min_vertex = None
        
        for simplex in self.simplices:
            min_idx = np.argmin(simplex.values)
            if simplex.values[min_idx] < min_value:
                min_value = simplex.values[min_idx]
                min_vertex = simplex.vertices[min_idx]
        
        if min_value < self.upper_bound:
            self.upper_bound = min_value
            self.best_vertex = min_vertex

    def solve(self, initial_vertices: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Main routine implementing the Falk-Hoffman algorithm.
        """
        # Step 1
        self.construct_initial_approximation(initial_vertices)
        
        while True:
            # Step 2
            regions_to_refine = self.identify_refinement_regions()
            if not regions_to_refine:
                break
                
            new_simplices = []
            for i, s in enumerate(self.simplices):
                if i in regions_to_refine:
                    new_simplices.extend(self.subdivide_simplex(s))
                else:
                    new_simplices.append(s)
            
            self.simplices = new_simplices
            
            # Steps 3 & 4
            self.update_bounds()
            
            # Step 5: stopping criterion
            if self.upper_bound - self.lower_bound < self.tolerance:
                break
                
        return self.upper_bound, self.best_vertex

###############################################################################
# 3. GlobalMinimization Class: Partitioning and Using Falk-Hoffman
###############################################################################
class GlobalMinimization:
    def __init__(self, objective_func, constraints, n_dims):
        """
        Initialize the global minimization solver
        
        Args:
            objective_func: Concave objective function to minimize
            constraints:    Linear constraints Ax ≤ b
            n_dims:         Number of dimensions
        """
        self.phi = objective_func
        self.constraints = constraints
        self.n = n_dims
        
    def compute_constrained_maximum(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Step 1: Compute constrained maximum point of concave phi (i.e. we maximize phi).
        Also compute Hessian at that point for directions. 
        """
        x0 = np.zeros(self.n)
        
        # For SLSQP we transform A x <= b => b[i] - A[i,:] x >= 0
        def constraint_factory(i):
            return lambda x, i=i: self.constraints['b'][i] - np.dot(self.constraints['A'][i], x)
        
        constraints_list = []
        for i in range(len(self.constraints['A'])):
            constraints_list.append({
                'type': 'ineq',
                'fun': constraint_factory(i)
            })
        
        # Because we want to maximize a concave phi, we minimize -phi
        result = optimize.minimize(lambda x: -self.phi(x),
                                   x0,
                                   constraints=constraints_list,
                                   method='SLSQP')
        
        x_bar = result.x
        hessian = self._compute_hessian(x_bar)
        # Eigenvectors
        eigenvals, eigenvecs = eigh(hessian)
        return x_bar, eigenvecs

    def solve_multiple_cost_row_lp(self, x_bar: np.ndarray, ui: np.ndarray) \
            -> Tuple[List[np.ndarray], np.ndarray, float]:
        """
        Step 2: Solve multiple-cost-row LP to get vertices along directions ±u_i.
        We store these vertices to help define subdomains or tight simplices.
        """
        vertices = []
        phi_min = float('inf')
        v_bar = None
        
        for i in range(self.n):
            direction = ui[:, i]
            for sign in [-1, 1]:
                c = sign * direction
                # Solve LP:  max c'x s.t. A x <= b  => minimize -c'x
                result = optimize.linprog(-c,
                                          A_ub=self.constraints['A'],
                                          b_ub=self.constraints['b'],
                                          method='highs')
                if not result.success:
                    continue
                vtx = result.x
                vertices.append(vtx)
                
                val = self.phi(vtx)
                if val < phi_min:
                    phi_min = val
                    v_bar = vtx
        
        return vertices, v_bar, phi_min

    def compute_lower_bound(self, vertices: List[np.ndarray]) -> float:
        """
        Step 3: A simple approach to get a 'lower bound' is the max of phi(v) for v in vertices
                (since the function is concave and we are finding a global minimum).
        """
        phi_val = float('-inf')
        for v in vertices:
            val = self.phi(v)
            if val > phi_val:
                phi_val = val
        return phi_val

    def partition_feasible_region(self, x_bar: np.ndarray, v_bar: np.ndarray) -> List[Dict]:
        """
        Partition the feasible region by introducing a hyperplane passing 
        halfway between x_bar and v_bar along the direction (x_bar - v_bar).
        Returns a list of subdomains, each subdomain is a dictionary {'A': A, 'b': b}.
        """
        subdomains = []

        direction = x_bar - v_bar
        if np.linalg.norm(direction) < 1e-9:
            # If direction is near zero, return the entire domain as one subdomain
            subdomains.append(self.constraints)
            return subdomains

        # Create the hyperplane at the midpoint
        mid_point = 0.5 * (x_bar + v_bar)
        d = np.dot(direction, mid_point)

        # Subdomain 1: A x <= b + direction^T x <= d
        A_cut1 = np.vstack((self.constraints['A'], direction.reshape(1, -1)))
        b_cut1 = np.hstack((self.constraints['b'], d))
        subdomains.append({'A': A_cut1, 'b': b_cut1})

        # Subdomain 2: A x <= b + -direction^T x <= -d
        A_cut2 = np.vstack((self.constraints['A'], -direction.reshape(1, -1)))
        b_cut2 = np.hstack((self.constraints['b'], -d))
        subdomains.append({'A': A_cut2, 'b': b_cut2})

        return subdomains


    def determine_nonempty_subdomains(self, subdomains: List[Dict]) -> List[int]:
        """
        Step 4: For each subdomain, check feasibility quickly by an LP: 
                max 0 s.t. A_sub x <= b_sub. 
        """
        feasible_indices = []
        for i, sd in enumerate(subdomains):
            # We do a trivial LP: min 0 (or max 0), subject to A_sub x <= b_sub
            c = np.zeros(self.n)
            res = optimize.linprog(c, A_ub=sd['A'], b_ub=sd['b'], method='highs')
            if res.success:
                feasible_indices.append(i)
        return feasible_indices

    def compute_simplex_vertices(self, subdomain: Dict) -> List[np.ndarray]:
        """
        Step 5: For an n-dim feasible region, find some extreme points.
                We'll do repeated directions ±e_i to find corners (like a bounding box).
                This is just a demonstration and not necessarily a minimal vertex set.
        """
        vertices = []
        # For each coordinate direction e_i, we solve an LP in that subdomain.
        eye_n = np.eye(self.n)
        for i in range(self.n):
            for sign in [-1, 1]:
                c = sign * eye_n[:, i]
                result = optimize.linprog(-c, A_ub=subdomain['A'], b_ub=subdomain['b'], method='highs')
                if result.success:
                    vertices.append(result.x)
        # You can refine to remove duplicates, etc.
        return vertices

    def apply_falk_hoffman(self, subdomain: Dict, simplex_vertices: List[np.ndarray]) -> Tuple[float, np.ndarray]:
        """
        Step 6: Apply Falk-Hoffman on the subdomain. We pass the subdomain constraints 
                and an initial set of simplex vertices.
        """
        fh_solver = FalkHoffmanSolver(self.phi, subdomain)
        
        # We require an (N+1)-vertex simplex if truly subdividing in N-d space,
        # but as a simple fallback, pass in all the found vertices.
        if len(simplex_vertices) < 2:
            # If there's too few vertices, just replicate an easy scenario
            # or fall back on subdomain boundary logic.
            # We'll at least pass them as if they define some initial region.
            initial = np.vstack(simplex_vertices) if simplex_vertices else np.zeros((1, self.n))
        else:
            initial = np.vstack(simplex_vertices)
        
        phi_val, x_opt = fh_solver.solve(initial)
        return phi_val, x_opt

    def solve(self) -> Tuple[float, np.ndarray]:
        """
        Main routine that follows the paper's structure, incorporating 
        partitioning + subdomain elimination.
        """
        # --- Step 1: Constrained maximum (for directions)
        x_bar, ui = self.compute_constrained_maximum()

        # --- Step 2: Get multiple vertex candidates
        # along ± eigen-directions
        vertices, v_bar, phi_bar = self.solve_multiple_cost_row_lp(x_bar, ui)

        # --- Step 3: Compute a simple lower bound
        lower_bound = self.compute_lower_bound(vertices)
        
        # If that matches the best (phi_bar from v_bar), we are done
        if abs(phi_bar - lower_bound) < 1e-7:
            return phi_bar, v_bar
        
        # --- Partition feasible region
        subdomains = self.partition_feasible_region(x_bar, v_bar)
        
        # --- Step 4: Determine which subdomains are feasible
        feasible_subs = self.determine_nonempty_subdomains(subdomains)
        if not feasible_subs:
            # If none is feasible, we just return the best we had
            return phi_bar, v_bar
        
        # Optional: we can skip subdomains if min of phi on them 
        # is already above phi_bar, etc. (Here, we do a simple check.)
        
        phi_star = phi_bar
        v_star = v_bar
        
        # For each feasible subdomain
        for j in feasible_subs:
            # Step 5: Compute subdomain vertices
            S_j = self.compute_simplex_vertices(subdomains[j])
            
            # Check if the min over those vertices already exceeds phi_star
            # because f is concave => if all these corners have a bigger value,
            # the subdomain might not contain anything better than phi_star.
            local_min_val = min(self.phi(v) for v in S_j) if S_j else float('inf')
            if local_min_val > phi_star:
                # Exclude
                continue
            
            # Step 6: Apply Falk-Hoffman in that subdomain
            val_j_star, x_j_star = self.apply_falk_hoffman(subdomains[j], S_j)
            
            # Update global best
            if val_j_star < phi_star:
                phi_star = val_j_star
                v_star = x_j_star
        
        return phi_star, v_star

    # -----------------------
    # Helper: numeric Hessian
    # -----------------------
    def _compute_hessian(self, x: np.ndarray) -> np.ndarray:
        """
        Numeric Hessian via finite differences (expensive).
        """
        eps = 1e-8
        n = len(x)
        H = np.zeros((n, n))
        
        f0 = self.phi(x)
        for i in range(n):
            for j in range(n):
                x_pp = x.copy()
                x_pp[i] += eps
                x_pp[j] += eps

                x_p = x.copy()
                x_p[i] += eps

                x_q = x.copy()
                x_q[j] += eps

                fpp = self.phi(x_pp)
                fp  = self.phi(x_p)
                fq  = self.phi(x_q)
                
                # (f(x+e_i+e_j) - f(x+e_i) - f(x+e_j) + f(x)) / eps^2
                H[i, j] = (fpp - fp - fq + f0) / (eps * eps)
        return H

###############################################################################
# 4. Our Concave Objective
###############################################################################
def concave_objective(x):
    """
    A "concave" objective = x^T (q_concave) x + p^T x + r_const,
    where q_concave = Q - (lambda_max + alpha)*I (hopefully negative semidefinite).
    """
    x = np.array(x, dtype=float).flatten()
    quad_part = x @ q_concave @ x
    linear_part = p_vec @ x
    return quad_part + linear_part + r_const

###############################################################################
# 5. Putting It All Together
###############################################################################
def main():
    # Build constraints in the dictionary form: A x <= b
    constraints = {
        'A': A_combined,
        'b': b_combined.flatten()
    }
    
    # Create solver instance for the new concave QP
    solver = GlobalMinimization(concave_objective, constraints, n_dims)
    
    # Solve the problem using the multi-step approach (partitioning + Falk-Hoffman)
    phi_star, v_star = solver.solve()
    print("Global Minimization (Partition + Falk-Hoffman) =>")
    print("  Optimal value:", phi_star)
    print("  Optimal point shape:", v_star.shape)
    print("  Reshaped (for clarity):\n", v_star.reshape((n, m)))
    
    # Solve a binary QP version via CVXPY + XPRESS for comparison
    x_binary, obj_binary = solve_binary_qp_xpress(
        q_convex, p_vec, r_const, 
        A1, b1, A2, b2
    )
    print("\nBinary QP (CVXPY+XPRESS) =>")
    print("  Optimal value:", obj_binary)
    print("  Optimal point shape:", x_binary.shape)
    print("  Reshaped:\n", x_binary.reshape((n, m)))
    
    # Compare the two solutions in the original Q = indefinite sense
    falk_hoffman_val = 0.5 * v_star @ Q @ v_star + p_vec @ v_star + r_const
    cvx_val = 0.5 * x_binary @ Q @ x_binary + p_vec @ x_binary + r_const
    print("\nComparison of objective values under original Q (indefinite):")
    print("  Falk-Hoffman value:", falk_hoffman_val)
    print("  CVXPY (binary) value:", cvx_val)

if __name__ == "__main__":
    main()
