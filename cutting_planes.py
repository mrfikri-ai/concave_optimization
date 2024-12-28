import numpy as np
from scipy.optimize import linprog

# First, generate the problem data
np.random.seed(123)
values = np.random.randint(35, 60, size=15)
cluster_size = np.random.randint(45, 70, size=15)

def qp_calculation(values, clstr_size):
    sensing_range = np.array(values)
    cluster_size = clstr_size
    task_num = len(cluster_size)
    # Set parameter for objective function 1
    q = np.kron(np.eye(task_num), sensing_range.T)
    Q = 2 * (q.T @ q)  # indefinite
    p = -2 * (cluster_size.T @ q)
    r = cluster_size @ cluster_size
    return Q, p, r

def ccv_qp_calculation(Q):
    alpha = 0.2
    eigQ = np.linalg.eigvals(Q)
    q = (Q - (np.max(eigQ.real) + alpha) * np.eye(len(Q)))
    return q

# Generate problem parameters
q, p, r = qp_calculation(values, cluster_size)
Q = ccv_qp_calculation(q)

# Build constraints
n = len(values)
m = len(cluster_size)
A1 = np.kron(np.eye(n), np.ones((1,m)))   # shape (n, n*m)
b1 = np.ones((n,1))
A2 = np.kron(np.ones((1,n)), np.eye(m))   # shape (m, n*m)
b2 = np.ones((m,1))

class ConcaveQPSolver:
    def __init__(self, Q, p, r, A, b):
        self.Q = Q
        self.p = p.reshape(-1)  # Ensure p is a 1D array
        self.r = float(r)       # Ensure r is a scalar
        self.A = A
        self.b = b
        self.n = Q.shape[0]     # dimension of the problem
        
    def objective_function(self, x):
        """Quadratic objective function"""
        x = np.asarray(x).reshape(-1)  # Ensure x is a 1D array
        return float(x @ self.Q @ x + self.p @ x + self.r)
    
    def generate_initial_vertices(self, K=5):
        """Step 1: Generate initial vertices through LP problems"""
        vertices = []
        for k in range(K):
            # Generate random direction for LP
            c = np.random.randn(self.n)
            # Solve LP
            result = linprog(c, A_ub=self.A, b_ub=self.b.flatten())
            if result.success:
                vertices.append(result.x)
        return np.array(vertices)
    
    def generate_tuy_cut(self, x_current):
        """Generate Tuy's cutting plane"""
        x_current = np.asarray(x_current).reshape(-1)  # Ensure x is a 1D array
        gradient = 2 * self.Q @ x_current + self.p
        a_cut = gradient
        b_cut = float(gradient @ x_current)
        return a_cut, b_cut
    
    def local_search(self, start_vertex, tabu_list, step_size=0.1):
        """Perform local search from a vertex"""
        current = np.asarray(start_vertex).reshape(-1)  # Ensure current is a 1D array
        best_value = self.objective_function(current)
        
        for _ in range(10):  # Limited number of local search iterations
            # Generate neighbors
            directions = np.eye(self.n)
            best_neighbor = None
            best_neighbor_value = float('inf')
            
            for direction in directions:
                neighbor = current + step_size * direction
                # Check if neighbor is feasible and not in tabu list
                if self.is_feasible(neighbor) and not any(np.allclose(neighbor, t) for t in tabu_list):
                    value = self.objective_function(neighbor)
                    if value < best_neighbor_value:
                        best_neighbor = neighbor
                        best_neighbor_value = value
            
            if best_neighbor_value < best_value:
                current = best_neighbor
                best_value = best_neighbor_value
            else:
                break
                
        return current
    
    def is_feasible(self, x):
        """Check if point satisfies constraints"""
        x = np.asarray(x).reshape(-1)  # Ensure x is a 1D array
        return np.all(self.A @ x <= self.b.flatten())
    
    def algorithm_tt(self, max_iter=100, tabu_size=10):
        """Implementation of Algorithm TT"""
        # Step 1: Generate initial vertices
        V = self.generate_initial_vertices()
        
        if len(V) == 0:
            print("No initial feasible vertices found")
            return None, float('inf')
            
        # Step 2: Initialize
        f_best = float('inf')
        x_best = None
        tabu_list = []
        k = 0
        X_k = {'A': self.A.copy(), 'b': self.b.flatten()}
        
        while k < max_iter and len(V) > 0:
            # Step 3: Choose vertex
            v = V[0]
            if any(np.allclose(v, t) for t in tabu_list):
                V = V[1:]
                continue
                
            # Step 4: Local search
            x_tilde = self.local_search(v, tabu_list)
            f_tilde = self.objective_function(x_tilde)
            
            if f_tilde < f_best:
                f_best = f_tilde
                x_best = x_tilde.copy()
                
            # Step 5-6: Generate and apply cut
            a_cut, b_cut = self.generate_tuy_cut(x_tilde)
            
            # Update constraints
            X_k['A'] = np.vstack([X_k['A'], a_cut])
            X_k['b'] = np.append(X_k['b'], b_cut)
            
            # Update tabu list
            tabu_list.append(x_tilde)
            if len(tabu_list) > tabu_size:
                tabu_list.pop(0)
                
            k += 1
            print(f"Iteration {k}: Best objective value = {f_best}")
            
        return x_best, f_best
    
    def rosen_hyperrectangle_cut(self):
        """Generate Rosen's hyperrectangle cuts"""
        subregions = []
        # Find variable bounds
        x_min = np.min(self.b)
        x_max = np.max(self.b)
        mid_point = (x_max + x_min) / 2
        
        # Create 2^p subregions (p=1 for simplicity)
        for i in range(2):
            if i == 0:
                A_new = np.vstack([self.A, np.eye(self.n)])
                b_new = np.vstack([self.b, mid_point * np.ones((self.n, 1))])
            else:
                A_new = np.vstack([self.A, -np.eye(self.n)])
                b_new = np.vstack([self.b, -mid_point * np.ones((self.n, 1))])
            subregions.append({'A': A_new, 'b': b_new})
        
        return subregions
    
    def algorithm_trt(self, max_iter=100):
        """Implementation of Algorithm TRT"""
        # Step 1-2: Same as TT
        initial_vertices = self.generate_initial_vertices()
        
        if len(initial_vertices) == 0:
            print("No initial feasible vertices found")
            return None, float('inf')
            
        # Step 3: Partition using Rosen's cuts
        subregions = self.rosen_hyperrectangle_cut()
        
        # Step 4: Apply TT to each subregion
        best_solution = None
        best_value = float('inf')
        
        for i, subregion in enumerate(subregions):
            print(f"\nProcessing subregion {i+1}/{len(subregions)}")
            # Create new solver instance for subregion
            subsolver = ConcaveQPSolver(self.Q, self.p.reshape(-1, 1), self.r, 
                                      subregion['A'], subregion['b'])
            x_sub, f_sub = subsolver.algorithm_tt(max_iter=max_iter//len(subregions))
            
            if x_sub is not None and f_sub < best_value:
                best_solution = x_sub.copy()
                best_value = f_sub
                print(f"New best solution found in subregion {i+1} with value: {best_value}")
                
        return best_solution, best_value

# Create the solver instance
solver = ConcaveQPSolver(Q=Q, p=p, r=r, A=np.vstack([A1, A2]), b=np.vstack([b1, b2]))

# Run Algorithm TT
print("Running Algorithm TT...")
x_tt, f_tt = solver.algorithm_tt()
if x_tt is not None:
    print(f"\nTT Solution found with objective value: {f_tt}")
    print("Solution vector:", x_tt)
    obj_val = 0.5 * x_tt.T @ q @ x_tt + p.T @ x_tt + r
    print("Objective value:", obj_val)
else:
    print("TT Algorithm failed to find a solution")

# Run Algorithm TRT
print("\nRunning Algorithm TRT...")
x_trt, f_trt = solver.algorithm_trt()
if x_trt is not None:
    print(f"\nTRT Solution found with objective value: {f_trt}")
    print("Solution vector:", x_trt)
    obj_val = 0.5 * x_trt.T @ q @ x_trt + p.T @ x_trt + r
    print("Objective value:", obj_val)
else:
    print("TRT Algorithm failed to find a solution")

###############################################################################
# 8) Gurobi/XPRESS Binary QP Solver (optional usage)
###############################################################################
import cvxpy as cp
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
        A1 @ x == b1,
        A2 @ x >= b2
    ]
    prob = cp.Problem(cp.Minimize(objective), constraints)
    
    prob.solve(solver=cp.XPRESS, xpress_options={
        "miprelstop": 1e-9, 
        "mipabsstop": 1e-9,
        "presolve": 2,
    })
    
    if prob.status not in ["infeasible", "unbounded", None]:
        return x.value, prob.value
    else:
        raise ValueError(f"XPRESS failed to find a solution. Status: {prob.status}")

def cvx_qp_calculation(Q):
        alpha = 0.2
        eigQ = np.linalg.eigvals(Q)
        q = (Q - (np.min(eigQ.real) - alpha) * np.eye(len(Q)))
        return q

b1_flat = b1.flatten()
b2_flat = b2.flatten()
A1_neg = -A1
b1_neg = -b1_flat
A2_neg = -A2
b2_neg = -b2_flat
A_combined = np.vstack([A1, A1_neg, A2_neg])
b_combined = np.concatenate([b1_flat, b1_neg, b2_neg])

# Optional: Solve with a binary QP (if it truly is integer/binary problem)
print("\n=== (Optional) Solving with Gurobi/XPRESS Binary QP ===")
QQ = cvx_qp_calculation(q)  # Another indefinite version
sol_gurobi, val_gurobi = solve_binary_qp_xpress(QQ, p, r, A1, b1_flat, A2, b2_flat)
val_actual_gurobi = 0.5 * sol_gurobi.dot(q).dot(sol_gurobi) + p.dot(sol_gurobi) + r
print("Binary QP solution:", sol_gurobi)
print("Binary solver objective value:", val_actual_gurobi)
