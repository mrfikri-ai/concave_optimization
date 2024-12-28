import numpy as np
import os
import time

# For the concave QP solve (Step 1)
from scipy.optimize import minimize, LinearConstraint, Bounds

# For the small LP in Step 2
from scipy.optimize import linprog

###############################################################################
# 1) SciPy-based function to "maximize phi(x) = -1/2 x^T Q x + c^T x + r"
#    subject to A x <= b, x >= 0.
###############################################################################

def solve_concave_qp_max_scipy(A, b, Q, c, r, x0=None):
    """Solve the concave QP with improved numerical stability"""
    n = Q.shape[0]
    if x0 is None:
        x0 = np.zeros(n)

    # Add small regularization to Q for numerical stability
    epsilon = 1e-8
    Q_reg = Q + epsilon * np.eye(n)
    
    def f(x):
        # We minimize 0.5 x^T (Q_reg) x - c^T x - r,
        # which is the negative of our original objective (-phi).
        # Because phi(x) = -0.5 x^T Q x + c^T x + r.
        return 0.5 * x.dot(Q_reg).dot(x) - c.dot(x) - r
    
    def grad_f(x):
        # Gradient of the function we're minimizing
        return Q_reg.dot(x) - c
    
    def hess_f(x):
        return Q_reg

    lin_con = LinearConstraint(A, -np.inf, b)
    bnds = Bounds(lb=np.zeros(n), ub=np.full(n, np.inf))

    # Try multiple optimization methods in order
    methods = [
        ('SLSQP', {'ftol': 1e-6, 'maxiter': 1000}),
        ('trust-constr', {
            'xtol': 1e-4,
            'gtol': 1e-4,
            'barrier_tol': 1e-4,
            'maxiter': 2000,
            'factorization_method': None
        })
    ]

    best_result = None
    best_value = np.inf

    for method, options in methods:
        try:
            if method == 'SLSQP':
                res = minimize(
                    fun=f,
                    x0=x0,
                    method=method,
                    jac=grad_f,
                    constraints=[lin_con],
                    bounds=bnds,
                    options=options
                )
            else:
                res = minimize(
                    fun=f,
                    x0=x0,
                    method=method,
                    jac=grad_f,
                    hess=hess_f,
                    constraints=[lin_con],
                    bounds=bnds,
                    options=options
                )
            
            if res.success and (best_result is None or res.fun < best_value):
                best_result = res
                best_value = res.fun
                
        except Exception as e:
            print(f"Warning: Method {method} failed with error: {str(e)}")
            continue

    if best_result is not None:
        return best_result.x
    
    # If all methods fail, try one last attempt with higher regularization
    try:
        Q_reg = Q + 1e-4 * np.eye(n)  # Stronger regularization
        res = minimize(
            fun=lambda x: 0.5 * x.dot(Q_reg).dot(x) - c.dot(x) - r,
            x0=x0,
            method='SLSQP',
            jac=lambda x: Q_reg.dot(x) - c,
            constraints=[lin_con],
            bounds=bnds,
            options={'ftol': 1e-4, 'maxiter': 2000}
        )
        if res.success:
            return res.x
    except:
        pass

    raise ValueError("All QP solver attempts failed to find a solution.")


###############################################################################
# 2) Replacing the "small LP in each direction" with SciPy linprog
###############################################################################

def solve_LP_in_direction_scipy(A, b, x0, direction):
    """
    In Step 2, we "maximize alpha subject to (x0 + alpha * direction) in feasible set".
    That is:
       max alpha
       s.t.  A(x0 + alpha*d) <= b
             x0 + alpha*d >= 0
             alpha >= 0

    We'll do that with scipy.optimize.linprog (which MINIMIZES an objective).
    We'll minimize -alpha, i.e. objective c = [ -1 ], to get max alpha.
    The variable is alpha (a scalar).
    """
    n = len(x0)
    c_obj = np.array([-1.0])  # Because linprog does "minimize c^T alpha"
    
    m = A.shape[0]

    G = []
    h = []
    # For each row i
    for i in range(m):
        coef_i = A[i,:].dot(direction)
        rhs_i  = b[i] - A[i,:].dot(x0)
        G.append(coef_i)
        h.append(rhs_i)

    # Nonnegativity: x0 + alpha*d >= 0 => -alpha*d[j] <= x0[j]
    for j in range(n):
        G.append(-direction[j])
        h.append(x0[j])

    G = np.array(G).reshape(-1,1)  # shape: (m+n, 1)
    h = np.array(h)

    # alpha >= 0 => bounds on alpha
    bounds_alpha = [(0, None)]  # (lower=0, upper=None)
    
    res = linprog(
        c=c_obj,
        A_ub=G,
        b_ub=h,
        bounds=bounds_alpha,
        method='highs'
    )
    if res.success and res.fun is not None and not np.isinf(res.fun):
        alpha_star = -res.fun  # since objective = -alpha
        return x0 + alpha_star * direction
    else:
        return x0


###############################################################################
# 3) Approximate eigenvectors (for partition directions in Step 1)
###############################################################################

def approximate_eigenvectors(Q, n_eig=None):
    """
    For the Hessian = -Q (Q is PSD => -Q is NSD => phi is concave),
    we can simply look at Q's eigen-decomposition.
    """
    vals, vecs = np.linalg.eigh(Q)
    # Sort by descending magnitude
    order = np.argsort(-np.abs(vals))
    if n_eig is not None:
        order = order[:n_eig]
    top_vecs = vecs[:, order]
    return top_vecs  # Already orthonormal from eigh


###############################################################################
# 4) The Falk–Hoffman code.
###############################################################################
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, value

class FalkHoffmanInstance(object):
    """
    Solve concave optimization problem on a convex polytope using 
    Falk–Hoffman "collapsing polytopes" approach. 
    """

    class solutionTreeNode(object):
        def __init__(self, problemInstance, x, y, w, s):
            self._problemInstance = problemInstance
            self._solution = self._problemInstance.combineSolutionVector(x, y, s)
            k = self._problemInstance.A.shape[1] + 1

            # "collapse" n+1 smallest elements to 0 if fewer than k are zero
            if np.sum(self._solution == 0) < k:
                k_smallest_indices = np.argpartition(self._solution, k)
                np.put(self._solution, k_smallest_indices[:k], 0)
                self._x, self._y, self._s = self._problemInstance.getSolutionComponents(self._solution)
            else:
                self._x = x
                self._w = w
                self._y = y
                self._s = s

            self._feasible = True if np.isclose(self._y, 0) else False
            self._children = []
            self._f = self._problemInstance.f(w)
            self._problemInstance.registerSolutionNode(self)
            print("New Node Created: x =", self._x, ", f =", self._f)

        def getLowestTargetFunctionLeaf(self):
            if self.isLeaf:
                return self
            else:
                currentLeaf = None
                currentMinValue = np.inf
                for c in self._children:
                    childBestLeaf = c.getLowestTargetFunctionLeaf()
                    if childBestLeaf.f < currentMinValue:
                        currentLeaf = childBestLeaf
                        currentMinValue = childBestLeaf.f
                return currentLeaf

        def branchOut(self):
            isBaseVector = (self._solution > 0)
            t = self._problemInstance.originalTableau
            B = t[:, isBaseVector]
            B_inv = np.linalg.inv(B)
            N = B_inv.dot(t[:, ~isBaseVector])
            currentRhs = B_inv.dot(self._problemInstance.b)
            yCol = B_inv.dot(t[:, self._problemInstance.A.shape[1]])
            yRow = np.argmax(np.isclose(yCol, 1))

            for N_col, j in enumerate(np.array(range(t.shape[1]))[~isBaseVector]):
                col = N[:, N_col]
                if col[yRow] <= 0:
                    continue
                thetaVector = currentRhs / col

                # new v
                solution_new_v = self._solution.copy()
                idx_positive = (thetaVector > 0)
                if not np.any(idx_positive):
                    continue
                min_theta = np.min(thetaVector[idx_positive])
                solution_new_v[isBaseVector] = currentRhs - min_theta * col
                solution_new_v[j] = min_theta
                v_x, v_y, v_s = self._problemInstance.getSolutionComponents(solution_new_v)
                if self._problemInstance.solutionNodeExists(v_x):
                    continue

                # new w
                solution_new_w = self._solution.copy()
                solution_new_w[isBaseVector] = currentRhs - thetaVector[yRow] * col
                solution_new_w[j] = thetaVector[yRow]
                w_x, w_y, w_s = self._problemInstance.getSolutionComponents(solution_new_w)

                self._children.append(
                    FalkHoffmanInstance.solutionTreeNode(self._problemInstance, v_x, v_y, w_x, v_s)
                )

        def setF(self, newValue):
            if self == self._problemInstance.solutionTreeRoot:
                self._f = newValue

        @property
        def isLeaf(self):
            return len(self._children) == 0

        @property
        def x(self):
            return self._x

        @property
        def y(self):
            return self._y

        @property
        def f(self):
            return self._f

        @f.setter
        def f(self, newValue):
            self.setF(newValue)

        @property
        def feasible(self):
            return self._feasible

    def __init__(self, f, A, b):
        self._f = f
        self._A = np.array(A)
        self._b = b
        self._a = np.sum(self._A**2, axis=1)**0.5
        self._originalTableau = self.getInitialTableau()
        self._solutionNodeDict = {}

        # solve CP to get root
        x, y, s = self.solveCP()
        self._solutionTreeRoot = FalkHoffmanInstance.solutionTreeNode(self, x, y, x, s)
        self._solutionTreeRoot.f = np.inf  # artificially set root's f => +inf

    def registerSolutionNode(self, node):
        self._solutionNodeDict[tuple(node.x)] = node

    def solutionNodeExists(self, x):
        return tuple(x) in self._solutionNodeDict

    def combineSolutionVector(self, x, y, s):
        return np.concatenate([x, [y], s])

    def getSolutionComponents(self, solution):
        x = solution[:self._A.shape[1]]
        y = solution[self._A.shape[1]]
        s = solution[self._A.shape[1] + 1:]
        return x, y, s

    def getInitialTableau(self):
        # shape = (m rows, n + 1 + m columns)
        t = np.zeros((self._A.shape[0], self._A.shape[1] + 1 + self._A.shape[0]))
        t[:, :self._A.shape[1]] = self._A
        t[:, self._A.shape[1]] = self._a
        t[:, self._A.shape[1] + 1:] = np.identity(self._A.shape[0])
        return t

    @property
    def originalTableau(self):
        return self._originalTableau

    @property
    def f(self):
        return self._f

    @property
    def A(self):
        return self._A

    @property
    def b(self):
        return self._b

    @property
    def solutionTreeRoot(self):
        return self._solutionTreeRoot

    def solve(self, maxK=10000):
        k = 0
        status = "stopped"
        while k <= maxK:
            k += 1
            n = self._solutionTreeRoot.getLowestTargetFunctionLeaf()
            if np.isclose(n.y, 0.0):
                status = "optimal"
                break
            else:
                n.branchOut()

        # gather best feasible solutions
        optimalNodes = []
        optimalVal = np.inf
        for xKey, node in self._solutionNodeDict.items():
            if node.feasible:
                if np.isclose(node.f, optimalVal):
                    optimalNodes.append(xKey)
                elif node.f < optimalVal:
                    optimalNodes = [xKey]
                    optimalVal = node.f

        return optimalVal, optimalNodes, status

    def solveCP(self):
        """
        Solve the "CP" linear problem with improved numerical stability: 
        Maximize y s.t. A x + a_i * y + s_i = b_i, 
                      y <= (b_i - A x) / a_i, etc.
        Uses pulp internally with numerical tolerances.
        """
        prob = LpProblem("FHAlgorithmCP", LpMaximize)

        # Add small epsilon to prevent numerical issues
        eps = 1e-10
        
        x_vars = [LpVariable(f"x_{j}", lowBound=0) for j in range(self._A.shape[1])]
        y_var = LpVariable("y", lowBound=0)
        s_vars = [LpVariable(f"s_{i}", lowBound=0) for i in range(self._A.shape[0])]

        # objective
        prob += y_var, "Maximize_y"

        # constraints with numerical tolerance
        for i in range(self._A.shape[0]):
            # Add slack variables to make constraints less strict
            slack_pos = LpVariable(f"slack_pos_{i}", lowBound=0, upBound=eps)
            slack_neg = LpVariable(f"slack_neg_{i}", lowBound=0, upBound=eps)
            
            prob += lpSum(self._A[i, j] * x_vars[j] for j in range(self._A.shape[1])) \
                    + self._a[i] * y_var + s_vars[i] + slack_pos - slack_neg == self._b[i], f"Constraint_{i}"

            # Add tolerance to y-min constraints
            prob += y_var <= (self._b[i] - lpSum(self._A[i, j]*x_vars[j] 
                            for j in range(self._A.shape[1]))) / self._a[i] + eps, f"y-min_{i}"

        # Try solving with different solvers and parameters
        solvers = [
            ('PULP_CBC_CMD', {'msg': 0, 'primalTolerance': 1e-7, 'dualTolerance': 1e-7}),
            ('GLPK_CMD', {'msg': 0}),
            ('COIN_CMD', {'msg': 0})
        ]

        for solver, options in solvers:
            try:
                prob.solve(solver=solver, **options)
                if LpStatus[prob.status] == "Optimal":
                    xRet = np.array([var.varValue for var in x_vars])
                    yRet = y_var.varValue
                    sRet = np.array([var.varValue for var in s_vars])

                    # Clean up temporary files
                    for ftmp in os.listdir("."):
                        if ftmp.endswith((".mps", ".lp", ".sol")):
                            try:
                                os.remove(ftmp)
                            except:
                                pass

                    # Verify solution feasibility with tolerance
                    if self.verify_cp_solution(xRet, yRet, sRet, tolerance=1e-6):
                        return xRet, yRet, sRet
            except:
                continue

        # If all solvers fail, try with relaxed tolerances
        try:
            prob.solve(solver='PULP_CBC_CMD', msg=0, primalTolerance=1e-5, dualTolerance=1e-5)
            if LpStatus[prob.status] == "Optimal":
                xRet = np.array([var.varValue for var in x_vars])
                yRet = y_var.varValue
                sRet = np.array([var.varValue for var in s_vars])
                
                if self.verify_cp_solution(xRet, yRet, sRet, tolerance=1e-4):
                    return xRet, yRet, sRet
        except:
            pass

        # If everything fails, try to return a feasible point
        try:
            return self.find_feasible_cp_point()
        except:
            raise ValueError("Initial CP could not be solved optimally and no feasible point found.")

    def verify_cp_solution(self, x, y, s, tolerance=1e-6):
        """Verify if a CP solution is feasible within tolerance"""
        for i in range(self._A.shape[0]):
            # Check Ax + ay + s = b
            lhs = np.dot(self._A[i], x) + self._a[i] * y + s[i]
            if abs(lhs - self._b[i]) > tolerance:
                return False
            
            # Check y <= (b - Ax)/a
            if y > (self._b[i] - np.dot(self._A[i], x)) / self._a[i] + tolerance:
                return False
        
        return True

    def find_feasible_cp_point(self):
        """Try to find any feasible point for CP"""
        x = np.zeros(self._A.shape[1])
        s = self._b.copy()  # This ensures Ax + s = b
        y = 0.0
        return x, y, s




###############################################################################
# 6) Apply the Falk–Hoffman algorithm to each refined subdomain
###############################################################################

class FalkHoffmanSolver:
    """
    Minimal wrapper for running Falk–Hoffman in Step 6 on each subdomain.
    """
    def __init__(self, phi_func):
        self.phi = phi_func

    def run_on_subdomain(self, A_sub, b_sub):
        """
        Actually run the Falk–Hoffman on subdomain A_sub x <= b_sub, x>=0
        """
        fh_inst = FalkHoffmanInstance(f=self.phi, A=A_sub, b=b_sub)
        val, points, status = fh_inst.solve()
        return val, points, status


###############################################################################
# 7) Main 6-step Algorithm (concave minimization), 
#    using the SciPy-based QP solver & SciPy-based small LP in Step 2.
###############################################################################

def concave_min_with_falkhoffman(A, b, Q, c, r, verbose=False):
    """
    Illustrates the 6 steps:
       1) Constrained maximum of phi(x)
       2) 'Multiple-cost-row' LP => get 2n boundary vertices
       3) Compute lower bound
       4) Determine active vertices
       5) Refine subdomains
       6) Apply Falk–Hoffman
    
    Returns (x_best, phi_best).
    """
    #################################################################
    # Step 1: Constrained maximum
    #################################################################
    x_max = solve_concave_qp_max_scipy(A, b, Q, c, r, x0=None)
    def phi(x): return -0.5 * x.dot(Q).dot(x) + c.dot(x) + r
    if verbose:
        print("Step 1) x_max =", x_max, "phi(x_max) =", phi(x_max))

    # approximate directions from Q (Hessian is -Q)
    M = approximate_eigenvectors(Q)  
    n = Q.shape[0]

    #################################################################
    # Step 2: Get 2n vertices by solving LP in +/- directions
    #################################################################
    vertices = []
    for i in range(n):
        d_plus  = M[:, i]
        d_minus = -M[:, i]
        v_plus  = solve_LP_in_direction_scipy(A, b, x_max, d_plus)
        v_minus = solve_LP_in_direction_scipy(A, b, x_max, d_minus)
        vertices.append(v_plus)
        vertices.append(v_minus)

    phi_vals = [phi(v) for v in vertices]
    idx_min = np.argmin(phi_vals)
    v_best = vertices[idx_min]
    phi_best = phi_vals[idx_min]
    if verbose:
        print("Step 2) Among 2n boundary vertices, best = ", v_best, " with phi =", phi_best)

    #################################################################
    # Step 3: Compute a lower bound
    #################################################################
    # For demonstration, we pick a trivial bound, e.g. -1e9
    phi_LB = -1e9  
    if verbose:
        print("Step 3) Using a trivial lower bound of", phi_LB)

    # If the lower bound >= best known, we can stop
    if phi_LB >= phi_best:
        if verbose:
            print("Step 3) Lower bound >= current best => returning best solution.")
        return (v_best, phi_best)

    #################################################################
    # Step 4: Determine active vertices
    #################################################################
    # Example: let's call a vertex 'active' if phi(v) is within 10% of phi_best
    active_vertices = []
    for v in vertices:
        if phi(v) <= 1.1 * phi_best:
            active_vertices.append(v)

    if verbose:
        print(f"Step 4) Found {len(active_vertices)} active vertices.")

    if not active_vertices:
        if verbose:
            print("No active vertices => stop. Best sol so far => phi =", phi_best)
        return (v_best, phi_best)

    #################################################################
    # Step 5: Refine subdomains
    #################################################################
    # Real code would add new constraints or do a branch (like x_i <= v_i or something).
    # I skipped that in this code and kept the same domain for demonstration.
    subdomains = [(A, b)]  


    #################################################################
    # Step 6: Apply Falk–Hoffman on each subdomain
    #################################################################
    fh_solver = FalkHoffmanSolver(phi)
    global_best_val = phi_best
    global_best_sol = v_best

    for (A_sub, b_sub) in subdomains:
        val, points, status = fh_solver.run_on_subdomain(A_sub, b_sub)
        if verbose:
            print(f"Falk–Hoffman subdomain => best feasible val = {val}, status={status}, points={points}")
        if val < global_best_val:
            global_best_val = val
            # Store the first feasible solution's x
            if points:
                global_best_sol = np.array(points[0])

    return (global_best_sol, global_best_val)


###############################################################################
# 8) Gurobi/XPRESS Binary QP Solver (example usage)
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
    
    # 1) Define the variable (binary)
    x = cp.Variable(n, boolean=True)
    
    # 2) Define the objective
    objective = 0.5 * cp.quad_form(x, Q) + p @ x + r
    
    # 3) Define the constraints
    constraints = [
        A1 @ x == b1,
        A2 @ x >= b2
    ]
    
    # 4) Formulate and solve
    prob = cp.Problem(cp.Minimize(objective), constraints)
    
    prob.solve(solver=cp.XPRESS, xpress_options={
        "miprelstop": 1e-9,   # Relative MIP gap
        "mipabsstop": 1e-9,   # Absolute MIP gap
        "presolve": 2,        # More aggressive presolve
    })
    
    # 5) Check solution status
    if prob.status not in ["infeasible", "unbounded", None]:
        return x.value, prob.value
    else:
        raise ValueError(f"XPRESS failed to find a solution. Status: {prob.status}")


###############################################################################
# Example: Using the entire pipeline
###############################################################################
if __name__ == "__main__":
    # 1) Build Q, p from example 3 (arbitrary demonstration data).
    np.random.seed(123)
    values = np.random.randint(35, 60, size=40)
    cluster_size = np.random.randint(45, 70, size=40)

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

    def cvx_qp_calculation(Q):
        alpha = 0.2
        eigQ = np.linalg.eigvals(Q)
        q = (Q - (np.min(eigQ.real) - alpha) * np.eye(len(Q)))
        return q

    q, p, r = qp_calculation(values, cluster_size)
    Q = ccv_qp_calculation(q)
    QQ = cvx_qp_calculation(q)

    # 2) Build A, b (simple demonstration constraints).
    n = len(values)
    m = len(cluster_size)
    A1 = np.kron(np.eye(n), np.ones((1,m)))   # shape (n, n*m)
    b1 = np.ones((n,1))
    A2 = np.kron(np.ones((1,n)), np.eye(m))   # shape (m, n*m)
    b2 = np.ones((m,1))

    # Convert them to <= constraints
    b1_flat = b1.flatten()
    b2_flat = b2.flatten()
    A1_neg = -A1
    b1_neg = -b1_flat
    A2_neg = -A2
    b2_neg = -b2_flat
    A_combined = np.vstack([A1, A1_neg, A2_neg])
    b_combined = np.concatenate([b1_flat, b1_neg, b2_neg])

    # 3) Run the 6-step method
    print("\n=== Running Rosen + Falk–Hoffman Example ===")
    start_time = time.time()
    sol_2, val_2 = concave_min_with_falkhoffman(
        A_combined,
        b_combined,
        Q,  # Q from example
        p,  # interpret as linear term
        r,
        verbose=True
    )
    end_time = time.time()
    print("Time elapsed (Rosen + FH):", end_time - start_time)

    # Evaluate the objective with the original 'q' matrix
    # (But be mindful that the code used Q=ccv_qp_calculation(q), so the objective is actually using that Q.)
    sol_2_rounded = np.round(sol_2)
    # We can re-check the objective with the "q" (or the "Q" used).
    # For demonstration:
    val_actual = 0.5 * sol_2_rounded.dot(q).dot(sol_2_rounded) + p.dot(sol_2_rounded) + r
    print("Rounded solution:", sol_2_rounded)
    print("Actual objective value (with original 'q'):", val_actual)

    # 4) (Optional) Solve with a binary QP approach
    #    This part only works if you have XPRESS installed.
    # print("\n=== Solving with Gurobi/XPRESS Binary QP ===")
    # sol_gurobi, val_gurobi = solve_binary_qp_xpress(QQ, p, r, A1, b1_flat, A2, b2_flat)
    # val_actual_gurobi = 0.5 * sol_gurobi.dot(q).dot(sol_gurobi) + p.dot(sol_gurobi) + r
    # print("Gurobi/XPRESS Binary Solution:", sol_gurobi)
    # print("Actual Objective Value (binary solver):", val_actual_gurobi)
