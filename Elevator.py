import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csr_matrix
from itertools import product, combinations, chain
import time


#region INITALIZING NUMBERS
#===================
start = time.time()

NUMBER_OF_FLOORS=6
s=NUMBER_OF_FLOORS
NUMBER_OF_ELEVATORS=2
p=NUMBER_OF_ELEVATORS
CAPASITY_ELEATOR=np.zeros(p)
m=10000

#time plug ins 
row_vec = np.arange(s)[None, :] 
col_vec = np.arange(s)[:, None]
T_elevator = np.round(np.abs(col_vec - row_vec) ** 0.9, 2)
print(T_elevator) 

idx = np.arange(s)
diff = idx[None, :] - idx[:, None]   # diff[i,j] = j - i
T_stairs = np.where(diff > 0, 2 * diff, np.where(diff < 0, -1 * diff, 0))
print(T_stairs)

T_dwell=np.array([0.5]*s)

#supply matrix 
E = np.array([
    [0, 4, 0, 0, 2, 0],
    [1, 0, 0, 3, 0, 0],
    [0, 0, 0, 0, 0, 2],
    [0, 1, 0, 0, 0, 0],
    [3, 0, 0, 0, 0, 1],
    [0, 0, 2, 0, 0, 0],
])
print(E)
#endregion

#region GETTING NUMS FOR EACH VARIABLE 
#=======================================
i_j_pairs = (s - 1) * (s - 1)

v_num=p*s
f_alpha_num=s*(s-1)*p
f_beta_num = s*s*p
f_lambda_num=s*p*i_j_pairs
f_eta_num=s*i_j_pairs
f_gamma_num=s*(s-1)
f_omega_num=s
#endregion

# region INDEXING
#================
idx = {}
col = 0

idx['f_omega']  = (col, col + f_omega_num);  col += f_omega_num
idx['f_alpha']  = (col, col + f_alpha_num);  col += f_alpha_num
idx['f_beta']   = (col, col + f_beta_num);   col += f_beta_num
idx['f_lambda'] = (col, col + f_lambda_num); col += f_lambda_num
idx['f_eta']    = (col, col + f_eta_num);    col += f_eta_num
idx['f_gamma']  = (col, col + f_gamma_num);  col += f_gamma_num

idx['v']        = (col, col + v_num);        col += v_num

total_vars = col

alpha_pair_lookup = {}
for d in range(s):
    idx_i = 0
    for i in range(s):
        if i == d:
            continue
        alpha_pair_lookup[(d, i)] = idx_i
        idx_i += 1


def pair_index_of(i, j, s): #tells us witch link we are dealing with 
    return i * (s - 1) + (j if j < i else j - 1)
    
def col_f_omega(i):
    return idx['f_omega'][0] + i

def col_f_alpha(i, e, d):
    return idx['f_alpha'][0] + d*(s-1)*p + alpha_pair_lookup[(d, i)]*p + e

def col_f_beta(j, e, d):
    return idx['f_beta'][0] + d*s*p + j*p + e

def col_f_lambda(d, e, pair_idx):
    return idx['f_lambda'][0] + d * p * i_j_pairs + e * i_j_pairs + pair_idx

def col_f_eta(d, pair_idx):
    return idx['f_eta'][0] + d * i_j_pairs + pair_idx

def col_f_gamma(i, d):
    return idx['f_gamma'][0] + d*(s-1) + alpha_pair_lookup[(d, i)]

def col_v(e, i):
    return idx['v'][0] + e * s + i

#Pair lookup, indexes each valid i,j pair given a destination d 
pair_lookup = {}
for d in range(s):
    pair_idx = 0
    for i in range(s):
        if i == d:
            continue
        for j in range(s):
            if j == i:
                continue
            pair_lookup[(d, i, j)] = pair_idx
            pair_idx += 1

# endregion 

#region COST FUNCTION 
#===================
c_obj = np.zeros(total_vars)

#STAIRS COSTS
for d in range(s):
    for i in range(s):
        if i == d:
            continue
        for j in range(s):
            if j == i:
                continue
            c_obj[col_f_eta(d, pair_lookup[(d, i, j)])] += T_stairs[i][j]

#ELEVATOR COSTS 
def C_lambda(i,j):
    #IF you want to add diffrent travel times and dwell times depending on the elevator this must be changed
    return T_elevator[i, j] + T_dwell[i] / 2 + T_dwell[j] / 2
for d in range(s):
    for i in range(s):
        if i == d:
            continue
        for j in range(s):
            if j == i:
                continue
            for e in range(p):
                c_obj[col_f_lambda(d, e, pair_lookup[(d, i, j)])] += C_lambda(i, j)
#WIATING COST 
#Put off for a second 

print(f"Initalizing took {time.time() - start:.2f} seconds")
#endregion 

#region A MATRIX
#================
a = time.time()

# Count rows needed for the v-gated f_lambda constraint
vgate1_rows = p * s          # one row per (e, i), upper-bounded; skip zero-term ones if you want exact count
vgate2_rows = p * s          # one row per (e, k)
vgate3_rows = 0
for i in range(s):
    for k in range(s):
        if k == i:
            continue
        lo, hi = min(i, k), max(i, k)
        vgate3_rows += (hi - lo - 1)
vgate3_rows *= p
vgate_total_rows = vgate1_rows + vgate2_rows + vgate3_rows

a_len = (
    vgate_total_rows+
    s * (s - 1) +                 # Eq. 5  (entry demand: f_gamma = E[o][d])
    s +                           # Eq. 6  (aggregate exit: f_omega = total demand)
    s +                           # Eq. 7  (f_eta + f_beta = f_omega)
    s * s * p +                   # Eq. 8  (flow balance at each stop per elevator)
    s * (s - 1)                   # Eq. 9  (floor-pool conservation)
)

b_l=np.zeros(a_len)
b_u=np.zeros(a_len)
A = lil_matrix((a_len, total_vars)) 
show=True 
row=0

#Eq 1
#to take people from floor i elevator needs to stop there 
row_before = row
for e in range(p):
    for i in range(s):
        row_vec = np.zeros(total_vars)
        any_term = False
        for d in range(s):
            if i == d:
                continue
            for k in range(s):
                if k == i:
                    continue
                if (d, i, k) not in pair_lookup:
                    continue
                row_vec[col_f_lambda(d, e, pair_lookup[(d, i, k)])] += 1
                any_term = True
        if not any_term:
            continue
        row_vec[col_v(e, i)] -= m
        A[row] = row_vec
        b_l[row] = -np.inf
        b_u[row] = 0
        row += 1

#Eq 2 
#To take people to floor k elevator needs to stop there 
for e in range(p):
    for k in range(s):
        row_vec = np.zeros(total_vars)
        any_term = False
        for d in range(s):
            for i in range(s):
                if i == k or i == d:
                    continue
                if (d, i, k) not in pair_lookup:
                    continue
                row_vec[col_f_lambda(d, e, pair_lookup[(d, i, k)])] += 1
                any_term = True
        if not any_term:
            continue
        row_vec[col_v(e, k)] -= m
        A[row] = row_vec
        b_l[row] = -np.inf
        b_u[row] = 0
        row += 1

#Eq 3
#To take people from i to k every j stop must be zero 
for e in range(p):
    for i in range(s):
        for k in range(s):
            if k == i:
                continue
            lo, hi = min(i, k), max(i, k)
            for j in range(lo + 1, hi):
                row_vec = np.zeros(total_vars)
                any_term = False
                for d in range(s):
                    if i == d:
                        continue
                    if (d, i, k) not in pair_lookup:
                        continue
                    row_vec[col_f_lambda(d, e, pair_lookup[(d, i, k)])] += 1
                    any_term = True
                if not any_term:
                    continue
                row_vec[col_v(e, j)] += m
                A[row] = row_vec
                b_l[row] = -np.inf
                b_u[row] = m
                row += 1
if show:
    print(f"V eqs: {row - row_before} rows| expected",(vgate_total_rows))

# Eq. 5: 
# entry flow from o toward destination d must equal demand E[o][d]
row_before = row
for o in range(s):
    for d in range(s):
        if d == o:
            continue
        A[row, col_f_gamma(o, d)] += 1
        b_l[row] = E[o][d]
        b_u[row] = E[o][d]
        row += 1
if show:
    print(f"Eq. 5: {row - row_before} rows | expected {s*(s-1)}")

# Eq. 6
# aggregate exit at floor d must equal total demand destined for d
row_before = row
for d in range(s):
    A[row, col_f_omega(d)] += 1
    total_demand = sum(E[o][d] for o in range(s) if o != d)
    b_l[row] = total_demand
    b_u[row] = total_demand
    row += 1
if show:
    print(f"Eq. 6: {row - row_before} rows | expected {s}")

# Eq 7: 
# f_eta(i->d) + f_beta(d) = f_omega(d)
row_before = row
for d in range(s):
    A[row, col_f_omega(d)] -= 1
    for e in range(p):
        A[row, col_f_beta(d, e, d)] += 1
    for i in range(s):
        if i == d:
            continue
        if (d, i, d) in pair_lookup:
            A[row, col_f_eta(d, pair_lookup[(d, i, d)])] += 1
    b_l[row] = 0
    b_u[row] = 0
    row += 1
if show:
    print(f"Eq. 7: {row - row_before} rows | expected {s}")

# Eq 8: 
# f_lambda(in) + f_alpha(j) = f_lambda(out) + f_beta(j)
row_before = row
for d in range(s):
    for j in range(s):
        for e in range(p):
            for i in range(s):
                if i == j:
                    continue
                if (d, i, j) in pair_lookup:
                    A[row, col_f_lambda(d, e, pair_lookup[(d, i, j)])] += 1
            if j != d:
                A[row, col_f_alpha(j, e, d)] += 1
            for k in range(s):
                if k == j:
                    continue
                if (d, j, k) in pair_lookup:
                    A[row, col_f_lambda(d, e, pair_lookup[(d, j, k)])] -= 1
            A[row, col_f_beta(j, e, d)] -= 1
            b_l[row] = 0
            b_u[row] = 0
            row += 1
if show:
    print(f"Eq. 8: {row - row_before} rows | expected {s*s*p}")

# Eq 9: 
# floor-pool conservation (mode-agnostic) at non-destination floors
row_before = row
for d, j in product(range(s), range(s)):
    if j == d:
        continue
    A[row, col_f_gamma(j, d)] += 1
    for i in range(s):
        if i == j:
            continue
        for e in range(p):
            if (d, i, j) in pair_lookup:
                A[row, col_f_lambda(d, e, pair_lookup[(d, i, j)])] += 1
        if (d, i, j) in pair_lookup:
            A[row, col_f_eta(d, pair_lookup[(d, i, j)])] += 1
    for k in range(s):
        if k == j:
            continue
        for e in range(p):
            if (d, j, k) in pair_lookup:
                A[row, col_f_lambda(d, e, pair_lookup[(d, j, k)])] -= 1
        if (d, j, k) in pair_lookup:
            A[row, col_f_eta(d, pair_lookup[(d, j, k)])] -= 1
    b_l[row] = 0
    b_u[row] = 0
    row += 1
if show:
    print(f"Eq. 9: {row - row_before} rows | expected {s*(s-1)}")

print(f"\nTotal rows: {row} | a_len: {a_len}")
A = A.tocsr()

print(f"A matrix took {time.time() - a:.2f} seconds")
#endregion

#region OPTIMIZATION
#===================
o = time.time()

cont = sum([f_alpha_num, f_beta_num, f_lambda_num, f_omega_num, f_eta_num, f_gamma_num])
x_var=[0]*cont+[1]*v_num
l=[0]*(total_vars) 
u=[m]*cont+[1]*v_num
constraints = LinearConstraint(A, b_l, b_u)
bounds = Bounds(l, u)

result = milp(
    c=c_obj,
    constraints=constraints,
    integrality=x_var,
    bounds=bounds,
)
x_opt = result.x
print(result.success,result.message)
print(result.fun)

print(f"Optimization took {time.time() - o:.2f} seconds")
#endregion 

def plot_flow_matrix(matrix, title, ax, vmax):
    im = ax.imshow(matrix, cmap='viridis', vmin=0, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("To floor")
    ax.set_ylabel("From floor")
    ax.set_xticks(range(s))
    ax.set_yticks(range(s))
    for i in range(s):
        for j in range(s):
            val = matrix[i, j]
            if val > 0.01:
                ax.text(j, i, f"{val:.1f}", ha='center', va='center',
                         color='white' if val > vmax/2 else 'black')

def output(x_opt):
    # === SERVED FLOORS (v) ===
    print("\n=== Served floors ===")
    for e in range(p):
        served = [i for i in range(s) if x_opt[col_v(e, i)] > 0.5]
        print(f"  Elevator {e}: serves floors = {served}")

    # === EXIT FLOWS (f_omega) ===
    print("\n=== Exit flows (f_omega) ===")
    for i in range(s):
        val = x_opt[col_f_omega(i)]
        if val > 0.01:
            print(f"  f_omega[i={i}] = {val:.2f}")

    # === STARTING FLOWS (f_gamma) ===
    print("\n=== Starting flows (f_gamma) ===")
    for d in range(s):
        for i in range(s):
            if i == d:
                continue
            val = x_opt[col_f_gamma(i, d)]
            if val > 0.01:
                print(f"  f_gamma[i={i}, d={d}] = {val:.2f}")

    # === BOARDING FLOWS (f_alpha) ===
    print("\n=== Boarding flows (f_alpha) ===")
    for d in range(s):
        for i in range(s):
            if i == d:
                continue
            for e in range(p):
                val = x_opt[col_f_alpha(i, e, d)]
                if val > 0.01:
                    print(f"  f_alpha[i={i}, e={e}, d={d}] = {val:.2f}")

    # === DE-BOARDING FLOWS (f_beta) ===
    print("\n=== De-boarding flows (f_beta) ===")
    for j in range(s):
        for e in range(p):
            for d in range(s):
                val = x_opt[col_f_beta(j, e, d)]
                if val > 0.01:
                    print(f"  f_beta[j={j}, e={e}, d={d}] = {val:.2f}")

    # === INTER-STOP FLOWS (f_lambda) ===
    print("\n=== Inter-stop flows (f_lambda) ===")
    elevator_matrices = [np.zeros((s, s)) for _ in range(p)]
    for e in range(p):
        for d in range(s):
            for i in range(s):
                if i == d:
                    continue
                for j in range(s):
                    if j == i:
                        continue
                    if (d, i, j) not in pair_lookup:
                        continue
                    val = x_opt[col_f_lambda(d, e, pair_lookup[(d, i, j)])]
                    if val > 0.01:
                        print(f"  f_lambda[e={e}, d={d}, i={i}, j={j}] = {val:.2f}")
                        elevator_matrices[e][i, j] += val

    # === STAIRS FLOWS (f_eta) ===
    print("\n=== Stairs flows (f_eta) ===")
    stairs_matrix = np.zeros((s, s))
    for d in range(s):
        for i in range(s):
            if i == d:
                continue
            for j in range(s):
                if j == i:
                    continue
                if (d, i, j) not in pair_lookup:
                    continue
                val = x_opt[col_f_eta(d, pair_lookup[(d, i, j)])]
                if val > 0.01:
                    print(f"  f_eta[d={d}, i={i}, j={j}] = {val:.2f}")
                    stairs_matrix[i, j] += val

    # === PLOTS ===
    n_plots = 2 + p
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))

    all_matrices = [E, stairs_matrix] + elevator_matrices
    vmax = max(mat.max() for mat in all_matrices)
    if vmax == 0:
        vmax = 1   # avoid a degenerate 0-0 color scale if everything's empty

    plot_flow_matrix(E, "Demand (E)", axes[0], vmax)
    plot_flow_matrix(stairs_matrix, "Stairs flow", axes[1], vmax)
    for e in range(p):
        served = [i for i in range(s) if x_opt[col_v(e, i)] > 0.5]
        plot_flow_matrix(elevator_matrices[e], f"Elevator {e} (stops: {served})", axes[2 + e], vmax)

    plt.tight_layout()
    plt.show()

output(x_opt)
