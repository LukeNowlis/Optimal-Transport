import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csr_matrix
from itertools import product, combinations, chain
import timeimport numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csr_matrix
from itertools import product, combinations, chain
import time
import math
from pyscipopt import Model, quicksum


#region INITALIZING NUMBERS
#===================
start = time.time()

NUMBER_OF_FLOORS=5
s=NUMBER_OF_FLOORS
NUMBER_OF_ELEVATORS=2
p=NUMBER_OF_ELEVATORS
CAPASITY_ELEATOR=5
m=10000

#Elevator times 
ACEL=0.9
DISCOUNT=0.7

row_vec = np.arange(s)[None, :] 
col_vec = np.arange(s)[:, None]
T_elevator = np.round((np.abs(col_vec - row_vec) ** ACEL)*DISCOUNT, 2)
print(T_elevator) 

#dwell times 
DWELL_TIME=0.3

T_dwell=np.array([DWELL_TIME]*s)

#stairs times 
STAIR_ACEL = 0.99     # below 1 makes direct trips slightly cheaper; 1.0 means no preference
UP_COST = 9          # cost per floor going up
DOWN_COST = 5

fl = np.arange(s)
diff = fl[None, :] - fl[:, None]   # diff[i,j] = j - i
T_stairs = np.where(diff > 0, UP_COST * np.abs(diff) ** STAIR_ACEL,
            np.where(diff < 0, DOWN_COST * np.abs(diff) ** STAIR_ACEL, 0))
print(np.round(T_stairs, 3))

#supply matrix 
E = np.array([
    [0, 13, 5, 15, 0],
    [0, 0, 15, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
    ])
print(E)
#endregion 

#region GETTING NUMS FOR EACH VARIABLE 
#=======================================
i_j_pairs = (s - 1) * (s - 1)

v_num=p*s
h_num=(s*(s-1))/2
f_alpha_num=s*(s-1)*p
f_beta_num = s*s*p
f_lambda_num=s*p*i_j_pairs
f_eta_num=s*i_j_pairs
f_gamma_num=s*(s-1)
f_omega_num=s
n_num=p
#endregion

# region INDEXING
#================

# h_pair_lookup: maps (e, T, count) -> a compact column offset within elevator e's h-block
h_pair_lookup = {}
h_idx = 0
for T in range(1, s):           # top floor, 1 to s-1
    for count in range(2, T + 2):   # count ranges from 2 to T+1 inclusive
        h_pair_lookup[(T, count)] = h_idx
        h_idx += 1
h_num = p * h_idx   # h_idx now equals (s-1)*s/2 exactly, times p elevators

idx = {}
col = 0

idx['f_omega']  = (col, col + f_omega_num);  col += f_omega_num
idx['f_alpha']  = (col, col + f_alpha_num);  col += f_alpha_num
idx['f_beta']   = (col, col + f_beta_num);   col += f_beta_num
idx['f_lambda'] = (col, col + f_lambda_num); col += f_lambda_num
idx['f_eta']    = (col, col + f_eta_num);    col += f_eta_num
idx['f_gamma']  = (col, col + f_gamma_num);  col += f_gamma_num

idx['v']        = (col, col + v_num);        col += v_num
idx['h']        = (col, col + h_num);        col += h_num
idx['n']     = (col, col + n_num);        col += n_num

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

def col_h(e, T, count):
    return idx['h'][0] + e * (h_idx) + h_pair_lookup[(T, count)]

def col_n(e):
    return idx['n'][0]+e

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

#region HEADWAY- Travel time options- NNEDS LOOKING INTO
def compute_headway(T, count):
    if count <= 1:
        return 0.0
    hop = T / (count - 1)
    travel_one_way = (count - 1) * DISCOUNT * (hop ** ACEL)
    stop_events = 2 * count - 2          # see note below
    dwell_total = stop_events * DWELL_TIME
    return round(2 * travel_one_way + dwell_total, 2)

headway_lookup = {
    (T, count): compute_headway(T, count)
    for (T, count) in h_pair_lookup
}

def get_elevator_headway(x_opt, e):
    for (T, count) in h_pair_lookup:
        if x_opt[col_h(e, T, count)] > 0.5:
            return headway_lookup[(T, count)], T, count
    return None, None, None   # should not happen if "exactly one h" constraint holds
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

#Count rows needs for h-related constraints 
rows_topmatch = 0
for T in range(1, s):
    for count in range(2, T + 2):
        rows_topmatch += 1              # the v(e,T) row
        rows_topmatch += (s - 1 - T)    # one row per j > T
rows_topmatch *= p
rows_countmatch = 2 * h_idx * p   # h_idx = (s-1)*s/2 from before
rows_exactlyone = p
a_len_h_block = rows_topmatch + rows_countmatch + rows_exactlyone

a_len = (
    vgate_total_rows+             #V equations 
    a_len_h_block+                #H equations 
    s * (s - 1) +                 # Eq. 5  (entry demand: f_gamma = E[o][d])
    s +                           # Eq. 6  (aggregate exit: f_omega = total demand)
    s +                           # Eq. 7  (f_eta + f_beta = f_omega)
    s * s * p +                   # Eq. 8  (flow balance at each stop per elevator)
    s * (s - 1)+                  # Eq. 9  (floor-pool conservation)
    p +                           #Every elevator stops at floor one   
    2*p                             #n is greater than f_alpha/K
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

#H equations 

# Top-floor match
row_before = row
for e in range(p):
    for (T, count), _ in h_pair_lookup.items():
        A[row, col_h(e, T, count)] += 1
        A[row, col_v(e, T)] -= 1
        b_l[row] = -np.inf
        b_u[row] = 0
        row += 1

        for j in range(T + 1, s):
            A[row, col_h(e, T, count)] += 1
            A[row, col_v(e, j)] += 1
            b_l[row] = -np.inf
            b_u[row] = 1
            row += 1

# Stop-count match
for e in range(p):
    for (T, count), _ in h_pair_lookup.items():
        row_vec_upper = np.zeros(total_vars)
        for i in range(s):
            row_vec_upper[col_v(e, i)] += 1
        row_vec_upper[col_h(e, T, count)] += m
        A[row] = row_vec_upper
        b_l[row] = -np.inf
        b_u[row] = count + m
        row += 1

        row_vec_lower = np.zeros(total_vars)
        for i in range(s):
            row_vec_lower[col_v(e, i)] -= 1
        row_vec_lower[col_h(e, T, count)] += m
        A[row] = row_vec_lower
        b_l[row] = -np.inf
        b_u[row] = -count + m
        row += 1

#One h per elevator 
for e in range(p):
    for (T, count), _ in h_pair_lookup.items():
        A[row, col_h(e, T, count)] += 1
    b_l[row] = 1
    b_u[row] = 1
    row += 1
print(f"H block : {row - row_before} rows | expected {a_len_h_block}")

# Eq. lobby: every elevator must stop at floor 0
row_before = row
for e in range(p):
    A[row, col_v(e, 0)] += 1
    b_l[row] = 1
    b_u[row] = 1
    row += 1
if show:
    print(f"Eq. lobby: {row - row_before} rows | expected {p}")

# Eq. n: F_alpha/k <= n(e) < F_alpha/k + 1   (so n = ceil(F_alpha / k))
row_before = row
for e in range(p):
    # lower bound: n - F_alpha/k >= 0
    A[row, col_n(e)] += 1
    for d in range(1, s):
        A[row, col_f_alpha(0, e, d)] -= 1 / CAPASITY_ELEATOR
    b_l[row] = 0
    b_u[row] = np.inf
    row += 1

    # upper bound: n - F_alpha/k <= 1 - small
    A[row, col_n(e)] += 1
    for d in range(1, s):
        A[row, col_f_alpha(0, e, d)] -= 1 / CAPASITY_ELEATOR
    b_l[row] = -np.inf
    b_u[row] = 1 - 1e-4
    row += 1
if show:
    print(f"Eq. n: {row - row_before} rows | expected {2 * p}")

if show:
    print(f"Eq. n: {row - row_before} rows | expected {p}")

print(f"\nTotal rows: {row} | a_len: {a_len}")
A = A.tocsr()

print(f"A matrix took {time.time() - a:.2f} seconds")
#endregion

#region OPTIMIZATION 
#===================
o = time.time()

cont = sum([f_alpha_num, f_beta_num, f_lambda_num, f_omega_num, f_eta_num, f_gamma_num])
bi= sum([v_num,h_num])

model = Model()

binary_blocks = {'v', 'h'}
vars_list = [None] * total_vars

integer_blocks = {'n'}
n_max = int(np.ceil(E.sum() / CAPASITY_ELEATOR)) 

#Establish variables, type and bounds 
for name, (start, end) in idx.items():
    for c in range(start, end):
        if name in binary_blocks:
            vars_list[c] = model.addVar(name=f"{name}_{c}", vtype="B")
        elif name in integer_blocks:
            vars_list[c] = model.addVar(name=f"{name}_{c}", vtype="I", lb=0, ub=n_max)
        else:
            vars_list[c] = model.addVar(name=f"{name}_{c}", vtype="C", lb=0, ub=m)

#Turn A matrix into constraint equations 
for r in range(row):
    start, end = A.indptr[r], A.indptr[r + 1]
    if start == end:
        continue              
    cols = A.indices[start:end]
    vals = A.data[start:end]
    expr = quicksum(v * vars_list[c] for c, v in zip(cols, vals) if v != 0)

    lo, hi = b_l[r], b_u[r]
    if lo == hi:
        model.addCons(expr == lo)
    else:
        if lo > -np.inf:
            model.addCons(expr >= lo)
        if hi < np.inf:
            model.addCons(expr <= hi)

#Objective function
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

#WAITING COST
WAIT_FACTOR = 0.5   # average wait is half a loop; set to 1.0 to go back to worst case
MID_CONSTANT = 0.55

def H_expr(e):
    return WAIT_FACTOR * quicksum(headway_lookup[(T, count)] * vars_list[col_h(e, T, count)]
                                  for (T, count) in h_pair_lookup)                 
def F_alpha_cost(i, e, direction):
    dests = range(i + 1, s) if direction == 'up' else range(0, i)
    return quicksum(vars_list[col_f_alpha(i, e, d)] for d in dests)

def F_lambda_cost(i, e, direction):
    targets = range(i + 1, s) if direction == 'up' else range(0, i)
    return quicksum(
        vars_list[col_f_lambda(d, e, pair_lookup[(d, i, j)])]
        for d in range(s) if d != i
        for j in targets)

#Mid floors 
def G(i,e,direction):
    H=H_expr(e)
    N=F_lambda_cost(i,e,direction)/CAPASITY_ELEATOR
    return(N+1)*H*F_alpha_cost(i,e,direction)*MID_CONSTANT

#BOTTUM FLOOR 
base_sum = 0
for e in range(p):
    n = vars_list[col_n(e)]
    H = H_expr(e)
    k = CAPASITY_ELEATOR
    base_sum += k * H * n * (n - 1) / 2
    base_sum += (F_alpha_cost(0, e, 'up') - (n - 1) * k) * H * n

#Mid floors 
mid_sum=0
for e in range(p):
    for i in range(1,s-1):
        mid_sum+=(G(i,e,'up')+G(i,e,'down'))
    mid_sum+=G(s-1,e,'down')
obj_expr = quicksum(c_obj[c] * vars_list[c] for c in range(total_vars) if c_obj[c] != 0)+base_sum+mid_sum

t = model.addVar(name="t", vtype="C", lb=None, ub=None)
model.addCons(t >= obj_expr)
model.hideOutput()
model.setObjective(t, "minimize")
model.optimize()

print(model.getStatus())
x_opt = np.array([model.getVal(v) for v in vars_list])


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
    print("\n=== Served floors ===")
    for e in range(p):
        served = [i for i in range(s) if x_opt[col_v(e, i)] > 0.5]
        print(f"  Elevator {e}: serves floors = {served}")

    print("\n=== Headway selection ===")
    for e in range(p):
        hw, T, count = get_elevator_headway(x_opt, e)
        print(f"  Elevator {e}: headway = {hw}, top = {T}, count = {count}")

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

def cost_breakdown(x_opt, model=None):
    k = CAPASITY_ELEATOR

    # --- stairs and elevator riding (the parts in c_obj) ---
    stairs_cost = 0.0
    elev_cost = 0.0
    for d in range(s):
        for i in range(s):
            if i == d:
                continue
            for j in range(s):
                if j == i:
                    continue
                pidx = pair_lookup[(d, i, j)]
                stairs_cost += T_stairs[i][j] * x_opt[col_f_eta(d, pidx)]
                for e in range(p):
                    elev_cost += C_lambda(i, j) * x_opt[col_f_lambda(d, e, pidx)]

    # --- numeric versions of F_alpha and F_lambda ---
    def num_alpha(i, e, direction):
        dests = range(i + 1, s) if direction == 'up' else range(0, i)
        return sum(x_opt[col_f_alpha(i, e, d)] for d in dests)

    def num_lambda(i, e, direction):
        targets = range(i + 1, s) if direction == 'up' else range(0, i)
        return sum(x_opt[col_f_lambda(d, e, pair_lookup[(d, i, j)])]
                   for d in range(s) if d != i for j in targets)

    # --- wait cost as a function of load, used for the marginal calculation ---
    def bottom_wait(F, H):
        n = math.ceil(F / k - 1e-9) if F > 1e-9 else 0
        return k * H * n * (n - 1) / 2 + (F - (n - 1) * k) * H * n

    def mid_wait(F_a, F_l, H):
        return (F_l / k + 1) * H * F_a * MID_CONSTANT

    # --- bottom floor: exact, uses n ---
    print("\n=== Bottom-floor wait (floor 0, up) ===")
    print(f"{'elev':>4} {'F_alpha':>8} {'n':>3} {'H':>7} {'cost':>8} {'marginal':>9}")
    base_cost = 0.0
    for e in range(p):
        H = get_elevator_headway(x_opt, e)[0] * WAIT_FACTOR
        n = round(x_opt[col_n(e)])
        F = num_alpha(0, e, 'up')
        cost = k * H * n * (n - 1) / 2 + (F - (n - 1) * k) * H * n
        marginal = bottom_wait(F + 1, H) - bottom_wait(F, H)
        base_cost += cost
        print(f"{e:>4} {F:>8.2f} {n:>3} {H:>7.2f} {cost:>8.2f} {marginal:>9.2f}")

    # --- mid floors (plus top floor, down only) ---
    print("\n=== Mid/top-floor wait (nonzero only) ===")
    print(f"{'elev':>4} {'floor':>5} {'dir':>5} {'F_alpha':>8} {'F_lambda':>9} {'N':>6} {'H':>7} {'cost':>8} {'marginal':>9}")
    mid_cost = 0.0
    for e in range(p):
        H = get_elevator_headway(x_opt, e)[0] * WAIT_FACTOR
        floors = [(i, d) for i in range(1, s - 1) for d in ('up', 'down')] + [(s - 1, 'down')]
        for i, direction in floors:
            F_a = num_alpha(i, e, direction)
            if F_a < 0.01:
                continue
            F_l = num_lambda(i, e, direction)
            N = F_l / k
            cost = (N + 1) * H * F_a * MID_CONSTANT
            marginal = mid_wait(F_a + 1, F_l + 1, H) - mid_wait(F_a, F_l, H)
            mid_cost += cost
            print(f"{e:>4} {i:>5} {direction:>5} {F_a:>8.2f} {F_l:>9.2f} {N:>6.2f} {H:>7.2f} {cost:>8.2f} {marginal:>9.2f}")

    total = stairs_cost + elev_cost + base_cost + mid_cost
    print("\n=== Cost summary ===")
    print(f"  Stairs riding     : {stairs_cost:8.2f}")
    print(f"  Elevator ride     : {elev_cost:8.2f}")
    print(f"  Wait (bottom)     : {base_cost:8.2f}")
    print(f"  Wait (mid/top)    : {mid_cost:8.2f}")
    print(f"  Recomputed sum    : {total:8.2f}")
    if model is not None:
        print(f"  SCIP objective    : {model.getObjVal():8.2f}")

output(x_opt)
cost_breakdown(x_opt, model)


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
