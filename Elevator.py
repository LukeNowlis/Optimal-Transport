import numpy as np
import matplotlib.pyplot as plt
import time
import math
from pyscipopt import Model, quicksum


#region INITALIZING NUMBERS
#===================
t_init_start = time.perf_counter()

NUMBER_OF_FLOORS=2
s=NUMBER_OF_FLOORS
NUMBER_OF_ELEVATORS=2
p=NUMBER_OF_ELEVATORS
CAPASITY_ELEVATOR=5
m=25

#Elevator times 
ACEL=0.9        #discount given to direct trips i.e. skipping floors is faster than stopping(just travel wise)
DISCOUNT=0.7    #discount given in general to elevator use vs. stairs (THis can be accounted for by changing stair costs)

row_vec = np.arange(s)[None, :] 
col_vec = np.arange(s)[:, None]
T_elevator = np.round((np.abs(col_vec - row_vec) ** ACEL)*DISCOUNT, 2)
print('ELEVATOR')
print(T_elevator) 

#dwell time(time taken while elevator is stopped at a floor)
DWELL_TIME=0.3

T_dwell=np.array([DWELL_TIME]*s)

#stairs times 
STAIR_ACEL = 0.99     # below 1 makes direct trips slightly cheaper; 1.0 means no preference
UP_COST = 7           # cost per floor going up
DOWN_COST = 4         #cost per floor going down  

fl = np.arange(s)
diff = fl[None, :] - fl[:, None]   # diff[i,j] = j - i
T_stairs = np.where(diff > 0, UP_COST * np.abs(diff) ** STAIR_ACEL,
            np.where(diff < 0, DOWN_COST * np.abs(diff) ** STAIR_ACEL, 0))
print('STAIRS')
print(np.round(T_stairs, 3))

#supply matrix
print('Supply')
rng = np.random.default_rng()            
E = rng.integers(0, m-10, size=(s, s))    
np.fill_diagonal(E, 0)              
print(E)
#endregion 

# region INDEXING
#================
#GETTING NUMS FOR EACH VARIABLE 
#how many pairs of floor to floor options there are given a destination, meaning i =! d and j=! i  
i_j_pairs_given_d = (s - 1)**2
v_num=p*s
h_num=(s*(s-1))/2
f_alpha_num=s*(s-1)*p
f_beta_num = s*s*p
f_lambda_num=s*p*i_j_pairs_given_d
f_eta_num=s*i_j_pairs_given_d
f_omega_num=s*(s-1)
n_num=p

# h_pair_lookup: maps (T, count) finds all possible Top floor and count combinations, useful for constraints and used later on 
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

def col_f_alpha(i, e, d):
    return idx['f_alpha'][0] + d*(s-1)*p + alpha_pair_lookup[(d, i)]*p + e

def col_f_beta(j, e, d):
    return idx['f_beta'][0] + d*s*p + j*p + e

def col_f_lambda(d, e, pair_idx):
    return idx['f_lambda'][0] + d * p * i_j_pairs_given_d + e * i_j_pairs_given_d + pair_idx

def col_f_eta(d, pair_idx):
    return idx['f_eta'][0] + d * i_j_pairs_given_d + pair_idx

def col_f_omega(i, d):
    return idx['f_omega'][0] + d*(s-1) + alpha_pair_lookup[(d, i)]

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

#region HEADWAY- Travel time options
#WHEN I USE THE WORD HEADWAY IT IS A MISUSE OF THE WORD, I REALLY MEAN TRAVEL TIME ROUND TRIP  

#computes headway based on Top floor and count of floors stoped at (uses inputs from the first section)
def compute_headway(T, count):
    if count <= 1:
        return 0.0
    hop = T / (count - 1)
    travel_one_way = (count - 1) * DISCOUNT * (hop ** ACEL)
    stop_events = 2 * count - 2
    dwell_total = stop_events * DWELL_TIME
    return round(2 * travel_one_way + dwell_total, 2)

#uses h_pair_lookup from earlier to create a dictionary of all possible headways given every T and count 
headway_lookup = {
    (T, count): compute_headway(T, count)
    for (T, count) in h_pair_lookup
}

#find the correct headway too use in headway_lookup by using the correct index 
def get_elevator_headway(x_opt, e):
    for (T, count) in h_pair_lookup:
        if x_opt[col_h(e, T, count)] > 0.5:
            return headway_lookup[(T, count)], T, count
    return None, None, None   # should not happen if "exactly one h" constraint holds
#endregion

#region OPTIMIZATION INITALIZATION 
#=================================
cont = sum([f_alpha_num, f_beta_num, f_lambda_num, f_omega_num, f_eta_num])
bi= sum([v_num,h_num])

model = Model()

binary_blocks = {'v', 'h'}
vars_list = [None] * total_vars

integer_blocks = {'n'}
n_max = int(np.ceil(E.sum() / CAPASITY_ELEVATOR)) 

#Establish variables, type and bounds 
for name, (start, end) in idx.items():
    for c in range(start, end):
        if name in binary_blocks:
            vars_list[c] = model.addVar(name=f"{name}_{c}", vtype="B")
        elif name in integer_blocks:
            vars_list[c] = model.addVar(name=f"{name}_{c}", vtype="I", lb=0, ub=n_max)
        else:
            vars_list[c] = model.addVar(name=f"{name}_{c}", vtype="C", lb=0, ub=m)
#endregion

#region ADDING CONSTRIANTS
#==========================

#EQ 1 and 2 
#Constaint on f_lambda, to take people from i to j elevator must stop at both floors 
for e in range(p):
    for (d, i, j), pair_idx in pair_lookup.items():
        f = vars_list[col_f_lambda(d, e, pair_idx)]
        model.addCons(f <= m * vars_list[col_v(e, i)])    # must stop at origin
        model.addCons(f <= m * vars_list[col_v(e, j)])    # must stop at destination

#EQ 3 
#Constraint on f_lambda, to take people from i to k elevator cant stop on any j floors 
for e in range(p):
    for (i, k) in {(i, k) for (d, i, k) in pair_lookup}:
        lo, hi = min(i, k), max(i, k)
        for j in range(lo + 1, hi):
            expr = m * vars_list[col_v(e, j)]
            any_term = False
            for d in range(s):
                if (d, i, k) not in pair_lookup:
                    continue
                expr += vars_list[col_f_lambda(d, e, pair_lookup[(d, i, k)])]
                any_term = True
            if not any_term:
                continue
            model.addCons(expr <= m)

# EQ 4 
# entry flow (f_omega) from o toward destination d must equal demand E[o][d]
for o in range(s):
    for d in range(s):
        if d == o:
            continue
        model.addCons(vars_list[col_f_omega(o,d)]==E[o][d])

#EQ 5
#people exiting the elevator at their destination + people taking the stairs to their destination must equal total people with this floor as their destination
for d in range(s):
    exit_sum=0
    stair_sum=0
    for e in range(p):
        exit_sum+=vars_list[col_f_beta(d,e,d)]
    for i in range(s):
        if i == d:
            continue
        if (d, i, d) in pair_lookup:
            stair_sum+=vars_list[col_f_eta(d, pair_lookup[(d, i, d)])]
    E_sum = quicksum(E[o][d] for o in range(s) if o != d)
    model.addCons(exit_sum+stair_sum == E_sum)

# EQ 6: 
# f_lambda(in) + f_alpha(j) = f_lambda(out) + f_beta(j)
for d in range(s):
    for j in range(s):
        for e in range(p):
            left = 0
            right = 0

            # riders arriving at j on the elevator (from any previous stop i)
            for i in range(s):
                if i != j and (d, i, j) in pair_lookup:
                    left += vars_list[col_f_lambda(d, e, pair_lookup[(d, i, j)])]

            # riders boarding at j (nobody boards at their own destination)
            if j != d:
                left += vars_list[col_f_alpha(j, e, d)]

            # riders continuing to the next stop k
            for k in range(s):
                if k != j and (d, j, k) in pair_lookup:
                    right += vars_list[col_f_lambda(d, e, pair_lookup[(d, j, k)])]

            # riders getting off at j (includes j == d, the final exit)
            right += vars_list[col_f_beta(j, e, d)]

            model.addCons(left == right)

#EQ 7:
#people coming into a floor have to equal people coming out of a floor(for non destination floors)
for d in range(s):
    for j in range(s):
        if j == d:
            continue          # people leave the system at their destination

        in_sum = vars_list[col_f_omega(j, d)]   # people who start at j heading to d
        out_sum = 0

        for i in range(s):
            #IN FLOWS 
            if i != j and (d, i, j) in pair_lookup:
                pidx = pair_lookup[(d, i, j)]
                in_sum += vars_list[col_f_eta(d, pidx)]
                for e in range(p):
                    in_sum += vars_list[col_f_lambda(d, e, pidx)]

            #OUT FLOWS
            if i != j and (d, j, i) in pair_lookup:
                pidx = pair_lookup[(d, j, i)]
                out_sum += vars_list[col_f_eta(d, pidx)]
                for e in range(p):
                    out_sum += vars_list[col_f_lambda(d, e, pidx)]

        model.addCons(in_sum == out_sum)

#H equations(8,9,10)
# EQ 8
# H has to match the top floor that an elevator stops on 
for e in range(p):
    for (T, count), _ in h_pair_lookup.items():
        # h(e,T,count) <= v(e,T)
        model.addCons(vars_list[col_h(e, T, count)] <= vars_list[col_v(e, T)])

        # h(e,T,count) + sum_{j>T} v(e,j) <= 1   (aggregate the "no stops above T" rows)
        above = quicksum(vars_list[col_v(e, j)] for j in range(T + 1, s))
        model.addCons(vars_list[col_h(e, T, count)] + above <= 1)

# EQ 9
#count must match number of stops
for e in range(p):
    total_stops = quicksum(vars_list[col_v(e, i)] for i in range(s))
    for (T, count), _ in h_pair_lookup.items():
        h = vars_list[col_h(e, T, count)]
        # total_stops = count  =>  h = 1   (equivalently: h = 0  =>  total_stops != count)
        # two-sided big-M on |total_stops - count| <= (s) * (1 - h)
        model.addCons(total_stops - count <= s * (1 - h))
        model.addCons(count - total_stops <= s * (1 - h))

#EQ 10
# exactly one (T, count) is active per period
for e in range(p):
    model.addCons(quicksum(vars_list[col_h(e, T, c)] for (T, c) in h_pair_lookup) == 1)

# EQ 11
# lobby: every elevator must stop at floor 0
for e in range(p):
    model.addCons(vars_list[col_v(e, 0)]==1)

# EQ 12
# F_alpha/k <= n(e) < (F_alpha/k) + 1   (so n = ceil(F_alpha / k))
for e in range(p):
    total = quicksum(vars_list[col_f_alpha(0, e, d)] for d in range(1, s))
    n = total / CAPASITY_ELEVATOR
    model.addCons(vars_list[col_n(e)] >= n)
    model.addCons(vars_list[col_n(e)] <= n + 1- 1e-4)
#endregion

#region Objective function
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
    N=F_lambda_cost(i,e,direction)/CAPASITY_ELEVATOR
    return(N+1)*H*F_alpha_cost(i,e,direction)*MID_CONSTANT

#BOTTUM FLOOR 
base_sum = 0
for e in range(p):
    n = vars_list[col_n(e)]
    H = H_expr(e)
    k = CAPASITY_ELEVATOR
    base_sum += k * H * n * (n - 1) / 2
    base_sum += (F_alpha_cost(0, e, 'up') - (n - 1) * k) * H * n

#Mid floors 
mid_sum=0
for e in range(p):
    for i in range(1,s-1):
        mid_sum+=(G(i,e,'up')+G(i,e,'down'))
    mid_sum+=G(s-1,e,'down')
obj_expr = quicksum(c_obj[c] * vars_list[c] for c in range(total_vars) if c_obj[c] != 0)+base_sum+mid_sum
print(f"Initalizing took {time.perf_counter() - t_init_start:.2f} seconds") 
#endregion

#region SOLVING AND OUTPUT 
#==========================
t_opt_start = time.perf_counter() 
t = model.addVar(name="t", vtype="C", lb=None, ub=None)
model.addCons(t >= obj_expr)
model.hideOutput()
model.setObjective(t, "minimize")
model.optimize()

print(model.getStatus())
x_opt = np.array([model.getVal(v) for v in vars_list])


print(f"Optimization took {time.perf_counter() - t_opt_start:.2f} seconds") 

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

    # === STARTING FLOWS (f_omega) ===
    print("\n=== Starting flows (f_omega) ===")
    for d in range(s):
        for i in range(s):
            if i == d:
                continue
            val = x_opt[col_f_omega(i, d)]
            if val > 0.01:
                print(f"  f_omega[i={i}, d={d}] = {val:.2f}")

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
    k = CAPASITY_ELEVATOR

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

        # --- all-in cost of one more rider on each elevator hop vs stairs ---
    print("\n=== Marginal cost: elevator vs stairs (direct hops only) ===")
    print(f"{'elev':>4} {'hop':>7} {'wait':>7} {'ride':>6} {'elev_tot':>9} "
          f"{'stairs':>7} {'elev-stairs':>12} {'elev_flow':>10} {'stair_flow':>11}")

    def marginal_wait(i, j, e, H):
        direction = 'up' if j > i else 'down'
        F_a = num_alpha(i, e, direction)
        if i == 0:
            return bottom_wait(F_a + 1, H) - bottom_wait(F_a, H)
        F_l = num_lambda(i, e, direction)
        return mid_wait(F_a + 1, F_l + 1, H) - mid_wait(F_a, F_l, H)

    for e in range(p):
        H = get_elevator_headway(x_opt, e)[0] * WAIT_FACTOR
        served = [i for i in range(s) if x_opt[col_v(e, i)] > 0.5]
        for i in served:
            for j in served:
                if i == j:
                    continue
                lo, hi = min(i, j), max(i, j)
                if any(lo < mid < hi for mid in served):
                    continue  # not a direct hop for this elevator
                wait = marginal_wait(i, j, e, H)
                ride = C_lambda(i, j)
                elev_tot = wait + ride
                stairs = T_stairs[i][j]
                elev_flow = sum(x_opt[col_f_lambda(d, e, pair_lookup[(d, i, j)])]
                                for d in range(s) if (d, i, j) in pair_lookup)
                stair_flow = sum(x_opt[col_f_eta(d, pair_lookup[(d, i, j)])]
                                 for d in range(s) if (d, i, j) in pair_lookup)
                print(f"{e:>4} {f'{i}->{j}':>7} {wait:>7.2f} {ride:>6.2f} {elev_tot:>9.2f} "
                      f"{stairs:>7.2f} {elev_tot - stairs:>12.2f} {elev_flow:>10.2f} {stair_flow:>11.2f}")

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
#endregion


