def joint_headway():
    'SIMPLIFICATIONS SO FAR'
    'no transferes'
    'no eq. 20'

    #GENERATES INFROMATION ABOUT C_R, RETURNS C_R, PERCEIVED HEADWAY AND C 
    def generate_combos(headway, pattern_num):
        combos = [()]
        for _ in range(pattern_num):
            new_combos = []
            for combo in combos:
                for h in headway:
                    new_combos.append(combo + (h,))
            combos = new_combos 
        valid_combos = [c for c in combos if any(h != 0 for h in c)]
        perceived_headway=np.zeros(len(valid_combos))
        for idx, combo in enumerate(valid_combos):
            total_frequency = 0
            for h in combo:
                if h != 0:
                    total_frequency += 1 / h
            perceived_headway[idx] = 1 / total_frequency
        c=len(valid_combos)
        return valid_combos, perceived_headway, c

    #INPUTS
    #===========
    NUMBER_OF_STOPS=3
    NUMBER_OF_PATTERNS=2
    headway=[0,5,10]
    CAPASITY_PER_TRAIN=10000 #B_r
    NUMBER_OF_VECHICLES=1000 #N_r
    WAITING_COEFFICENT=1
    E = np.array([ #This is the input data telling us real world demand 
    [0, 0,10,0,0, 0],  
    [0, 0,0,0,0,  0],
    [0, 0,0,0,0,  0],
    [0,0,0,0, 10,  0], 
    [0, 0,0,0,0,  0],
    [0, 0,0,0,0,  0], 
    ])

    #INITALIZE NUMBERS THAT WE NEED
    #==============================
    c_r,perceived_headway,c=generate_combos(headway,NUMBER_OF_PATTERNS)# combos of headway and pattern, perceived headway for each combo, number of combos 
    s=NUMBER_OF_STOPS*2 #number of stops(going both ways)
    p=NUMBER_OF_PATTERNS #number of patterns
    h=len(headway) #number of headway options
    m=1*10**6 #A sufficently large number
    i_j_pairs = sum(1 for i in range(s) if i not in {0, s-1} #number of valid i,j pairs acoring to papers constraints
                for j in range(i+1, s))
    T_travel = np.zeros((s, s)) #Cost matrix of time it takes to get from one stop i to stop j, this can be changed 
    for i in range(s):
        for j in range(s):
            T_travel[i][j] = abs(i - j)
    
    active_combos_count = sum(# Count combinations with at least 2 active patterns (for Eq. 28)
        1 for combo in c_r
        if sum(1 for h in combo if h != 0) >= 2
    )
    pattern_pairs = p * (p - 1) // 2    # Number of (p1, p2) pairs with p1 < p2

    #FIND NUMBER OF ALL VARIABLES
    #=============================
    f_alpha_num=s*(s-2)*c*p
    f_beta_num=s*p
    f_lambda_num=s*p*i_j_pairs
    f_omega_num=s*(s-2)*c
    x_num=s*(s-1)*p
    x_eta_num=s*(s-1)*p*h
    y_num=p*h
    z_num=s*(s-2)*c

    #INDEXING
    #===========
    # Column index mapping for each type of variable 
    idx = {}
    col = 0

    idx['f_omega']  = (col, col + f_omega_num);  col += f_omega_num
    idx['f_alpha']  = (col, col + f_alpha_num);  col += f_alpha_num
    idx['f_lambda'] = (col, col + f_lambda_num); col += f_lambda_num
    idx['f_beta']   = (col, col + f_beta_num);   col += f_beta_num

    idx['x']        = (col, col + x_num);        col += x_num
    idx['x_eta']    = (col, col + x_eta_num);    col += x_eta_num
    idx['y']        = (col, col + y_num);        col += y_num
    idx['z']        = (col, col + z_num);        col += z_num

    total_vars = col

    #FUNTIONS THAT GIVE US THE INDEX OF EACH SPECIFIC ENTRY 
    #======================================================
    #each input to the function is multiplied by the number of other inputs before that one changes again
    def col_f_alpha(d, i, c_idx, p_idx):
        return idx['f_alpha'][0] + d*(s-2)*c*p + i*c*p + c_idx*p + p_idx
    def col_f_omega(d, i, c_idx):
        return idx['f_omega'][0] + d*(s-2)*c + i*c + c_idx
    def col_f_lambda(d, p_idx, pair_idx):
        return idx['f_lambda'][0] + d*i_j_pairs*p + p_idx*i_j_pairs + pair_idx
    def col_f_beta(j, p_idx):
        return idx['f_beta'][0] + j*p + p_idx
    def col_x(p_idx, i, j):
        return idx['x'][0] + p_idx*s*(s-1) + i*(s-1) + (j if j < i else j-1)
    def col_x_eta(p_idx, i, j, h_idx):
        return idx['x_eta'][0] + p_idx*s*(s-1)*h + i*(s-1)*h + (j if j < i else j-1)*h + h_idx
    def col_y(p_idx, h_idx):
        return idx['y'][0] + p_idx*h + h_idx

    # Build z_pair_lookup BEFORE defining col_z
    z_pair_lookup = {}
    for i in range(s):
        forbidden = {i, s - 1 - i}
        z_idx = 0
        for d in range(s):
            if d in forbidden:
                continue
            z_pair_lookup[(i, d)] = z_idx
            z_idx += 1

    def col_z(i, d, c_idx):
        return idx['z'][0] + i*(s-2)*c + z_pair_lookup[(i, d)]*c + c_idx

    #FUNCTIONS FOR NORTH SOUTH 
    #=========================
    def phys_to_north(phys):
        """Map physical stop index to its northbound node."""
        return phys
    def phys_to_south(phys):
        """Map physical stop index to its southbound node."""
        return s - 1 - phys
    def node_to_phys(node):
        """Map a direction-node to its physical stop."""
        if node < s // 2:
            return node
        else:
            return s - 1 - node

    #OPTIMIZATION SET UP
    #===================
    cont = sum([f_alpha_num, f_beta_num, f_lambda_num, f_omega_num])
    bi   = sum([x_num, x_eta_num, y_num, z_num])
    var_num=cont+bi
    x_var=[0]*cont+[1]*bi
    l=[0]*(cont+bi)
    u=[m]*cont+[1]*bi
    # Objective function coefficients (length = var_num)
    c_obj = np.zeros(var_num)
    # Term 1: Riding time — T_ij * f_lambda(d, p, i, j)
    for d in range(s):
        forbidden = {d, s - 1 - d}
        pair_idx = 0
        for i in range(s):
            if i in forbidden:
                continue
            for j in range(i + 1, s):
                for p_idx in range(p):
                    c_obj[col_f_lambda(d, p_idx, pair_idx)] += T_travel[i][j]
                pair_idx += 1
    # Term 2: Waiting time — coefficent * (T_c / 2) * f_omega(d, i, c)
    for d in range(s):
        forbidden = {d, s - 1 - d}
        for i in range(s):
            if i in forbidden:
                continue
            for c_idx in range(c):
                c_obj[col_f_omega(d, i, c_idx)] += WAITING_COEFFICENT * perceived_headway[c_idx] / 2

    #MAKING A
    #=========
    def make_a():

        eq26_per_od = sum(sum(1 for h in combo if h != 0) for combo in c_r)
        a_len = (
        s * p +                                  # Eq. 15
        s * p +                                  # Eq. 16
        f_lambda_num +                           # Eq. 17
        p +                                      # Eq. 19
        x_eta_num +                              # Eq. 21
        x_num +                                  # Eq. 22
        s * (s - 1) // 2 * p +                   # Eq. 23
        1 +                                      # Eq. 24 (single row)
        s * (s - 2) +                            # Eq. 25
        s*(s-2)*eq26_per_od +                    #Eq.26
        f_alpha_num +                            # Eq. 27
        2 * s * (s - 2) * active_combos_count * pattern_pairs +  # Eq. 28 (2 rows per instance)
        s * (s - 2) +                            # Eq. 29
        s +                                      # Eq. 30
        s * (s - 2) * c +                        # Eq. 31
        s * (s - 2) * p +                        # Eq. 32
        s * p                                    # Eq. 33
    )
        b_l=np.zeros(a_len)
        b_u=np.zeros(a_len)
        A = np.zeros((a_len, var_num))
        show=False 
        row=0
        # Eq. (15): 
        # For every stop s and every pattern p there needs to be the same ammont of trains coming in as there are leaving. Constraint on the x
        row_before = row
        for p_idx in range(p):
            for j in range(s):
                for i in range(s):
                    if i != j:
                        A[row, col_x(p_idx, i, j)] += 1
                for k in range(s):
                    if k != j:
                        A[row, col_x(p_idx, j, k)] -= 1
                b_l[row] = 0
                b_u[row] = 0
                row += 1
        if show:
            print(f"Eq. 15: {row - row_before} rows | expected {s*p}")

        # Eq. (16): 
        # For every p and s there can only be 0 or 1 train coming into that stop s constraint on x
        row_before = row
        for p_idx in range(p):
            for j in range(s):
                for i in range(s):
                    if i != j:
                        A[row, col_x(p_idx, i, j)] += 1
                b_l[row] = 0
                b_u[row] = 1
                row += 1
        if show:
            print(f"Eq. 16: {row - row_before} rows | expected {s*p}")

        # Eq. (17): 
        # the flow of passengers from i to j with pattern p must be less than or equal to M*x i.e. it will be zero if x is 0 and it can be at most M if x is one 
        row_before = row
        for d in range(s):
            forbidden = {d, s - 1 - d}
            pair_idx = 0
            for i in range(s):
                if i in forbidden:
                    continue
                for j in range(i + 1, s):
                    for p_idx in range(p):
                        A[row, col_f_lambda(d, p_idx, pair_idx)] += 1
                        A[row, col_x(p_idx, i, j)] -= m
                        b_l[row] = -np.inf
                        b_u[row] = 0
                        row += 1
                    pair_idx += 1
        if show:
            print(f"Eq. 17: {row - row_before} rows | expected {f_lambda_num}")

        # Eq. (19): Each pattern gets exactly one headway
        row_before = row
        for p_idx in range(p):
            for h_idx in range(h):
                A[row, col_y(p_idx, h_idx)] += 1
            b_l[row] = 1
            b_u[row] = 1
            row += 1
        if show:
            print(f"Eq. 19: {row - row_before} rows | expected {p}")
    
        # Eq. (21): 
        # x_eta <= y, you can only have a link on pattern p with headway h if that headway is assigned to pattern p by y 
        row_before = row
        for p_idx in range(p):
            for i in range(s):
                for j in range(s):
                    if i == j:
                        continue
                    for h_idx in range(h):
                        A[row, col_x_eta(p_idx, i, j, h_idx)] += 1
                        A[row, col_y(p_idx, h_idx)] -= 1
                        b_l[row] = -np.inf
                        b_u[row] = 0
                        row += 1
        if show:
            print(f"Eq. 21: {row - row_before} rows | expected {x_eta_num}")

        # Eq. (22): 
        # sum_h x_eta = x, for each i,j,p stop connection there can only be one headway. Hence the sum of x_eta should be the same as the x 
        row_before = row
        for p_idx in range(p):
            for i in range(s):
                for j in range(s):
                    if i == j:
                        continue
                    for h_idx in range(h):
                        A[row, col_x_eta(p_idx, i, j, h_idx)] += 1
                    A[row, col_x(p_idx, i, j)] -= 1
                    b_l[row] = 0
                    b_u[row] = 0
                    row += 1
        if show:
            print(f"Eq. 22: {row - row_before} rows | expected {x_num}")

        # Eq. (23):
        #The number of passengers on a link x_eta ie the f varible for this cannot exed the vechicle capasity * the frequency of trains for that time period for every pattern
        row_before = row
        for p_idx in range(p):
            for i in range(s):
                for j in range(i + 1, s):
                    # Add flow terms: sum_d f_lambda(d, p, i, j)
                    for d in range(s):
                        forbidden = {d, s - 1 - d}
                        if i in forbidden:
                            continue
                        # Recompute pair_idx for (d, i, j)
                        pair_idx = 0
                        for ii in range(s):
                            if ii in forbidden:
                                continue
                            for jj in range(ii + 1, s):
                                if ii == i and jj == j:
                                    break
                                pair_idx += 1
                            else:
                                continue
                            break
                        A[row, col_f_lambda(d, p_idx, pair_idx)] += 1
                    # Add capacity terms: -B/T * x_eta(p, i, j, h)
                    for h_idx in range(h):
                        if headway[h_idx] != 0:
                            A[row, col_x_eta(p_idx, i, j, h_idx)] -= CAPASITY_PER_TRAIN / headway[h_idx]
                    b_l[row] = -np.inf
                    b_u[row] = 0
                    row += 1
        if show:
            print(f"Eq. 23: {row - row_before} rows | expected {s * (s - 1) // 2 * p}")

        # Eq. (24): 
        # the fleet needed to run operate a strategy can be found by suming the time it takes for a pattern to complete it's loop and multiplying this by the headway it runs at for every pattern, this number has to be less than the availbe fleet
        # This one I dont understand super well imma be honest 
        row_before = row
        for p_idx in range(p):
            for i in range(s):
                for j in range(s):
                    if i == j:
                        continue
                    for h_idx in range(h):
                        if headway[h_idx] != 0:
                            # T_ij^lambda * (1 / T_h^gamma) * x_eta(p, i, j, h)
                            A[row, col_x_eta(p_idx, i, j, h_idx)] += T_travel[i][j] / headway[h_idx]
        b_l[row] = -np.inf
        b_u[row] = NUMBER_OF_VECHICLES
        row += 1
        if show:
            print(f"Eq. 24: {row - row_before} rows | expected 1")

        # Eq. (25): 
        # For each O-D pair theie can only be one z value, meaning there can only be one perceived headway. Constraint on z 
        row_before = row
        for d in range(s):
            forbidden = {d, s - 1 - d}
            for i in range(s):
                if i in forbidden:
                    continue
                for c_idx in range(c):
                    A[row, col_z(i, d, c_idx)] += 1
                b_l[row] = 0
                b_u[row] = 1
                row += 1
        if show:
            print(f"Eq. 25: {row - row_before} rows | expected {s*(s-2)}")

        # Eq. (26): 
        # z_idc <= y_ph for all patterns in combination c, meaning we can only use the perceived headway if all the headways are actually being used 
        row_before = row
        for d in range(s):
            forbidden = {d, s - 1 - d}
            for i in range(s):
                if i in forbidden:
                    continue
                for c_idx, combo in enumerate(c_r):
                    for p_idx, h_idx in enumerate(combo):
                        if h_idx == 0:
                            continue  # Pattern is out of service in this combo; skip
                        A[row, col_z(i, d, c_idx)] += 1
                        A[row, col_y(p_idx, h_idx)] -= 1
                        b_l[row] = -np.inf
                        b_u[row] = 0
                        row += 1
        if show:
            print(f"Eq. 26: {row - row_before} rows | expected {s*(s-2)*eq26_per_od}")

        # Eq. (27): 
        # f_alpha <= M * z_idc, Boarding can only happen when combo c is assigned to that O-D pair, also it is constrained by (M in the paper) but I will use VECHCAL CAPSITY
        row_before = row
        for d in range(s):
            forbidden = {d, s - 1 - d}
            for i in range(s):
                if i in forbidden:
                    continue
                for c_idx in range(c):
                    for p_idx in range(p):
                        A[row, col_f_alpha(d, i, c_idx, p_idx)] += 1
                        A[row, col_z(i, d, c_idx)] -= m
                        b_l[row] = -np.inf
                        b_u[row] = 0
                        row += 1
        if show:
            print(f"Eq. 27: {row - row_before} rows | expected {f_alpha_num}")

        # Eq. (28): 
        # Frequency share rule
        # also not sure abot this one  
        row_before = row
        for d in range(s):
            forbidden = {d, s - 1 - d}
            for i in range(s):
                if i in forbidden:
                    continue
                for c_idx, combo in enumerate(c_r):
                    for p1 in range(p):
                        for p2 in range(p1 + 1, p):
                            h1 = combo[p1]
                            h2 = combo[p2]
                            if h1 == 0 or h2 == 0:
                                continue
                            
                            # Left inequality: -M <= T_h1*f_p1 - T_h2*f_p2 - M*z
                            A[row, col_f_alpha(d, i, c_idx, p1)] += h1
                            A[row, col_f_alpha(d, i, c_idx, p2)] -= h2
                            A[row, col_z(i, d, c_idx)] -= m
                            b_l[row] = -m
                            b_u[row] = np.inf
                            row += 1
                            
                            # Right inequality: T_h1*f_p1 - T_h2*f_p2 + M*z <= M
                            A[row, col_f_alpha(d, i, c_idx, p1)] += h1
                            A[row, col_f_alpha(d, i, c_idx, p2)] -= h2
                            A[row, col_z(i, d, c_idx)] += m
                            b_l[row] = -np.inf
                            b_u[row] = m
                            row += 1
        if show:
            print(f"Eq. 28: {row - row_before} rows | expected {2*s*(s-2)*active_combos_count*pattern_pairs}")

        # Eq. (29): 
        # the sum of people entering a stop going to a destination d must equal the demand for that O-D pair. We need to check both 'stops' as in the one runnig north and the one running south
        row_before = row
        for o in range(s):
            for d in range(s):
                if d == o or d == s - 1 - o:
                    continue
                for c_idx in range(c):
                    A[row, col_f_omega(d, o, c_idx)] += 1
                b_l[row] = E[o][d]
                b_u[row] = E[o][d]
                row += 1
        if show:
            print(f"Eq. 29: {row - row_before} rows | expected {s*(s-2)}")

        # Eq. (30): 
        # Same thing as 29 but for exit flows so that no one stays on the trains forever
        row_before = row
        for d in range(s):
            for p_idx in range(p):
                A[row, col_f_beta(d, p_idx)] += 1
            total_demand = sum(E[o][d] for o in range(s) if o != d and o != s - 1 - o)
            b_l[row] = total_demand
            b_u[row] = total_demand
            row += 1
        if show:
            print(f"Eq. 30: {row - row_before} rows | expected {s}")

        # Eq. (31): 
        # Entry = Boarding (simplified, no transfers), every passenger who enters a stop must board a pattern 
        row_before = row
        for d in range(s):
            forbidden = {d, s - 1 - d}
            for i in range(s):
                if i in forbidden:
                    continue
                for c_idx in range(c):
                    A[row, col_f_omega(d, i, c_idx)] += 1
                    for p_idx in range(p):
                        A[row, col_f_alpha(d, i, c_idx, p_idx)] -= 1
                    b_l[row] = 0
                    b_u[row] = 0
                    row += 1
        if show:
            print(f"Eq. 31: {row - row_before} rows | expected {s*(s-2)*c}")

        # Eq. (32):
        # Flow balance at non-destination stops (simplified, no transfers)
        # For stops along peoples route people who arrived there plus people who board must be equal to people who continue to the next stop  
        row_before = row
        for d in range(s):
            forbidden = {d, s - 1 - d}
            for j in range(s):
                if j in forbidden:
                    continue
                for p_idx in range(p):
                    # Boarding flows at j
                    for c_idx in range(c):
                        A[row, col_f_alpha(d, j, c_idx, p_idx)] += 1
                    # Arriving flows: from i < j to j
                    for i in range(j):
                        if i in forbidden:
                            continue
                        # Recompute pair_idx for (d, i, j)
                        pair_idx = 0
                        found = False
                        for ii in range(s):
                            if ii in forbidden:
                                continue
                            for jj in range(ii + 1, s):
                                if ii == i and jj == j:
                                    found = True
                                    break
                                pair_idx += 1
                            if found:
                                break
                        if found:
                            A[row, col_f_lambda(d, p_idx, pair_idx)] += 1
                    # Continuing flows: from j to k > j
                    for k in range(j + 1, s):
                        if j in forbidden:
                            continue
                        # Recompute pair_idx for (d, j, k)
                        pair_idx = 0
                        found = False
                        for ii in range(s):
                            if ii in forbidden:
                                continue
                            for jj in range(ii + 1, s):
                                if ii == j and jj == k:
                                    found = True
                                    break
                                pair_idx += 1
                            if found:
                                break
                        if found:
                            A[row, col_f_lambda(d, p_idx, pair_idx)] -= 1
                    b_l[row] = 0
                    b_u[row] = 0
                    row += 1
        if show:
            print(f"Eq. 32: {row - row_before} rows | expected {s*(s-2)*p}")

        # Eq. (33): 
        # Arriving = Exiting at destination stops, people who arive at their destination need to get off 
        row_before = row
        for d in range(s):
            forbidden = {d, s - 1 - d}
            d_tilde = s - 1 - d
            for p_idx in range(p):
                # The destination stop j can be either d or d_tilde
                for j in [d]:
                    # Sum of arriving flows: from i < j to j
                    for i in range(j):
                        if i in forbidden:
                            continue
                        # Recompute pair_idx for (d, i, j)
                        pair_idx = 0
                        found = False
                        for ii in range(s):
                            if ii in forbidden:
                                continue
                            for jj in range(ii + 1, s):
                                if ii == i and jj == j:
                                    found = True
                                    break
                                pair_idx += 1
                            if found:
                                break
                        if found:
                            A[row, col_f_lambda(d, p_idx, pair_idx)] += 1
                    # Exiting flow at j
                    A[row, col_f_beta(j, p_idx)] -= 1
                    b_l[row] = 0
                    b_u[row] = 0
                    row += 1
        if show:
            print(f"Eq. 33: {row - row_before} rows | expected {s*p}")

        #print(f"\nTotal rows: {row} | a_len: {a_len}")
        # assert row == a_len, f"Mismatch: {row} vs {a_len}"
        return A, b_l, b_u,idx
    A,b_l,b_u,idx=make_a()
    for i in range(len(b_l)):
        if b_l[i] > b_u[i]:
            print(f"Contradiction at row {i}: b_l={b_l[i]} > b_u={b_u[i]}")

    #SOLVE
    #============
    options={'presolve': False}
    constraints = LinearConstraint(A, b_l, b_u)
    bounds = Bounds(l, u)

    result = milp(
        c=c_obj,
        constraints=constraints,
        integrality=x_var,
        bounds=bounds,
    )

    #RESULTS
    #==========
    x_opt = result.x
    print(result.success,result.message)
    print(result.fun)

    # print("f_omega sum:", np.sum(x_opt[idx['f_omega'][0]:idx['f_omega'][1]]))
    # print("f_alpha sum:", np.sum(x_opt[idx['f_alpha'][0]:idx['f_alpha'][1]]))
    # print("f_lambda sum:", np.sum(x_opt[idx['f_lambda'][0]:idx['f_lambda'][1]]))
    # print("f_beta sum:", np.sum(x_opt[idx['f_beta'][0]:idx['f_beta'][1]]))

    # # Check binaries
    # print("x sum:", np.sum(x_opt[idx['x'][0]:idx['x'][1]]))
    # print("x_eta sum:", np.sum(x_opt[idx['x_eta'][0]:idx['x_eta'][1]]))
    # print("y sum:", np.sum(x_opt[idx['y'][0]:idx['y'][1]]))
    # print("z sum:", np.sum(x_opt[idx['z'][0]:idx['z'][1]]))


    # === HEADWAY ASSIGNMENTS (y) ===
    print("\n=== Headway assignments ===")
    for p_idx in range(p):
        for h_idx in range(h):
            val = x_opt[col_y(p_idx, h_idx)]
            if val > 0.5:
                print(f"  Pattern {p_idx}: headway = {headway[h_idx]} min")

    # === PATTERN STOP SEQUENCES (x) ===
    print("\n=== Pattern stop sequences ===")
    for p_idx in range(p):
        links = []
        for i in range(s):
            for j in range(s):
                if i == j:
                    continue
                val = x_opt[col_x(p_idx, i, j)]
                if val > 0.5:
                    links.append((i, j))
        print(f"  Pattern {p_idx}: links = {links}")

    # === COMBINATIONS ASSIGNED TO O-D PAIRS (z) ===
    print("\n=== Combinations assigned to O-D pairs ===")
    for i in range(s):
        for d in range(s):
            if d == i or d == s - 1 - i:
                continue
            for c_idx in range(c):
                val = x_opt[col_z(i, d, c_idx)]
                if val > 0.5:
                    print(f"  O-D ({i} -> {d}): combo {c_idx} = {c_r[c_idx]}")
                    print(f"    Perceived headway = {perceived_headway[c_idx]:.2f} min")

    # === FLOWS ===
    print("\n=== Entry flows (f_omega) ===")
    for d in range(s):
        for i in range(s):
            if d == i or d == s - 1 - i:
                continue
            for c_idx in range(c):
                val = x_opt[col_f_omega(d, i, c_idx)]
                if val > 0.01:
                    print(f"  f_omega[d={d}, i={i}, c={c_idx}] = {val:.2f}")

    print("\n=== Boarding flows (f_alpha) ===")
    for d in range(s):
        for i in range(s):
            if d == i or d == s - 1 - i:
                continue
            for c_idx in range(c):
                for p_idx in range(p):
                    val = x_opt[col_f_alpha(d, i, c_idx, p_idx)]
                    if val > 0.01:
                        print(f"  f_alpha[d={d}, i={i}, c={c_idx}, p={p_idx}] = {val:.2f}")

    print("\n=== Inter-stop flows (f_lambda) ===")
    for d in range(s):
        forbidden = {d, s - 1 - d}
        pair_idx = 0
        for i in range(s):
            if i in forbidden:
                continue
            for j in range(i + 1, s):
                for p_idx in range(p):
                    val = x_opt[col_f_lambda(d, p_idx, pair_idx)]
                    if val > 0.01:
                        print(f"  f_lambda[d={d}, p={p_idx}, i={i}, j={j}] = {val:.2f}")
                pair_idx += 1

    print("\n=== Exit flows (f_beta) ===")
    for j in range(s):
        for p_idx in range(p):
            val = x_opt[col_f_beta(j, p_idx)]
            if val > 0.01:
                print(f"  f_beta[j={j}, p={p_idx}] = {val:.2f}")
joint_headway()
