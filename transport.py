import numpy as np
import ot
import random 

def one_dimention():
    m_plus=[0,1,2,3]
    m_minus=[0,1,2,3]

    f_plus=[0.3,0.4,0.5,0]
    f_minus=[0.4,0.3,0.3,0.2]

    def find_cost(m_plus: list[int], m_minus: list[int]):
        cost = np.zeros((len(m_plus), len(m_minus)))
        for i in m_plus:
            for j in m_minus:
                cost[i,j]=(m_plus[i]-m_minus[j])**2
        print(cost)
        return cost

    def kant(f_plus: list[float], f_minus: list[float],cost):
        gamma = ot.emd(f_plus, f_minus, cost)
        print(gamma)


    cost=find_cost(m_plus, m_minus)
    kant(f_plus, f_minus,cost)

def two_dimentions(grid_size):
    def find_cost(m_plus: np.array[[int],[int]], m_minus: np.array[[int],[int]]):
        cost=np.zeros((grid_size**2,grid_size**2))
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    for l in range(grid_size):
                        (a,b)=m_plus[i,j]
                        (c,d)=m_minus[k,l]
                        cost[grid_size*(i)+j,grid_size*(k)+l]=abs(a-c)+abs(b-d)
        cost=cost+np.eye(grid_size**2)*10000
        print(cost)
        return(cost)
    def kant(f_plus: list[float], f_minus: list[float],cost):
        gamma = ot.emd(f_plus, f_minus, cost)
        print(gamma)
        

    # Create a 3x3 grid of coordinates
    x = np.arange(grid_size)
    y = np.arange(grid_size)
    xx, yy = np.meshgrid(x, y)
    m_plus = np.stack([xx, yy], axis=-1)
    m_minus=m_plus.copy()

    # Create two 3x3 matrices with random integers (0-5)
    f_plus = np.random.randint(0, 6, size=(grid_size, grid_size))
    f_minus = np.random.randint(0, 6, size=(grid_size, grid_size))

    # Make sure they have the same total sum
    sum_plus = np.sum(f_plus)
    sum_minus = np.sum(f_minus)

    if sum_minus != 0:
        # Scale f_minus to match f_plus sum
        f_minus = (f_minus / sum_minus * sum_plus).astype(int)
        
        # Fix rounding errors by adding/subtracting the difference
        diff = sum_plus - np.sum(f_minus)
        if diff != 0:
            # Add the difference to the first element
            f_minus[0, 0] += diff
    else:
        # If f_minus is all zeros, just copy f_plus
        f_minus = f_plus.copy()

    # Print the matrices
    print("f_plus:\n", f_plus)
    print("f_minus:\n", f_minus)
    print("Sum f_plus:", np.sum(f_plus))
    print("Sum f_minus:", np.sum(f_minus))

    # Flatten for ot.emd
    f_plus_flat = f_plus.flatten()
    f_minus_flat = f_minus.flatten()

    cost=find_cost(m_plus, m_minus)
    kant(f_plus_flat, f_minus_flat,cost)

two_dimentions(3)