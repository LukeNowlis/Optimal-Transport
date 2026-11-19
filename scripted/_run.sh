#!/bin/bash

for i in {6..6}; do
    # Calculate 1 / (2^i) with 10 decimal places of precision
    param=$(echo "scale=10; 1 / (2^$i)" | bc -l)
    
    echo "Running with parameter: $param (i=$i)"
    python3 Gurobi_Elevator.py "$param"
done
