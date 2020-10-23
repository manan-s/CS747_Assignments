import os
import sys
import numpy as np

def parseargs():
    opts = [opt for opt in sys.argv[1:] if opt.startswith("--")]
    args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]

    try:
        grid_path = args[opts.index('--grid')]
        value_file = args[opts.index('--value_policy')]
    
    except ValueError:
        print("Atleast one of the arguments isn't provided")
        sys.exit(f"Usage1: {sys.argv[0]} (--grid | --value_policy) <arguments>...")
    
    except:
        print("Please check the arguments")
        sys.exit(f"Usage2: {sys.argv[0]} (--grid | --value_policy) <arguments>...")
    
    grid = np.loadtxt(grid_path)
    file_obj = open(value_file, "r")
    file_lines = file_obj.readlines()

    I,J = grid.shape

    s_to_coord={}
    coord_to_s={}
    s = 0
    end = []

    for i in range(1, I - 1):
        for j in range(1, J - 1):
            if grid[i][j] == 1:
                continue
                    
            elif grid[i,j]==2:
                start = s
                
            elif grid[i,j]==3:
                end.append(s)
                
            s_to_coord[s] = (i,j)
            coord_to_s[(i,j)] = s
            s+=1
    
    policy = []
    answer = []
    for i in range(len(file_lines)):
        policy.append(int((file_lines[i].split())[-1]))
    
    curr_state = start 

    count = 0
    
    while curr_state not in end:
    
        action = policy[curr_state]
        
        if action == 0:
            answer.append("N")
            (i,j) = s_to_coord[curr_state]
            curr_state = coord_to_s[(i-1, j)]
        
        elif action == 1:
            answer.append("S")
            (i,j) = s_to_coord[curr_state]
            curr_state = coord_to_s[(i+1, j)]
        
        elif action == 2:
            answer.append("E")
            (i,j) = s_to_coord[curr_state]
            curr_state = coord_to_s[(i, j+1)]
        
        elif action == 3:
            answer.append("W")
            (i,j) = s_to_coord[curr_state]
            curr_state = coord_to_s[(i, j-1)]
        count+=1

    print(*answer)



parseargs()