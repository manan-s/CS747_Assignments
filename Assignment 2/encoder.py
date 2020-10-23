import os
import sys
import numpy as np

def parseargs():
    opts = [opt for opt in sys.argv[1:] if opt.startswith("--")]
    args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]

    try:
        grid_path = args[opts.index('--grid')]
    
    except ValueError:
        print("Atleast one of the arguments isn't provided")
        sys.exit(f"Usage1: {sys.argv[0]} (--grid) <arguments>...")
    
    except:
        print("Please check the arguments")
        sys.exit(f"Usage2: {sys.argv[0]} (--grid) <arguments>...")

    grid = np.loadtxt(grid_path)
    I,J = grid.shape
    num_ele = I*J

    #Mapping from state number to grid coordinates (and vice versa)
    s_to_coord={}
    coord_to_s={}
    
    num_states = num_ele - np.count_nonzero((grid==1))
    num_actions = 4   #0 - North, 1 - South, 2 - East, 3 - West
    print("numStates {}".format(num_states))
    print("numActions {}".format(num_actions))

    s = 0
    end = []
    discount = 0.9

    for i in range(1,I-1):
        for j in range(1,J-1):
            if grid[i][j] == 1:
                continue
                    
            elif grid[i,j]==2:
                start = s
                
            elif grid[i,j]==3:
                end.append(s)
                
            s_to_coord[s] = (i,j)
            coord_to_s[(i,j)] = s
            s+=1
    
    print("start {}".format(start))
    print("end {}".format(*end))
    
    R = np.zeros((num_states, num_actions, num_states), dtype=np.float128)
    T = np.zeros((num_states, num_actions, num_states), dtype=np.float128)
    
    bsr = -1.0  #reward for coming back to start
    slr = -1.0 #reward for self loop
    sr = 100000000.0   #success reward
    fr = -1.0    #failure reward

    s = 0

    for i in range(1, I - 1):
        
        for j in range(1, J - 1):
            
            if grid[i][j] == 1:
                continue
            else:
                if s in end:
                    s+=1
                    continue

                a = 0

                if grid[i-1][j] == 1:
                    T[s,a,s] = 1.0
                    R[s,a,s] = slr
                    print("transition {} {} {} {} {}".format(s,a,s,slr,1.0))

                elif grid[i-1][j] == 2:
                    T[s,a,coord_to_s[(i-1,j)]] = 1.0
                    R[s,a,coord_to_s[(i-1,j)]] = bsr
                    print("transition {} {} {} {} {}".format(s,a,coord_to_s[(i-1,j)],bsr,1.0))

                elif grid[i-1][j] == 3:
                    T[s,a,coord_to_s[(i-1,j)]] = 1.0
                    R[s,a,coord_to_s[(i-1,j)]] = sr
                    print("transition {} {} {} {} {}".format(s,a,coord_to_s[(i-1,j)],sr,1.0))
                
                else:
                    T[s,a,coord_to_s[(i-1,j)]] = 1.0
                    R[s,a,coord_to_s[(i-1,j)]] = fr
                    print("transition {} {} {} {} {}".format(s,a,coord_to_s[(i-1,j)],fr,1.0))

                a = 1

                if grid[i+1][j] == 1:
                    T[s,a,s] = 1.0
                    R[s,a,s] = slr
                    print("transition {} {} {} {} {}".format(s,a,s,slr,1.0))

                elif grid[i+1][j] == 2:
                    T[s,a,coord_to_s[(i+1,j)]] = 1.0
                    R[s,a,coord_to_s[(i+1,j)]] = bsr
                    print("transition {} {} {} {} {}".format(s,a,coord_to_s[(i+1,j)],bsr,1.0))

                elif grid[i+1][j] == 3:
                    T[s,a,coord_to_s[(i+1,j)]] = 1.0
                    R[s,a,coord_to_s[(i+1,j)]] = sr
                    print("transition {} {} {} {} {}".format(s,a,coord_to_s[(i+1,j)],sr,1.0))
                
                else:
                    T[s,a,coord_to_s[(i+1,j)]] = 1.0
                    R[s,a,coord_to_s[(i+1,j)]] = fr
                    print("transition {} {} {} {} {}".format(s,a,coord_to_s[(i+1,j)],fr,1.0))

                a = 2

                if grid[i][j+1] == 1:
                    T[s,a,s] = 1.0
                    R[s,a,s] = slr
                    print("transition {} {} {} {} {}".format(s,a,s,slr,1.0))

                elif grid[i][j+1] == 2:
                    T[s,a,coord_to_s[(i,j+1)]] = 1.0
                    R[s,a,coord_to_s[(i,j+1)]] = bsr
                    print("transition {} {} {} {} {}".format(s,a,coord_to_s[(i,j+1)],bsr,1.0))
                
                elif grid[i][j+1] == 3:
                    T[s,a,coord_to_s[(i,j+1)]] = 1.0
                    R[s,a,coord_to_s[(i,j+1)]] = sr
                    print("transition {} {} {} {} {}".format(s,a,coord_to_s[(i,j+1)],sr,1.0))
                
                else:
                    T[s,a,coord_to_s[(i,j+1)]] = 1.0
                    R[s,a,coord_to_s[(i,j+1)]] = fr
                    print("transition {} {} {} {} {}".format(s,a,coord_to_s[(i,j+1)],fr,1.0))

                a = 3

                if grid[i][j-1] == 1:
                    T[s,a,s] = 1.0
                    R[s,a,s] = slr
                    print("transition {} {} {} {} {}".format(s,a,s,slr,1.0))

                elif grid[i][j-1] == 2:
                    T[s,a,coord_to_s[(i,j-1)]] = 1.0
                    R[s,a,coord_to_s[(i,j-1)]] = bsr
                    print("transition {} {} {} {} {}".format(s,a,coord_to_s[(i,j-1)],bsr,1.0))
                
                elif grid[i][j-1] == 3:
                    T[s,a,coord_to_s[(i,j-1)]] = 1.0
                    R[s,a,coord_to_s[(i,j-1)]] = sr
                    print("transition {} {} {} {} {}".format(s,a,coord_to_s[(i,j-1)],sr,1.0))
                
                else:
                    T[s,a,coord_to_s[(i,j-1)]] = 1.0
                    R[s,a,coord_to_s[(i,j-1)]] = fr
                    print("transition {} {} {} {} {}".format(s,a,coord_to_s[(i,j-1)],fr,1.0))

                s+=1

    print("mdtype episodic")
    print("discount {}".format(discount))
    

parseargs()
