# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 14:03:11 2021

@author: deoxy
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import time


### Function to create interaction matrix ###
# Example: firm j has a policy of g = {0  1  1  0  1}
#                                      g1 g2 g3 g4 g5
#          For g1, the function randomly samples K-1 and E number of sub-searching and shaping policy using uniform distribution
#          like g1 is related to g3 and g5 (when K = 3), and e1 and e5 (when E = 2)
def interaction_matrix(N,K,E,Z):
    
    inter_mat = {}
    
    for g in range(N):
    
        g_interact = np.random.choice(np.delete(np.arange(0,N),g), replace = False, size = K-1)
        
        e_interact = np.random.choice(np.arange(0,Z), replace = False, size = E)
        
        inter_mat[g] = [[g_interact],[e_interact]]
    
    return inter_mat


### Function to calculate the scores ###
def scoring(N, K, E, Z, g_arr,e_arr, fitness_landscape, g_e_relationships):
    
    f_table = g_arr.copy()
        
    e_arr = pd.DataFrame(e_arr).transpose().copy()
    
    score_compilation = []
    
    type_col = f_table.pop("Type").to_frame()
    
    
    if "Score" in f_table.columns:
        
            f_table = f_table.drop(columns = ["Score"])

    
    for row in range(f_table.shape[0]):
        
        g_arr = f_table.iloc[row,:]
    
        g_arr = g_arr.to_frame().transpose().copy().reset_index(drop = True)
        
        score = np.array([])
                
        for i in range(N):
            
            eval_str = np.array([g_arr.iloc[0,i]])
            
            g_arr_red = g_arr.drop(columns = [i])
            
            select_g = g_e_relationships[i][0][0]
            
            g_arr_red_sel = np.array(g_arr_red.loc[0,select_g])
            
            eval_str = np.concatenate((eval_str,g_arr_red_sel))
            
            select_e = g_e_relationships[i][1][0]
            
            e_arr_sel = np.array(e_arr.loc[0,list(select_e)])
            
            eval_str = np.concatenate((eval_str,e_arr_sel))
            
            row = np.sum(eval_str[::-1]*2**np.arange(len(eval_str)))
            
            score = np.concatenate((score,np.array([fitness_landscape[row,i]])))
    
        score_compilation += [np.mean(score)]
    
    f_table = f_table.join(type_col)

    f_table["Score"] = score_compilation
        

    return f_table


### NKEZ Algorithm ###
def Standard(firm_count = 10, total_iteration = 50, runs = 1, 
         N = 12, K = 11, E = 12, Z = 12, 
         shaper_proportion = 0.5, plot = False, random_seeds = []):
    
    K = K+1
    
    if len(random_seeds) == 0:
        
        pass
    
    elif len(random_seeds) != runs:
        
        raise Exception("The number of random seeds must be equal to the number of runs.")
    
    master_avg_fit_searchers = {}
    master_avg_fit_shapers = {}
    
    total_start = time.time()
    
    for run in range(runs):
        
        g_e_relationships = interaction_matrix(N,K,E,Z)
        
        start = time.time()
        
        if len(random_seeds) != 0:
            
            np.random.seed(random_seeds[run])
        
        else:
            
            np.random.seed(run)
        
        fitness_landscape = np.random.uniform(0,1,size = (2**(K+E))*N).reshape(2**(K+E),N)
        
        search_policies = np.random.randint(0,2, size = N*firm_count).reshape(firm_count,N)
        shape_policy = np.random.randint(0,2,size = Z).reshape(1,Z)[0]
        
        shaper_index = list(np.random.choice(np.arange(0,firm_count), replace = False, size = int(shaper_proportion*firm_count)))
        
        searcher_search_policies = pd.DataFrame(np.delete(search_policies,shaper_index, axis = 0))
        searcher_search_policies["Type"] = "Searcher"
        
        shaper_search_policies = pd.DataFrame(search_policies[shaper_index,:])
        shaper_search_policies["Type"] = "Shaper"
        
        firm_table = searcher_search_policies.merge(shaper_search_policies, how = 'outer')
        
        firm_table = scoring(N,K,E,Z,firm_table,shape_policy,fitness_landscape, g_e_relationships)
        
        avg_fit_searchers = [firm_table[firm_table["Type"] == "Searcher"]["Score"].mean()]
        avg_fit_shapers = [firm_table[firm_table["Type"] == "Shaper"]["Score"].mean()]
        
        
        for iteration in range(total_iteration):
            
            start_iter = time.time()
            
            order = np.random.permutation(np.arange(firm_count))
            
            ft_copy = firm_table.copy()
            
            ft_copy = ft_copy.iloc[order,:]
            
            for firm in ft_copy.index:

                policies_0 = ft_copy.iloc[firm,:].to_frame().transpose()
                
                policies_g = policies_0.copy()
                
                g_mutate_index = np.random.randint(0,N)
                
                f0 = policies_0["Score"].values[0]
                
                policies_g.loc[:,g_mutate_index] = 1 - policies_0.loc[:,g_mutate_index]
    
                fg = scoring(N,K,E,Z,policies_g,shape_policy, fitness_landscape, g_e_relationships).iloc[0,:]["Score"]
                
                if policies_0["Type"].values[0] == "Shaper":
                    
                    e_mutate_index = np.random.randint(0,Z)
                    
                    shape_pol_mutate = shape_policy.copy()
                    
                    shape_pol_mutate[e_mutate_index] = 1 - shape_pol_mutate[e_mutate_index]
                    
                    fe = scoring(N,K,E,Z,policies_0,shape_pol_mutate, fitness_landscape, g_e_relationships).iloc[0,:]["Score"]
                
                    if fe > f0 and fe > fg:
                        
                        shape_policy = shape_pol_mutate
                        ft_copy = scoring(N,K,E,Z,ft_copy,shape_policy, fitness_landscape, g_e_relationships)
                        
                    
                    elif fg > f0 and fg > fe:
                        
                        policies_g["Type"] = "Shaper"
                        policies_g["Score"] = fg
                        
                        ft_copy.iloc[firm,:] = policies_g.iloc[0,:]  
                
                else:
                    
                    if fg > f0:
                        
                        policies_g["Type"] = "Searcher"
                        policies_g["Score"] = fg
                        
                        ft_copy.iloc[firm,:] = policies_g.iloc[0,:]  
                 
            firm_table = ft_copy
            
            avg_fit_searchers += [firm_table[firm_table["Type"] == "Searcher"]["Score"].mean()]
            
            avg_fit_shapers += [firm_table[firm_table["Type"] == "Shaper"]["Score"].mean()]
            
            end_iter = time.time()
            
            #print("Iteration",iteration,":",end_iter-start_iter,"s")
        
        master_avg_fit_searchers[run] = avg_fit_searchers
        master_avg_fit_shapers[run] = avg_fit_shapers
        
        end = time.time()
        
        #print("Landscape",run,":",end-start,"s\n")
    
    total_end = time.time()
    
    #print("Total Time:",total_end-total_start,"s\n")
    
    master_avg_fit_searchers = pd.DataFrame.from_dict(master_avg_fit_searchers)
    master_avg_fit_shapers = pd.DataFrame.from_dict(master_avg_fit_shapers)       
    

        
    return [master_avg_fit_searchers, master_avg_fit_shapers,firm_table]


def main(runs, iterations, N, K, Z, E, firm_count, shaper_proportion, random_seeds = []):
    
    searcher_results = np.nan
    shaper_results = np.nan
    
    if len(random_seeds) == 0:
        
        random_seeds = np.arange(runs)
        
    else:
        
        if len(random_seeds) != runs:
            
            raise Exception("The number of random seeds must be equal to the number of runs")
        
    
    for i,rs in enumerate(random_seeds):
        
        print(f"Random Seed no.{i}: {rs}")
        
        result = Standard(runs = runs, total_iteration = iterations, N = N, K = K, Z = Z, E = E, random_seeds = [],
                          firm_count = firm_count, shaper_proportion = shaper_proportion)
            
        if rs == 0:
                
            searcher_results = result[0]
            shaper_results = result[1]
            
        else:
                
            searcher_results[rs] = result[0]
            shaper_results[rs] = result[1]       

    return [searcher_results, shaper_results]


if __name__ == "__main__":
    
     results = main(runs = 1, 
                    iterations = 10, 
                    N = 12, 
                    K = 11, 
                    Z = 12, 
                    E = 12,  
                    shaper_proportion = 0.5,
                    firm_count = 10,
                    random_seeds = [])