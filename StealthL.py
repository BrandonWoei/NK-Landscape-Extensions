# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 14:19:18 2021

@author: deoxy
"""

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import time

class StealthL:
    
    def __init__(self, N=12,K=11,E=12,Z=12,
                 shaper_proportion=0.5,firm_count=10, random_state = 5, memory_size = 50, 
                 lr = 0.2, rm = 1, rm_decay = 0.999):
        
        ## Hyper-parameters
        
        self.N = N
        self.K = K+1         
        self.E = E
        self.Z = Z
        self.shaper_proportion = shaper_proportion
        self.firm_count = firm_count
        
        self.random_state = random_state
        np.random.seed(self.random_state)

        ## Initialize Model and Interaction Matrix
        
        self.interaction_matrix = self.interaction_matrix()
        
        self.initial_state = np.nan
        self.initial_state_red = np.nan
        
        self.fitness_landscape = self.NK_landscape()

        self.reset_warning = True
        
        self.origin_state = np.nan
        self.origin_state_red = np.nan
        self.fd_origin_state_red = {}
        
        self.pre_f_score = np.nan
        
        self.best_score = np.max(self.fitness_landscape)
        
        self.rl_agent = np.nan
        
        ##GA Variables
        
        self.mutation_vector = {}
        self.memory_size = memory_size
        self.memory = np.array([])
        self.lr = lr
        self.rm = rm
        self.rm_decay = rm_decay
        
    def interaction_matrix(self):
        
        inter_mat = {}
        
        for g in range(self.N):
        
            g_interact = np.random.choice(np.delete(np.arange(0,self.N),g), replace = False, size = self.K-1)
            
            e_interact = np.random.choice(np.arange(0,self.Z), replace = False, size = self.E)
            
            inter_mat[g] = [[g_interact],[e_interact]]
        
        return inter_mat


    def scoring(self, g_arr,e_arr, fitness_landscape, g_e_relationships):
        
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
                    
            for i in range(self.N):
                
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
    
    
    def NK_landscape(self):
        
        return np.random.uniform(0,1,size = (2**(self.K+self.E))*self.N).reshape(2**(self.K+self.E),self.N)
    
    
    def reset_agent(self):
        search_policies = np.random.randint(0,2, size = self.N*self.firm_count).reshape(self.firm_count,self.N)
        shape_policy = np.random.randint(0,2,size = self.Z).reshape(1,self.Z)[0]
         
        shaper_index = list(np.random.choice(np.arange(0,self.firm_count), replace = False, size = int(self.shaper_proportion*self.firm_count)))
            
        searcher_search_policies = pd.DataFrame(np.delete(search_policies,shaper_index, axis = 0))
        searcher_search_policies["Type"] = "Searcher"
            
        shaper_search_policies = pd.DataFrame(search_policies[shaper_index,:])
        shaper_search_policies["Type"] = "Shaper"
                
        firm_table = searcher_search_policies.merge(shaper_search_policies, how = 'outer')
            
        firm_table = self.scoring(firm_table,shape_policy,self.fitness_landscape, self.interaction_matrix)
        
        self.initial_state = [firm_table.copy(),shape_policy.copy()]
        
        self.mutation_vector = {}
        
        for agent in np.arange(self.firm_count):
            
            if self.initial_state[0].loc[agent,"Type"] == "Shaper":
                
                self.mutation_vector[agent] = np.random.random(self.N+self.Z)
            else:
                gvec = np.random.random(self.N)
                evec = np.repeat(0,self.Z)
                self.mutation_vector[agent] = np.append(gvec,evec)
        
        self.memory = np.array([])
        
        self.rm = 1

        self.reset_warning = False

            
        
    
    def sim(self, total_iteration = 1, plot = False):
        
        if self.reset_warning:
            raise Exception("Reset model using object.reset_agent() before calling object.model()")
            
            return 
        
        
        g_e_relationships = self.interaction_matrix
        
        master_avg_fit_searchers = {}
        master_avg_fit_shapers = {}
        
        #total_start = time.time()
        
        #start = time.time()
            
        firm_table = self.initial_state[0]
        shape_policy = self.initial_state[1]
        pseudo_initial_state = self.initial_state.copy()
        
        avg_fit_searchers = [firm_table[firm_table["Type"] == "Searcher"]["Score"].mean()]
        avg_fit_shapers = [firm_table[firm_table["Type"] == "Shaper"]["Score"].mean()]
        
        
        for iteration in range(total_iteration):
            
            start_iter = time.time()
            
            order = np.random.permutation(np.arange(self.firm_count))
            
            ft_copy = firm_table.copy()
            
            ft_copy = ft_copy.iloc[order,:]
            
            for firm in ft_copy.index:
                
                policies_0 = ft_copy.loc[firm,:].to_frame().transpose()
                
                policies_g = policies_0.copy()
                
                best_agent_position = np.argmax(ft_copy["Score"])
                
                ba_g = ft_copy.iloc[best_agent_position,0:self.N].values
                    
                self.mutation_vector[firm][0:self.N] = self.mutation_vector[firm][0:self.N]*(1-self.lr)+self.lr*ba_g
                    
                g_1 = np.where(np.random.uniform(low=np.nextafter(0.0, 1.0), high=1.0, size = self.N) < self.mutation_vector[0][:self.N])[0]
                g_0 = np.where(np.random.uniform(low=np.nextafter(0.0, 1.0), high=1.0, size = self.N) >= self.mutation_vector[0][:self.N])[0]
                
                ### RL Mutation for g ###
                
                #g_mutate_index = np.random.randint(0,self.N-1)
                
                f0 = policies_0["Score"].values[0]

                #policies_g.loc[:,g_mutate_index] = 1 - policies_0.loc[:,g_mutate_index]
                
                policies_g.loc[:,g_1] = 1
                policies_g.loc[:,g_0] = 0
    
                fg = self.scoring(policies_g,shape_policy, self.fitness_landscape, g_e_relationships).iloc[0,:]["Score"]
                
                if policies_0["Type"].values[0] == "Shaper":
                    
                    ### RL Mutation for e ###
                    
                    shape_pol_mutate = shape_policy.copy()
                    

                    e_mutate_index = np.random.randint(0,self.Z-1)

                    shape_pol_mutate[e_mutate_index] = 1 - shape_pol_mutate[e_mutate_index]
                    
                    fe = self.scoring(policies_0,shape_pol_mutate, self.fitness_landscape, g_e_relationships).iloc[0,:]["Score"]
                    
                if np.random.random() > self.rm:

                    best_mem_position = np.argmax(self.memory[:,-1])
                    best_mem = self.memory[best_mem_position,:]
                    
                    g_pol = best_mem[0:self.N]
                    e_pol = best_mem[self.N:self.N+self.Z]
                    
                    
                    if policies_0["Type"].values[0] == "Shaper":
                        
                        ft_copy.loc[firm,0:self.N] = g_pol
                        shape_policy = e_pol 
                        ft_copy = self.scoring(ft_copy,shape_policy, self.fitness_landscape, g_e_relationships)
                    
                    else:
                        ft_copy.loc[firm,0:self.N] = g_pol
                        ft_copy.loc[firm,"Score"] = self.scoring(ft_copy.loc[firm,:].to_frame().T,shape_policy, self.fitness_landscape, g_e_relationships).iloc[0,:]["Score"]
                        
                
                else:
                    
                    if policies_0["Type"].values[0] == "Shaper":
                        
                        if fe > f0 and fe > fg:
                                
                            shape_policy = shape_pol_mutate
                            ft_copy = self.scoring(ft_copy,shape_policy, self.fitness_landscape, g_e_relationships)
                                
                            
                        elif fg > f0 and fg > fe:
                            
                            policies_g["Type"] = "Shaper"
                            policies_g["Score"] = fg
                                
                            ft_copy.loc[firm,:] = policies_g.iloc[0,:]  
                    
                    else:
                        
                        if fg > f0:   
                            
                            policies_g["Type"] = "Searcher"
                            policies_g["Score"] = fg
                            
                            ft_copy.loc[firm,:] = policies_g.iloc[0,:]  
            
            
            firm_table = ft_copy
                
            avg_fit_searchers += [firm_table[firm_table["Type"] == "Searcher"]["Score"].mean()]
                
            avg_fit_shapers += [firm_table[firm_table["Type"] == "Shaper"]["Score"].mean()]
                
            end_iter = time.time()
            
            master_avg_fit_searchers[self.random_state] = avg_fit_searchers
            master_avg_fit_shapers[self.random_state] = avg_fit_shapers
            
            best_position = np.argmax(firm_table["Score"])
            best_gpolicy = firm_table.iloc[best_position,0:self.N].values
            best_score = firm_table.iloc[best_position,-1]
            memorize = np.append(best_gpolicy,shape_policy)
            memorize = np.append(memorize, best_score).reshape(1,self.N+self.Z+1)
        
            if self.memory.shape[0] == 0:
              
                self.memory = memorize
                
            elif self.memory.shape[0] < self.memory_size:
                
                if  not np.any(np.all(self.memory[:,self.N:-1] == shape_policy,axis = 1)):
                   
                    self.memory = np.concatenate((self.memory,memorize))
                
                else:
                    
                    get_index = np.where(np.all(self.memory[:,self.N:-1] == shape_policy,axis = 1))[0]
                    
                    for index in get_index:
                        
                        if best_score > self.memory[index,-1]:
                            
                            self.memory[index,:] = memorize
                    
            else:
                
                weakest_memory_index = np.argmin(self.memory[:,-1])
                
                if best_score > self.memory[weakest_memory_index,-1]:
                    
                    self.memory[weakest_memory_index,:] = memorize
            
            
            self.rm = self.rm*self.rm_decay
            

        master_avg_fit_searchers = pd.DataFrame.from_dict(master_avg_fit_searchers)
        master_avg_fit_shapers = pd.DataFrame.from_dict(master_avg_fit_shapers)       
        
        final_state = [firm_table,shape_policy]
        
       
        
        self.initial_state = final_state
                        
        
        return [master_avg_fit_searchers, master_avg_fit_shapers,final_state]




def main(runs,iterations, N, K, Z, E, lr, shaper_proportion, firm_count, random_seeds = []):
    
    if len(random_seeds) == 0:
        
        random_seeds = np.arange(runs)
        
    else:
        
        if len(random_seeds) != runs:
            
            raise Exception("The number of random seeds must be equal to the number of runs")
        
    
    for i,rs in enumerate(random_seeds):
        
        print(f"Random Seed no.{i}: {rs}")
    
        model = StealthL(N = N, 
                       K = K, 
                       Z = Z, 
                       E = E, 
                       lr = lr, 
                       shaper_proportion = shaper_proportion, 
                       firm_count = firm_count, 
                       random_state = rs)
    
        model.reset_agent()
        
        result = model.sim(total_iteration = iterations)

        if i == 0:
                    
            searcher_results = result[0]
            shaper_results = result[1]
                
        else:
                    
            searcher_results[rs] = result[0]
            shaper_results[rs] = result[1]
                
    
    return [searcher_results,shaper_results]


if __name__ == "__main__":
    
    results = main(runs = 100, 
                   iterations = 10, 
                   N = 12, 
                   K = 11, 
                   Z = 12, 
                   E = 12, 
                   lr = 0.2, 
                   shaper_proportion = 0.5,
                   firm_count = 10,
                   random_seeds = [])
    

