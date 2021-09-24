# NK-Landscape-Extensions

This repository contains three files:

1. Standard : Gavetti. et al (2017)'s NKZE model
2. StealthL : An extension of standard with elements from particle swarm optimisation and memory scheme genetic algorithms
3. StructC  : A constrained version of StealthL achieved by randomly grouping agents and limiting information within group boundaries


USER INSTRUCTIONS

Parameters of all models can be changed below the line if __name__ == "__main__": in all scripts

PARAMETER NOMENCLATURES
Note: some parameters are absent in some models 

runs              : number of runs - each run explores one randomly generated landscape
iterations        : the total number of iterations within a run (synonymous to the total number of actions an agent is take to make sequentially in a run)
N                 : search policy string length
K                 : ruggedness of landscape (0 <= K <= N-1)
Z                 : shape policy string length
E                 : malleability of landscape (0 <= E <= Z)
lr                : learning rate of agents (0 <= lr <= 1)
shaper_proportion : the proportion of firms (agents) that are shapers (0 <= shaper_proportion <= 1)
firm_count        : the agent population size
memory_size       : the maximum amount of memory that can be stored for reference (0 <= memory_size <= +inf)
rm                : the probability of an agent NOT referring to the stored memory (0 <= rm <= 1)
rm_decay          : the rate of decay of rm per iteration
max_group_size    : the largest group that can be formed randomly
random_seeds      : random seed specified in a list (default - unspecified, random seed of a run = run number)



