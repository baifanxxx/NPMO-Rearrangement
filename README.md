# NPMO-Rearrangement
This repo is our ICRA2022 paper about 'Hierarchical Policy for Non-prehensile Multi-object Rearrangement with Deep Reinforcement Learning and Monte Carlo Tree Search
'
# Introduction
We have implemented an efficient Hierarchical Policy for Non-prehensile Multi-object (NPMO) Rearrangement, with specific contributions as follows:
1. We model the NPMO rearrangement task and solve it with a hierarchical policy;
2. We propose a high-level MCTS policy accelerated by the policy network trained with imitation and reinforcement;
3. We design a low-level policy to control the robot to achieve path primitives execution.

![image](https://github.com/baifanxxx/IPM-MovePlanner/blob/main/IPM-MovePlaner/figs/Structure_diagram.png)

# Environment Installation
1. Install Requirements
  python 3.6
  pytorch 1.7.1
  CUDA 10.1
  tensorflow 1.15.0
2. Build Files 
```
  cd src/utils
  g++ -shared -O2 search.cpp --std=c++11 -ldl -fPIC -o search.so
```
The configuration method of these environments can refer to the [link](https://github.com/HanqingWangAI/SceneMover)

# Experimental results
## Compare network structure
.<img src="https://github.com/baifanxxx/IPM-MovePlanner/blob/main/IPM-MovePlaner/figs/net_success_rate.jpg" width="400" height="270" />
.<img src="https://github.com/baifanxxx/IPM-MovePlanner/blob/main/IPM-MovePlaner/figs/net_loss.jpg" width="400" height="270" />

## Compare PPO with PPO+IL
.<img src="https://github.com/baifanxxx/IPM-MovePlanner/blob/main/IPM-MovePlaner/figs/rewards.png" width="400" height="270" />
.<img src="https://github.com/baifanxxx/IPM-MovePlanner/blob/main/IPM-MovePlaner/figs/test_average_step.png" width="400" height="270" />
.<img src="https://github.com/baifanxxx/IPM-MovePlanner/blob/main/IPM-MovePlaner/figs/test_success_rate.png" width="400" height="270" />
.<img src="https://github.com/baifanxxx/IPM-MovePlanner/blob/main/IPM-MovePlaner/figs/loss.png" width="400" height="270" />

## Compare our method with others
.<img src="https://github.com/baifanxxx/IPM-MovePlanner/blob/main/IPM-MovePlaner/figs/table.jpg"/>
<!--  
>### Remark
>Part of the code in this project refers to [SceneMover](https://github.com/HanqingWangAI/SceneMover), if you use the code of this project, please refer to this project and >[SceneMover](https://github.com/HanqingWangAI/SceneMover)
--> 
