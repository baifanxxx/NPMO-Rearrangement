# NPMO-Rearrangement
This repository contains the implementation of the Non-prehensile Multi-object (NPMO) Rearrangement system for running on a UR3 Robot from the paper:

**Hierarchical Policy for Non-prehensile Multi-object Rearrangement with Deep Reinforcement Learning and Monte Carlo Tree Search**

*Fan Bai, Fei Meng, Jianbang Liu, Jiankun Wang, Max Q.-H. Meng*

[arXiv](https://arxiv.org/abs/2109.08973) | [Video]()

If you use this work, please cite the following as appropriate:

```text
@misc{bai2021hierarchical,
      title={Hierarchical Policy for Non-prehensile Multi-object Rearrangement with Deep Reinforcement Learning and Monte Carlo Tree Search}, 
      author={Fan Bai and Fei Meng and Jianbang Liu and Jiankun Wang and Max Q. -H. Meng},
      year={2021},
      eprint={2109.08973},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

**Contact**

Any questions or comments contact [Fan Bai](baifanxxx@gmail.com).


# Introduction
Non-prehensile multi-object rearrangement is a robotic task of planning feasible paths and transferring multiple objects to their predefined target poses without grasping. It needs to consider how each object reaches the target and the order of object movement, which significantly deepens the complexity of the problem. To address these challenges, We implemented an efficient Hierarchical Policy for Non-prehensile Multi-object (NPMO) Rearrangement, with specific contributions as follows:
1. We model the NPMO rearrangement task and solve it with a hierarchical policy;
2. We propose a high-level MCTS policy accelerated by the policy network trained with imitation and reinforcement;
3. We design a low-level policy to control the robot to achieve path primitives execution.

![image](https://github.com/baifanxxx/NPMO-Rearrangement/blob/main/figs/fig1.png)

# Method
We propose a hierarchical policy to divide and conquer for non-prehensile multi-object rearrangement. In the high-level policy, guided by a designed policy network, the Monte Carlo Tree Search efficiently searches for the optimal rearrangement sequence among multiple objects, which benefits from imitation and reinforcement. In the low-level policy, the robot plans the paths according to the order of path primitives and manipulates the objects to approach the goal poses one by one.

![image](https://github.com/baifanxxx/NPMO-Rearrangement/blob/main/figs/fig3_pipline.png)

# Experimental results
We verify through experiments that the proposed method can achieve a higher success rate, fewer steps, and shorter path length compared with the state-of-the-art.

## Qualitative result of algorithm execution in real robot experiments. The sequence of actions is a-j.
.<img src="https://github.com/baifanxxx/NPMO-Rearrangement/blob/main/figs/real_exp.png" width="400" height="270" />

## Comparing the performance of different methods. SR means Success Rate.
.<img src="https://github.com/baifanxxx/NPMO-Rearrangement/blob/main/figs/results.png" width="400" height="270" />

## The length of action sequence of increased objects for MCTS. 
.<img src="https://github.com/baifanxxx/NPMO-Rearrangement/blob/main/figs/fig5.png"/>
In each box, black data points are layed over a 1.96 Standard Error of Mean (95% confidence interval) in dark color and a 1 Standard Deviation in light color. Red lines represent mean values.

## The impact of different imitation levels on RL.
.<img src="https://github.com/baifanxxx/NPMO-Rearrangement/blob/main/figs/IL_curve.png"/>

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


<!--  
>### Remark
>Part of the code in this project refers to [SceneMover](https://github.com/HanqingWangAI/SceneMover), if you use the code of this project, please refer to this project and >[SceneMover](https://github.com/HanqingWangAI/SceneMover)
--> 
