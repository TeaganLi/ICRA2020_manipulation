# Learning Hierarchical Control for Robust In-Hand Manipulation

Robotic in-hand manipulation has been a long-standing challenge due to the complexity of modelling hand and object in contact and of coordinating finger motion for complex manipulation sequences. To address these challenges, the majority of prior work has either focused on model-based, low-level controllers or on model-free deep reinforcement learning that each have their own limitations. We propose a hierarchical method that relies on traditional, model-based controllers on the low-level and learned policies on the mid-level. The low-level controllers can robustly execute different manipulation primitives (reposing, sliding, flipping). The mid-level policy orchestrates these primitives. We extensively evaluate our approach in simulation with a 3-fingered hand that controls three degrees of freedom of elongated objects. We show that our approach can move objects between almost all the possible poses in the workspace while keeping them firmly grasped. We also show that our approach is robust to inaccuracies in the object models and to observation noise. Finally, we show how our approach generalizes to objects of other shapes.

The paper and video can be found at [Paper](https://arxiv.org/abs/1910.10985), [Video](https://www.youtube.com/watch?time_continue=8&v=s8j2b79ByuQ). You can find more information at our [project page](https://sites.google.com/view/learninghierarchicalcontrol/home).

## Get Started
- Clone the repo and cd into it:
  ```
  git clone https://github.com/TeaganLi/ICRA2020_manipulation.git
  cd ICRA2020_manipulation
  ``` 
- Uncompress feasible states under different contact configurations:
  ```
  cd envs
  tar -xvzf data.tar.gz
  ```
There are two types of environments, low-level controller environments and raw controller environments for end-to-end RL. low_level_controller_env.py contains three environments for easy, medium, and hard goals, respectively. raw_controller_env.py contains environments for end-to-end RL. The details of the low level controllers can be found in dynamics_calculator.py.


If you think our work is useful, please consider citing use with
```
@article{li2019learning,
  title={Learning Hierarchical Control for Robust In-Hand Manipulation},
  author={Li, Tingguang and Srinivasan, Krishnan and Meng, Max Qing-Hu and Yuan, Wenzhen and Bohg, Jeannette},
  journal={arXiv preprint arXiv:1910.10985},
  year={2019}
}
```
