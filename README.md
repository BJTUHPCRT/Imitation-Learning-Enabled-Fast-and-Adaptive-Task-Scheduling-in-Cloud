# Imitation-Learning-Enabled-Fast-and-Adaptive-Task-Scheduling-in-Cloud

# Abstract
Studies of resource provision in cloud computing have drawn extensive attention, since effective task scheduling solutions promise an energy-efficient way of utilizing resources while meeting diverse requirements of users.Deep reinforcement learning (DRL) has demonstrated its outstanding capability in tackling this issue with the ability of online self-learning, however, it is still prevented by the low sampling efficiency, poor sample validity, and slow convergence speed especially for deadline constrained applications.To address these challenges, an Imitation Learning Enabled Fast and Adaptive Task Scheduling (ILETS) framework based on DRL is proposed in this paper.
First, we introduce behavior cloning to provide a well-behaved and robust model through Offline Initial Network Parameters Training (OINPT) so as to guarantee the initial decision-making quality of DRL.
Next, we design a novel Online Asynchronous Imitation Learning (OAIL)-based method to assist the DRL agent to re-optimize its policy and to against the oscillations caused by the high dynamic of the cloud, which promises DRL agent moving toward the optimal policy with a fast and stable process.Extensive experiments on the real-world dataset have demonstrated that the proposed ILETS can consistently produce shorter response time, lower energy consumption and higher success rate than the baselines and other state-of-the-art methods at the accelerated convergence speed.

# Citation
@article{kang2024imitation,
  title={Imitation learning enabled fast and adaptive task scheduling in cloud},
  author={Kang, KaiXuan and Ding, Ding and Xie, HuaMao and Zhao, LiHong and Li, YiNong and Xie, YiXuan},
  journal={Future Generation Computer Systems},
  volume={154},
  pages={160--172},
  year={2024},
  publisher={Elsevier}
}
