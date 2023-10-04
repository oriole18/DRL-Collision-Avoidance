# DRL-Collision-Avoidance
Quadrotor collision avoidance and reaching the goal point

#### NOTE
It's a Toy Project for studying RL
It considers only a simple environment and a small map

# Environment 
I used binary files of AirSim: MSBuild2018
(https://github.com/microsoft/AirSim/releases)


# RL Model
1. DQN
2. Dueling DQN

### Reward function 
    if collision or out of map :
        reward =-2
    if goal :
        reward = 2
    if heading of Quadrotor  is the right direction:
        distance_travelled
    else:
        -distance_travelled    
### Parameter

-Clockspeed:5.0
-Command frequency: 20 hz
-The agent's objectives are collision avoidance and reaching the goal point, so the agent's policy is sensitive
-I solved this problem using the constraint MDP, setting alpha=0.6

### State

1. Depth image
2. The relative position between the current agent's position and the goal point's position
3. Quadrotor heading direction

# Result 

### Video
#### after 200 Episode
1. After 200 episodes, the agent has become very slow and is not performing well in avoiding obstacles
![after200epi](demo_after200epi.gif)
2. After 2000 episodes, the agent has become faster at reaching the goal point and better at avoiding obstacles
![after2000epi](demo_after2000episode.gif)

### tensorboard result
goal_avg is probability of reached goal
1. DQN
<p align="center">
    <img src="/tensorboard_log/DQN.png" width="700" height="300">
   
</p>

2. Dueling DQN

<p align="center">
    <img src="/tensorboard_log/Dueling_DQN_result.png" width="350" height="300">
    <img src="/tensorboard_log/Dueling_DQN_result_smooth.png" width="350" height="300">
</p>

3. Dueling DQN VS DQN
<p align="center">
    <img src="/tensorboard_log/Dueling_DQNvsDQN-1.png" width="700" height="300">
   
</p>

