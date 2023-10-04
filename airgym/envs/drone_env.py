
import os
import sys
import airsim
import numpy as np
import math
import gym
from gym import spaces
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from airgym.envs.airsim_env import AirSimEnv
import time

class AirSimDroneEnv(AirSimEnv):
    def __init__(self, ip_address,min_depth=0,max_depth=30):
        

        self.MIN_DEPT, self.MAX_DEPTH=min_depth,max_depth
        self.state = {
            "position": np.zeros(2),
            "collision": False,
            "velocity":np.zeros(2),
            "direction": np.zeros(2),
            "camera":np.zeros((84,84)),
            "relative_position" :np.zeros(2)
        }

        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.action_space = spaces.Discrete(7)
        self.pre_pos=None
        
        self._setup_flight()

        #self.image_request = airsim.ImageRequest( 3, airsim.ImageType.DepthPerspective, True, False)

    def __del__(self):
        self.drone.reset()

    def _setup_flight(self,yaw=10):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
           
        self.dist_pre = None
        self.start_point=[-2.0,77,-2]
        #self.start_point= [28.63,75.5,-2]
        self.target_point=[29.63,75.5,-2]

        self.start_x=self.start_point[0]
        self.start_y=self.start_point[1]
        self.start_z=self.start_point[2]
        self.start_yaw=yaw
        position = airsim.Vector3r(self.start_x,self.start_y)
        #orientation = airsim.to_quaternion(0, 0, self.start_yaw)
        #pose = airsim.Pose(position,orientation)
        pose = airsim.Pose(position)
        self.drone.simSetVehiclePose(pose, ignore_collision=True)

        self.drone.moveToPositionAsync(self.start_x, 
                                       self.start_y, 
                                       self.start_z, 
                                       0
                                       )
    
    def transform_obs(self, responses):
        
         # Reshape to a 2d array with correct width and height
        depth_img = airsim.list_to_2d_float_array(responses.image_data_float, responses.width, responses.height)
        depth_img = depth_img.reshape(responses.height, responses.width, 1)
        depth_img= np.interp(depth_img, (self.MIN_DEPT, self.MAX_DEPTH), (0,255))
        depth_img=depth_img/255
       
        return depth_img 

    def get_distance(self,position):

     #  print(self.target_point[:-1])
        distance= np.linalg.norm(position-self.target_point[:-1])
      #  print(distance)
        return distance
    def vector(self):
        
        direction_vector = self.state['direction']

        distance = (self.state["relative_position"][0]**2 + self.state["relative_position"][1]**2)**0.5
        target_vector = self.state["relative_position"]/distance # 현재 방향 벡터

        # 많이 다르면 0, 방향이 같으면 1
        cosine_similarity = np.dot(direction_vector, target_vector)
        
        vector_Force = cosine_similarity

        return vector_Force
    
    def get_direction(self):
        
        yaw = airsim.to_eularian_angles(self.drone_state.kinematics_estimated.orientation)[2]
        
        x = np.cos(yaw)
        y = np.sin(yaw)

        direction_vector = np.array([x, y])
        return direction_vector
    
    def check_out(self,pos_x,pos_y):
        
        if pos_x>32.63 or pos_x<-2.2 or pos_y> 82 or pos_y<73 :
            return True
        
        return False
        
    def distance_travelled(self):
        if self.dist_pre is None:
            dist_travelled = 0
        else:
            dist_travelled =  np.linalg.norm(self.state['position']-self.dist_pre)
        #print('dist_travelled:',dist_travelled)
        self.dist_pre = self.state['position']
        return dist_travelled
    def _get_obs(self):
        
        #depth 이미지 저장
        response = self.drone.simGetImages([airsim.ImageRequest("front", airsim.ImageType.DepthPerspective, True, False)])[0]
        image = self.transform_obs(response)
        self.drone_state = self.drone.getMultirotorState()
        image = np.squeeze(image)
        self.state["camera"] = np.array(image)
        
        #global position저장
        self.state["position"] = np.array([self.drone_state.kinematics_estimated.position.x_val,
                                           self.drone_state.kinematics_estimated.position.y_val
                                           ])
        
        #속도 저장
        self.state["velocity"] = np.array([self.drone_state.kinematics_estimated.linear_velocity.x_val,
                                           self.drone_state.kinematics_estimated.linear_velocity.y_val
                                           ])
        
        # local position(current_pos-target_pos)
        self.state["relative_position"] = np.array([
        self.target_point[0]-self.drone_state.kinematics_estimated.position.x_val,
        self.target_point[1]-self.drone_state.kinematics_estimated.position.y_val
        ])

        #드론의 방향 (x=cos(yaw), y=sin(yaw)로)
        self.state["direction"] = self.get_direction()
        
        
        #충돌 여부 저장
        collision = self.drone.simGetCollisionInfo().has_collided
        # 맵 벗어났는지 확인
        pos_x=self.state["position"][0]
        pos_y=self.state["position"][1]
        out=self.check_out(pos_x,pos_y)
        #맵 벗어난것도 충돌로 간주
        if out :
            collision=True
        
        self.state["collision"] = collision
        

        return self.state


    def get_distance(self):
        dist_current = np.linalg.norm(self.state['position']-self.target_point[:-1])
        return dist_current
    

    def _compute_reward(self):
        
        done = 0
        success=False
        vector = self.vector()
        x= self.distance_travelled()
        dist_current = self.get_distance()
        dist_reward = min(1.7,x)
        #print('vector',vector)
        if vector >0.98:
            
            dist_reward*=1
            #if vector >0.99:
             #   dist_reward+=0.05
        else:
            dist_reward*=-1
        reward = dist_reward
        
        #목적지 통과
        x = self.state['position'][0]
        y = self.state['position'][1]    
        
        
        
        if x > 28.63 and y < 78 and y > 73 :
            print('-----------------------------goal!!!!!!!!-----------------------')
            reward=2
            done=1
            success=True
            self.state['collision'] = False
            return reward, done,success
        # 충돌시 패널티
        if self.state['collision']:
            reward=-2
            done=1
            
            
        
        return reward,done,success            
        
       

        

    def _do_action(self, action):
       #discrete action 추출
        action=self.interpret_action(action)
       
        v_x=action[0]
        yaw=action[1]
       
        
        if yaw == 0 :
             v_z=-0.6
        else:
            v_z=-0.1
        if yaw==0 and v_x==0:
            v_z=0

        if self.drone.simIsPause():
            self.drone.simPause(is_paused=False)

       # now = time.time()

        self.drone.moveByVelocityZBodyFrameAsync(
            vx = v_x,
            vy = 0,
            z = v_z,#(self.random_alt),
            duration = 0.25,
            drivetrain = airsim.DrivetrainType.ForwardOnly,
            yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate=float(yaw))
        ).join()
        #print(time.time() - now)
        self.drone.simPause(True)

    def interpret_action(self, action):
       
        if action == 0:
            quad_offset = [7,0]
        elif action == 1:
            quad_offset = [2,0]
        elif action == 2:
            quad_offset = [0,30]
        elif action == 3:
            quad_offset = [0,-30]

        elif action == 4:
            quad_offset = [0,60]
        elif action == 5:
            quad_offset = [0,-60]
       
        elif action == 6 : 
            quad_offset = [4,0]    
        elif action == 7 : 
            quad_offset = [0,-45]
        elif action == 8 : 
            quad_offset = [0,45]
        elif action == 9 : 
            quad_offset = [0,-1]
        elif action == 10 : 
            quad_offset = [0,1]
        elif action == 11 : 
            quad_offset = [5,0]
        else:
            quad_offset = [0,0]

        return quad_offset


    def step(self, action):
        self._do_action(action)
        #time.sleep(0.01)
        obs = self._get_obs()
        reward, done,success = self._compute_reward()

        return obs, reward, done,success

    def reset(self):
        self._setup_flight()
        return self._get_obs()
