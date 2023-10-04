import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from airgym.envs.drone_env import AirSimDroneEnv
from model import DQN
import torch
import torch.nn as nn
from model import Dueling_DQN,DQN
from collections import deque
import torch.optim as optim
import torch.autograd as autograd 
import numpy as np
from replay_buffer import ReplayBuffer
import torch.nn.functional as F



class DQN_Agent(object):

    def __init__(self,args,model):
        
        self.device = torch.device("cuda")
        self.stack_size = args.stack_size
        
        if model == 1:
            print("start DQN!!!!!!!!!")
            self.q_network = DQN(self.stack_size,args.action_num).to(self.device)
            self.target_network =  DQN(self.stack_size,args.action_num).to(self.device)
            self.model = 'DQN'
        else:
            print("start Dueling DQN!!!!!!")
            self.q_network = Dueling_DQN(self.stack_size,args.action_num).to(self.device)
            self.target_network =  Dueling_DQN(self.stack_size,args.action_num).to(self.device)
            self.model = 'Dueling_DQN'
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.lr)
        
        self.action_num=args.action_num
        self.max_reward = args.max_reward
        self.min_reward = args.min_reward
        self.gamma = args.gamma
        #exploration param
        self.epsilon_start = 1
        self.epsilon = self.epsilon_start
        self.epsilon_end = 0.001
        self.exploration = args.exploration
        self.epsilon_decay_step = (self.epsilon-self.epsilon_end)/self.exploration
        self.alpha = args.alhpa
        # 프레임 스택을 위한 stack들
        self.camera_stack = deque(maxlen = self.stack_size)
        self.direction_stack = deque(maxlen = self.stack_size)
        self.relative_position_stack = deque(maxlen = self.stack_size)
        
        # replay buffer 선언
        self.replay_buffer =  ReplayBuffer(capacity=args.buffer_size,batch_size=args.batch_size)
        
        
        
    # state의 요소들을 stack에 추가
    def stack_frames(self, state, is_new_episode=False):
        frame = state['camera']
        direction = state['direction']
        relative_position = state['relative_position']
        
        # 에피소드가 시작될 때, 동일한 내용을 여러 번 stack
        if is_new_episode:
            self.camera_stack.extend([frame] * self.stack_size)
            self.direction_stack.extend([direction] * self.stack_size)
            self.relative_position_stack.extend([relative_position] * self.stack_size)
        else:
            # 요소를 각각의 deque에 추가
            self.camera_stack.append(frame)
            self.direction_stack.append(direction)
            self.relative_position_stack.append(relative_position)
        
        state_stack = {
                    'camera': np.array(self.camera_stack),
                    'direction': np.array(self.direction_stack),
                    'relative_position': np.array(self.relative_position_stack),
                    
                      }
        
        return state_stack
    
    def pre_processing_state(self,state,train_mode):
        
        #depth img
        img = state['camera']
        img = np.array(img, dtype=np.float32)
        img = torch.tensor(img).to(self.device)

        #드론의 heading 방향
        direction = (state['direction']+1)/2
        direction = np.array(direction, dtype=np.float32)
        direction = torch.tensor(direction).to(self.device)
        #현재 위치와 목적지의 상대적 위치
        state['relative_position'][0]=(state['relative_position'][0]+3)/34.83
        state['relative_position'][1]=(state['relative_position'][1]+6.5)/9
        
        position = state['relative_position']
        position = np.array(position, dtype=np.float32)
        position = torch.tensor(position).to(self.device)
        
        
        #action 추론시 차원 증가
        if train_mode is False:
            img = img.unsqueeze(0)
            position = position.unsqueeze(0)
            direction = direction.unsqueeze(0)

        return img,direction,position     


    def get_action(self,state):
       
        if self.epsilon > np.random.rand():
            return np.random.randint(0,self.action_num)
        else:
            with torch.no_grad():
                img,direction,position = self.pre_processing_state(state,train_mode=False)
                Q = self.q_network.forward(img,direction,position)
                return  np.argmax(Q.cpu().detach().numpy())
            
    

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    
    def train(self,writer,time_step):
        
        #epsilon 감소
        if self.epsilon >self.epsilon_end:
            self.epsilon-=self.epsilon_decay_step
        else:
            self.epsilon = 0
        
        #replay_buffer가 꽉차기 전까지는 학습 안함
        if self.replay_buffer.__len__() < self.replay_buffer.batch_size:
            return False,None,None
        
        #replay buffer로부터 sampleing 
        state_batch,action_batch,reward_batch,next_state_batch,done_batch =self.replay_buffer.sample()
        
        #stack된 state들 batch전환
        img_batch,direction_batch,velocity_batch = self.pre_processing_state(state_batch,train_mode=True)
        next_img_batch,next_direction_batch,next_velocity_batch = self.pre_processing_state(next_state_batch,train_mode= True)
        
        action_batch = torch.LongTensor(action_batch).to(self.device).unsqueeze(1)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        reward_batch=(reward_batch+2)/4
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)
        done_batch = done_batch.view(-1)
        
          
        #q_val
        q_value = self.q_network.forward(img_batch,direction_batch,velocity_batch)
        q_value = q_value.gather(1, action_batch).squeeze(1)
        # ,next_q_val
        with torch.no_grad():     
            next_q_value = self.target_network(next_img_batch,next_direction_batch,next_velocity_batch)
            next_q_value = next_q_value.max(1)[0]
            expected_q_value = self.alpha*(reward_batch + self.gamma * next_q_value * (1 - done_batch))

        #loss  
        loss = F.smooth_l1_loss(q_value, expected_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        writer.add_scalar('loss/time_step',loss,time_step)
    
    
    # Save model parameters
    def save_checkpoint(self,time_step,total_reward,episode,total_success):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        
        save_path = "checkpoints/"+self.model+"_checkpoint"
        print('Saving models to {}'.format(save_path))
        torch.save({'q_network': self.q_network.state_dict(),
                    'target_network': self.target_network.state_dict(),
                    #'optimizer_state_dict': self.optimizer.state_dict(),
                    }, save_path)
        
       # self.replay_buffer.save_buffer()



     # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
           # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if evaluate :
                self.q_network.eval()
                self.target_network.eval()
            else:
                self.q_network.train()
                self.target_network.train()
            
            #self.replay_buffer.load_buffer()
            

            




