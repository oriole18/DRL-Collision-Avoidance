
from collections import deque
import numpy as np
import random
import pickle
import os
import torch
class ReplayBuffer(object):
    def __init__(self, capacity,batch_size):
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size
    def push(self, state, action, reward, next_state, done):
        #state = np.expand_dims(state, 0)
        #next_state = np.expand_dims(next_state, 0)
            
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self):

        #state는 stack이된 state임
        states, action, reward, next_states, done = zip(*random.sample(self.buffer, self.batch_size))
        
        
        # 각 요소를 개별적인 batch로 변환
        camera_batch = np.array([sample['camera'] for sample in states])
        direction_batch = np.array([sample['direction'] for sample in states])
        relative_batch = np.array([sample['relative_position'] for sample in states])
        
        states_batch = {
                        'camera': camera_batch,
                       'direction': direction_batch,
                       'relative_position' : relative_batch,
                       
                        }
        
        next_camera_batch = np.array([sample['camera'] for sample in next_states])
        next_direction_batch = np.array([sample['direction'] for sample in next_states])
        next_relative_position_batch = np.array([sample['relative_position'] for sample in next_states])
        
        next_states_batch ={
                            'camera': next_camera_batch,
                            'direction' : next_direction_batch,
                            'relative_position' : next_relative_position_batch,
                            }
        #print('state_img_stack:',len(states_batch))
       # print('Camera Batch Shape:', states_batch['camera_batch'].shape)

        return states_batch, action, reward, next_states_batch, done
    
    def save_buffer(self):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        
        save_path = "checkpoints/dqn_buffer"
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, memory_path= "checkpoints/dqn_buffer"):
        # 파일에서 데크를 불러오기
        with open(memory_path, 'rb') as f:
            self.buffer = pickle.load(f)

    def __len__(self):
        return len(self.buffer)
