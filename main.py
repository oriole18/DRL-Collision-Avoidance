import os
import sys
import time
import argparse
import datetime
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from airgym.envs.drone_env import AirSimDroneEnv
from DQN import DQN_Agent
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='DQN Args')



parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')

parser.add_argument('--lr', type=float, default=0.0001, metavar='G',
                    help='learning rate (default: 0.0001)')

parser.add_argument('--stack_size', type=float, default=4, metavar='G',
                    help='stack_size (default: 2)')

parser.add_argument('--action_num', type=int, default = 12, metavar='G',
                    help='action_number (default: 3)')

parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')

parser.add_argument('--buffer_size', type=int, default=45000, metavar='N',
                    help='size of replay buffer (default: 50000)')

parser.add_argument('--exploration', type=float, default=300000, metavar='N',
                    help='exploration_step (default: 100000)')

parser.add_argument('--alhpa', type=float, default=0.6, metavar='N',
                    help='exploration_step (default: 100000)')


parser.add_argument('--max_reward', type=float, default =2 , metavar='N',
                    help='max_reward (default: 2)')

parser.add_argument('--min_reward', type=float, default=-2, metavar='N',
                    help='min_reward (default: -2)')

args = parser.parse_args()

env = AirSimDroneEnv(ip_address="127.0.0.1",min_depth=0,max_depth=15)

while True :
    model = int(input("dqn is 1 and dueling dqn is 2:"))
    if model !=1 and model!=2:
        print("wrong input!!!")
    else:
        break

agent= DQN_Agent(args=args,model = model)

print("--------------------------------------------------")
print("-----using-device:",agent.device)
print("--------------------------------------------------")
print("--------------------------------------------------")



while True :
    mode = input("loading model ???(yes or no):")
    #평가 모드
    if mode == 'yes':
        file = input("file_naem:")
        
        if model ==1:
            path= file +"/DQN_checkpoint"
        else:
            path= file +"/Dueling_DQN_checkpoint"
        
        if not os.path.exists(path):
            print("not exist file!!!!")
            pass

        agent.load_checkpoint(path,evaluate=True)
        print("complete loading!!!!!!!")
        time.sleep(1)
        train_mode = False
        break
    #학습모드
    elif mode =="no":
        writer = SummaryWriter('runs/{}_{}_DQN'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),model))
        train_mode = True
        break
    else:
        print(" error you should only yes or no !!!!!!!!!!!!!!")


def train(writer):
    
    max_episode =300000
    max_episdoe_len = 170
    update_target_rate = 10000
    save_frequency =10000
    time_step = 0 
    total_reward = 0 
    total_success = 0
    goal_rate=100
    
    for i_episode in range(1,max_episode):

        obs = env.reset()
        episode_reward = 0
        done = 0
    
        stack_obs = agent.stack_frames(obs,True)
        
        for episode_step in range(max_episdoe_len):
            #action 추출 
            action = agent.get_action(stack_obs)
            # env로 부터 observ
            #print(action)
            next_obs, reward, done, success = env.step(action)
            episode_reward += reward
            total_reward += reward
            
            
            if success :
                total_success+=1

            # next obs를 stack_state형태로    
            next_stack_obs = agent.stack_frames(next_obs,False)
            
            #memory에 저장
            agent.replay_buffer.push(stack_obs,action,reward,next_stack_obs,done)
            
            time_step+=1
            
            #model trainning
            agent.train(writer,time_step)
            
            if total_reward ==0:
                record = 0
            else:
                record = total_reward/time_step

            writer.add_scalar('reward_avg/time_step',record,time_step)
            writer.add_scalar('epsilon/time_step',agent.epsilon,time_step)
            
            # target network 업글
            if time_step % update_target_rate ==0:
                agent.update_target()
            #모델 및 환경 저장
            if time_step % save_frequency ==0:
                agent.save_checkpoint(time_step,total_reward,i_episode,total_success)

            stack_obs = next_stack_obs
            
            #에피소드 종료
            if done :
                break
        
        writer.add_scalar('reward/episode',episode_reward,i_episode)
        if total_reward == 0:
            record =0
        else:
            record = total_reward/i_episode
        writer.add_scalar('reward_avg/episode',record,i_episode)
        
        if i_episode % goal_rate==0:
            if total_success ==0:
                record =0
            else:
                record = total_success/goal_rate
            writer.add_scalar('goal_avg/episode',record,i_episode)
            total_success = 0
        
        print("----------------------------------")
        print('episode:',i_episode)
        print('step :',episode_step)
        print('reward:',episode_reward)
        print("----------------------------------")

def eval_mode():
    
    agent.epsilon = 0
    episode = 20
    for i_episode in range(episode):

        obs = env.reset()
        episode_reward = 0
        done = 0
    
        stack_obs = agent.stack_frames(obs,True)
        
        for episode_step in range(150):
            #action 추출 
            action = agent.get_action(stack_obs)
            # env로 부터 observ
            #print(action)
            next_obs, reward, done,_ = env.step(action)
            episode_reward += reward
           # print('---------------------------------------------')
            print("step reward:",reward)
            #print('---------------------------------------------')
            
        
            # next obs를 stack_state형태로    
            next_stack_obs = agent.stack_frames(next_obs,False)

            stack_obs = next_stack_obs
            
            #에피소드 종료
            if done :
                break
        
        print("----------------------------------")
        print('episode:',i_episode)
        print('step :',episode_step)
        print('reward:',episode_reward)
        print("----------------------------------")



if __name__ == '__main__':
    
    if train_mode is True:
        print('---------------------------------------')
        print('--------------train_mode---------------')
        print('---------------------------------------')
        time.sleep(1)
        train(writer)
    else:
        print('---------------------------------------')
        print('--------------evaluate_mode------------')
        print('---------------------------------------')
        time.sleep(1)
        eval_mode()