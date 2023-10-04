import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from airgym.envs.drone_env import AirSimDroneEnv

import gym




if __name__ == '__main__':
   
    drone= AirSimDroneEnv(ip_address="127.0.0.1"
                        
                        
    )
    while True:
        action=int(input("input:"))
        

        drone.step(action)
        #drone._do_action([5,0])
        print('ss')
        print('vector:',drone.vector())
