
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self,stack_size=2, num_actions=3):
        super(DQN,self).__init__()
        
        
        self.stack_size = stack_size
        self.out_size_img=300
        self.out_size_dyn = 50
        
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=self.stack_size, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.img_layers = nn.Sequential(
            nn.Linear(64*7*7, self.out_size_img),  # 이미지 크기가 84x84이므로, MaxPool2d로 인해 크기가 21x21이 됩니다.
            nn.ReLU(),
        )

        self.input_layer_dyn = nn.Sequential(
            nn.Linear(2, self.out_size_dyn),
            nn.ReLU(),
            nn.Flatten(),
            
        )

        # layers
        self.fc1 = nn.Linear(700,256)
        self.fc2 = nn.Linear(256,256)
        # Advantage stream
        self.fc = nn.Linear(256, num_actions)
        

    def forward(self,img,direction,position):

        features = self.cnn(img)
        img_features = self.img_layers(features)
            #print('img_feature',img_features.shape)
        
        direction_feature = self.input_layer_dyn(direction)
        position_feature = self.input_layer_dyn(position)
            
        x = torch.cat([img_features,direction_feature,position_feature], dim=1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        

        # Combine value and advantage to get Q-values
        q_values = self.fc(x)

        return q_values

    




class Dueling_DQN(nn.Module):
    def __init__(self,stack_size=2, num_actions=3):
        super(Dueling_DQN,self).__init__()
        
        
        self.stack_size = stack_size
        self.out_size_img=300
        self.out_size_dyn = 50
        
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=self.stack_size, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.img_layers = nn.Sequential(
            nn.Linear(64*7*7, self.out_size_img),  # 이미지 크기가 84x84이므로, MaxPool2d로 인해 크기가 21x21이 됩니다.
            nn.ReLU(),
        )

        self.input_layer_dyn = nn.Sequential(
            nn.Linear(2, self.out_size_dyn),
            nn.ReLU(),
            nn.Flatten(),
            
        )

        #shared layers
        self.fc1 = nn.Linear(700,256)
        self.fc2 = nn.Linear(256,256)
    
         # Value stream
        self.fc_val = nn.Linear(256, 1)

        # Advantage stream
        self.fc_adv = nn.Linear(256, num_actions)
        

    def forward(self,img,direction,position):

        features = self.cnn(img)
        img_features = self.img_layers(features)
            #print('img_feature',img_features.shape)
        
        direction_feature = self.input_layer_dyn(direction)
        position_feature = self.input_layer_dyn(position)
            
        x = torch.cat([img_features,direction_feature,position_feature], dim=1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Split the output into value and advantage streams
        val = self.fc_val(x)
        adv = self.fc_adv(x)

        # Combine value and advantage to get Q-values
        q_values = val + (adv - adv.mean(dim=1, keepdim=True))

        return q_values

    









