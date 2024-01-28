import torch as pt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from ml_funcs import save_load


class Linear_QNet(nn.Module):
    def __init__(self, input_features:int, output_features:int, hidden_units: int = 10):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
        nn.Linear(in_features=input_features, out_features=hidden_units),
        nn.ReLU(),
        nn.Linear(in_features=hidden_units, out_features=output_features)
        )
    def forward(self, x:pt.Tensor) -> pt.Tensor:
        return self.conv_block_1(x)
    
    def save(self, file_name = 'model.pth'):
        model_folder_path = './models'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        pt.save(self.state_dict(), file_name)
    

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optim = pt.optim.Adam(model.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = pt.tensor(state, dtype=pt.float)
        next_state = pt.tensor(state, dtype=pt.float)
        action = pt.tensor(state, dtype=pt.float)
        reward = pt.tensor(state, dtype=pt.float)

        if len(state.shape) == 1: # if we have 1 number
            # (1, x) to make 52+ functional
            state = pt.unsqueeze(state, dim=0)
            next_state = pt.unsqueeze(next_state, dim=0)
            action = pt.unsqueeze(action, dim=0)
            reward = pt.unsqueeze(reward, dim=0)
            done = (done, )
        # 1. predict Q value with current state
        pred = self.model(state)

        target = pred.clone()
        for i in range(len(done)):
            Q_new = reward[i]
            if not done[i]: # bc if done there are no future states
                Q_new = reward[i] + self.gamma*pt.max(self.model(next_state[i]))
                # basically it takes the reward as the Q value and adds the next reward which might come when the nn continues
            target[i][pt.argmax(action[i]).item()] = Q_new
            # picks out one state and 


        # 2. Q_new = r+y * max(next_predicted Q value) -> only do this step if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new

        # like in a training loop
        loss = self.loss_func(target, pred)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

