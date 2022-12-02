import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        # We will only use 3 layered neural network. 
        self.lin1 = nn.Linear(input_size,hidden_size)
        self.dropout = nn.Dropout(p=0.1)
        self.lin2 = nn.Linear(hidden_size,output_size)
    
    def forward(self,x):
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)
        return x
    
    def save(self,file_name='model.pth'):
        folder_path = './model'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        file_name = os.path.join(folder_path,file_name)
        torch.save(self,file_name)
    
class QTrainer:
    def __init__(self,model:Linear_QNet,lr,gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(),lr=self.lr) # We need to choose an optimizer to optimize our steps to optimal values for parameters. 
        self.criterion = nn.MSELoss() # The loss function that we will be using to improve the model parameters. MSELoss=Mean Squared Error.

    def train_step(self,state, action, reward, next_state, done):

        # This is tabular Deep Q network. We haven't used function approximation or Policy gradient.

        # Each parameter here can be one unique value or a tuple of values, we need to get it to a unique format. Better to go for a pytorch tensor.
        state = torch.tensor(state,dtype=torch.float)
        next_state = torch.tensor(next_state,dtype=torch.float)
        action = torch.tensor(action,dtype=torch.float)
        reward = torch.tensor(reward,dtype=torch.float)

        if len(state.shape) == 1: # tensor is 1D, we would want it in 2 D. This function will turn it into 2D.
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state,0)
            action = torch.unsqueeze(action,0)
            reward = torch.unsqueeze(reward,0)
            done = (done,)

        # 1. Predicted Q values with current state. As you know, we are training our DNN model such that we can get the best predictions for Q values in any state.
        
        pred = self.model(state) # Pred is the action-value for the 3 actions we will take.

        # 2. Q_new = r + gamma * max(next_predicted Q value)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]: # Any state where we find the game is not over, we will have a next state. Only for those state-action value pairs, we must update using Bellman's equation.
                Q_new += self.gamma * torch.max(self.model(next_state[idx]))
        
            target[idx][torch.argmax(action).item()] = Q_new
        
        self.optimizer.zero_grad() # This is a sanity check. It will initialize all the gradient values to zero.

        loss = self.criterion(target, pred) # Remember our loss function. This will calculate the loss/change between the previous values for Q for each state-action pair.
        # print(target,loss,action)
        loss.backward() # This will give a back propagate the error to previous neurons.

        self.optimizer.step() # This will update the parameters of our neurons.


