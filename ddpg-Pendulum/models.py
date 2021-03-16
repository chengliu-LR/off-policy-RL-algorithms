import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bounds):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, action_dim),
                        nn.Tanh()
                        )
        self.action_bounds = action_bounds

    def forward(self, state):
        return self.actor(state) * self.action_bounds

# critic network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
                        nn.Linear(state_dim + action_dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, 1)
                        #nn.Sigmoid()
                        )
    
    def forward(self, state, action):
        return self.critic(torch.cat([state, action], 1))

# ddpg policy gradient algorithm
class DDPG():
    def __init__(self, state_dim, action_dim, action_bounds, lr):
        self.actor = Actor(state_dim, action_dim, action_bounds).to(device)
        self.actor_target = Actor(state_dim, action_dim, action_bounds).to(device)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)

        # copy parameters from actor/critic network to target network
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.mseLoss = nn.MSELoss()
        # reward discount
        self.gamma = 0.95
        # target network update rate
        self.tau = 1e-2

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1,-1)).to(device)
        return self.actor(state).detach().cpu().data.numpy().flatten()
    
    def update(self, buffer, n_iter, batch_size):

        for i in range(n_iter):
            # sample a batch of transitions from replay buffer
            state, action, reward, next_state, done = buffer.sample(batch_size)
            
            # convert numpy arrays into torch tensors
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action).to(device)
            reward = torch.FloatTensor(reward).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            done = torch.FloatTensor(done).to(device)

            # select next action
            next_action = self.actor_target(next_state).detach()

            # compute target Q-value
            target_Q = self.critic_target(next_state, next_action).detach()
            target_Q = reward + (1-done)*self.gamma * target_Q

            # update critic
            critic_loss = self.mseLoss(self.critic(state, action), target_Q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # update actor
            actor_loss = -self.critic(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # gradually update target network
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
            
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
    
    def save(self, directory, name):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, name))
        torch.save(self.critic.state_dict(), '%s/%s_crtic.pth' % (directory, name))
        
    def load(self, directory, name):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, name), map_location='cpu'))
        self.critic.load_state_dict(torch.load('%s/%s_crtic.pth' % (directory, name), map_location='cpu'))