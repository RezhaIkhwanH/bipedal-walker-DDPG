# from cgi import test
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import gym
import os 

# Hyperparameters
env = gym.make("BipedalWalker-v3")
state_dim = 24  # Example: Flattened RGB image
action_dim = 4 # Example: 3-dimensional continuous action
batch_size = 100
gamma = 0.99
tau = 0.005
lr=0.00025
capacity = 20000

# Actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        x = self.max_action * torch.tanh(self.layer3(x))
        return x

# Critic network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 1)

    def forward(self, state, action):
        # state = state.view(state.size(0), -1) 
        # action = action.view(action.size(0), -1)
        x = torch.cat([state, action], 1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# Replay buffer
# class ReplayBuffer:
#     def __init__(self, capacity):
#         self.buffer = deque(maxlen=capacity)

#     def push(self, state, action, reward, next_state, done):
#         state = np.expand_dims(state, 0)
#         next_state = np.expand_dims(next_state, 0)
#         self.buffer.append((state, action, reward, next_state, done))

#     def sample(self, batch_size):
#         batch = random.sample(self.buffer, batch_size)
#         state, action, reward, next_state, done = map(np.stack, zip(*batch))
#         return state, action, reward, next_state, done


class ReplayMemory(object):

    def __init__(self, capacity):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.capacity = capacity
        self.memory = [] # daftar experience / pengalaman AI

    # fungsi yang akan menambakan data ke dalam memory
    def push(self, event): # event = pengalaman yang akan ditambah ke memory
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    #fungsi yang membuat sample untuk di train
    def sample(self,batch_size):
        experiences= random.sample(self.memory,k=batch_size)
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(self.device) # stak dari kondisi saat ini
        action= torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).float().to(self.device) # stak dari aksi nya
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.device) # array / stak hadiah
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(self.device) # stak dari kondisi  selanjutnya
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device) # stak dari done
        return states,next_states,action,rewards,dones
    
    
# DDPG Agent
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr=lr)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(),lr=lr)

        self.replay_buffer = ReplayMemory(capacity=capacity)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.count=0

    def select_action(self, state,epsilon=0.):
        
        if random.random() > epsilon:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.actor.eval()
            with torch.no_grad():
                action_values = self.actor(state).cpu().data.numpy().flatten()#mungkin butuh flatern
            self.actor.train()
            return action_values
        else:
        #ini untuk explorasi dimana aksi ditentukan benar benar random tanpa perhitungan
            return np.random.uniform(-1, self.max_action, size=4)
    
    def step(self,state, action, reward, next_state, done):
        self.replay_buffer.push([state, action, reward, next_state, done])
        self.count=(self.count+1) % 10
        if self.count==0:
            self.train(batch_size)

    def train(self, batch_size):
        if len(self.replay_buffer.memory) < batch_size:
            return

        state,next_state, action, reward, done = self.replay_buffer.sample(batch_size)

        next_action = self.actor_target(next_state)
        target_q = self.critic_target(next_state, next_action).detach()
        target_q = reward + (1 - done) * gamma * target_q
        #perbedaan bentuk q target 
        current_q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        policy_loss = -self.critic(state, self.actor(state)).mean()#adalah nilai rata-rata negatif dari Q-value yang diperkirakan oleh model kritik untuk tindakan yang diprediksi oleh model aktor pada keadaan saat ini

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def save_model(self, checkpoint_path='ddpg_checkpoint.pth'):
        # Menyimpan model dan lainnya
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            # Masukkan parameter lain yang ingin Anda simpan
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Model saved to {checkpoint_path}")

    def load_model(self, checkpoint_path='ddpg_checkpoint.pth'):
        # Memuat model dan lainnya
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            # Muat parameter lain yang Anda simpan
            print(f"Model loaded from {checkpoint_path}")
        else:
            print("Checkpoint not found.")


# Initialize DDPG agent
agent = DDPGAgent(state_dim, action_dim, max_action=1.0)

#seting
episodes = 2000
epsilon_starting_value  = 1.0 #epsinon awal 1 = exploration
epsilon_ending_value  = 0.01 #end number =0.01= exploitasi
epsilon_decay_value  = 0.9995 #number yang akan di kali epsilon unutk mengurangin nya secara bertahap asli0.995
epsilon = epsilon_starting_value #pengaturan explorasi / exploitasi
scores_on_100_episodes = deque(maxlen = 100)
hig_score=0
# Training loop
# for episode in range(1,episodes+1):
#     state,_ = env.reset()
#     score=0
#     for step_perEps in range(1000):
#         action = agent.select_action(state,epsilon)
#         next_state, reward, done, _ ,_= env.step(action)

       
#         agent.step(state, action, reward, next_state, done)
#         state = next_state
#         score+=reward

#         if done:
#             # print([next_state, reward, done ,step_perEps])
#             break
#     epsilon = max(epsilon_ending_value, epsilon_decay_value * epsilon)
#     scores_on_100_episodes.append(score)
#     if hig_score < np.array(scores_on_100_episodes).mean():
#         hig_score=np.array(scores_on_100_episodes).mean()
#         agent.save_model()
#     print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)), end = "")
#     if episode % 100 == 0:
#         print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)))
#     if np.array(scores_on_100_episodes).mean() >= 5.0:
#         print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode - 100, np.mean(scores_on_100_episodes)))
#         agent.save_model()
#         break
    


# visual 
import glob
import io
import base64
import imageio
# from IPython.display import HTML, display
from gym.wrappers.monitoring.video_recorder import VideoRecorder

# agent.load_model("ddpg_checkpoint (9).pth")
i = 0
def show_video_of_model(agent, env_name):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    step=1
    frames = []
    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.select_action(state)
        state, reward, done, _, _ = env.step(action)
        if step==5000:
            break
        step+=1
    env.close()
    imageio.mimsave('video(test).mp4', frames, fps=30)


agent.load_model('bobotDDPG.pth')
show_video_of_model(agent, 'BipedalWalker-v3')

# def show_video():
#     mp4list = glob.glob('*.mp4')
#     if len(mp4list) > 0:
#         mp4 = mp4list[0]
#         video = io.open(mp4, 'r+b').read()
#         encoded = base64.b64encode(video)
#         display(HTML(data='''<video alt="test" autoplay
#                 loop controls style="height: 400px;">
#                 <source src="data:video/mp4;base64,{0}" type="video/mp4" />
#              </video>'''.format(encoded.decode('ascii'))))
#     else:
#         print("Could not find video")

# show_video()