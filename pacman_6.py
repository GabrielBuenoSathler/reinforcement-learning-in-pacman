import numpy as np
from PIL import Image
import cv2
import torch 
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque  
import gym
import torch.nn.functional as F

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'done', 'next_state'))
ENV_NAME = 'MsPacman-v0'
class ObservationBuffer:
    def __init__(self, buffer_size=4):
        self.buffer_size = buffer_size
        self.buffer = []

    def add_observation(self, observation):
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)  # Remove the oldest observation if the buffer is full
        self.buffer.append(observation)

    def get_buffer(self):
        b = np.concatenate(self.buffer,-1)
        b = b.swapaxes(0,-1)
        return b
        # return self.buffer

class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), \
            np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), \
            np.array(next_states)

class FrameSkippingAgent:
    def __init__(self, env_name, skip_frames=4, buffer_capacity=10000):
        self.env = gym.make(env_name)
        self.skip_frames = skip_frames
        self.observation_buffer = ObservationBuffer(buffer_size=4)
        self.experience_buffer = ExperienceBuffer(capacity=buffer_capacity)

    @property
    def action_space(self):
        return self.env.action_space

    def preprocess_frame(self, frame):
        if isinstance(frame, tuple):
            frame_1, _ = frame
        else:
            frame_1 = frame

        img = np.reshape(frame_1, [210, 160, 3]).astype(np.float32)
        img = img[:, :, 0] * 0.229 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
        x_t = np.reshape(resized_screen, [84,84,1])
        x_t = (x_t - x_t.min()) / (x_t.max() - x_t.min())
        x_t = (x_t * 255.0).astype(np.uint8)

        return x_t

    def reset(self):
        state = self.env.reset()
        preprocessed_state = self.preprocess_frame(state)
        self.observation_buffer = ObservationBuffer(buffer_size=4)
        for _ in range(self.skip_frames):
            self.observation_buffer.add_observation(np.zeros(preprocessed_state.shape,preprocessed_state.dtype))
            ...
        self.observation_buffer.add_observation(preprocessed_state)
        return self.observation_buffer.get_buffer()

    def step(self, action):
        total_reward = 0
        current_state = self.observation_buffer.get_buffer()
        for _ in range(self.skip_frames):
            state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            self.observation_buffer.add_observation(self.preprocess_frame(state))
            if terminated:
                break

        # current_state = self.observation_buffer.get_buffer()
        next_state = self.observation_buffer.get_buffer()
        # Get the next state after one frame-skipped action
        # next_state, _, _, _, _ = self.env.step(action)
        # next_state = self.preprocess_frame(next_state)
        experience = Experience(state=current_state, action=action, reward=total_reward, done=terminated, next_state=next_state)

        self.experience_buffer.append(experience)

        return current_state, total_reward, terminated, truncated, info

class DuelingDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DuelingDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        conv_out_size = self._get_conv_out(input_shape)
        
        # Separate the network into two streams for value and advantage
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = x.float()
        x = x.to("cuda")

        # x = self.conv(x).view(x.size()[0], -1)
        x = self.conv(x)

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        # Combine value and advantage to get Q-values
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values
    
temp_env = gym.make(ENV_NAME)
online_net = DuelingDQN((4,84,84),temp_env.action_space.n).to("cuda")
target_net = DuelingDQN((4,84,84),temp_env.action_space.n).to("cuda")
class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def select_action_ddqn(self, net, state, epsilon=0.0, device="cuda"):
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
            q_value = 0.0  # Assign some default value when action is random
        else:
            state_a = np.array([state], copy=False)
            state_v = torch.tensor(state_a, dtype=torch.float32).to(device)

            # With the new modifications this is not needed
            # state_v = state_v.squeeze(0)

            q_vals_online = net(state_v)
            _, act_v_online = torch.max(q_vals_online, dim=1)

            # Use target network for Q-value estimation
            q_vals_target = target_net(state_v).detach()
            q_value = q_vals_target[0, act_v_online.item()]

            action = int(act_v_online.item())

        return action, q_value

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device="cuda"):
        done_reward = None

        action, q_value = self.select_action_ddqn(net, self.state, epsilon, device)

        new_state, reward, terminated, truncated, info = self.env.step(action)
        
        # self.state = np.moveaxis(self.state, -1, 0)
        # new_state = np.moveaxis(new_state, -1, 0)

        if isinstance(new_state, list):
            new_state = np.array(new_state)
        if isinstance(self.state, list):
            self.state = np.array(self.state)

        self.total_reward += reward

        exp = Experience(self.state, action, reward, terminated, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state

        if terminated:
            done_reward = self.total_reward
            self._reset()

        return done_reward


buffer_capacity = 100000
observation_buffer = ObservationBuffer(buffer_size=4)
experience_buffer = ExperienceBuffer(capacity=buffer_capacity)

def calc_loss(batch, net, tgt_net, device):
    states, actions, rewards, dones, next_states = batch

    
    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions, dtype=torch.int64).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    # state_action_values = net(states_v).gather(1, actions_v).squeeze(-1)

    with torch.no_grad():
        next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v

    # Use Huber loss instead of MSE loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    return loss


frame_skip_agent = FrameSkippingAgent(ENV_NAME)
experience_buffer = frame_skip_agent.experience_buffer
agent = Agent(frame_skip_agent, experience_buffer)


MAX_EPISODES = 5000  # Increase the number of episodes
MAX_STEPS_PER_EPISODE = 10000
BATCH_SIZE = 128
GAMMA = 0.99
LEARNING_RATE = 0.0001
TARGET_UPDATE = 1000
EPS_START = 1.0
EPS_END = 0.02
EPS_DECAY = MAX_EPISODES  # Adjusted based on 

optimizer = optim.Adam(online_net.parameters(), lr=LEARNING_RATE)

# Enable this only if you want to check if the bug is gone (faster that training)
experience_buffer = ExperienceBuffer(capacity=100000)
last_rewards = deque(maxlen=10)
# Training loop
for episode in range(MAX_EPISODES):
    frame_skip_agent.reset()
    total_reward = 0.0

    for step in range(MAX_STEPS_PER_EPISODE):
        # Play a step and store the experience

        # epsilon = max(EPS_END, EPS_START - episode / EPS_DECAY)
        epsilon = max(EPS_END, EPS_START - (EPS_START-EPS_END) * episode / EPS_DECAY )
        done_reward = agent.play_step(online_net, epsilon=epsilon, device="cuda")

        # NOTE: this is already down no need to repeat
        # if done_reward is not None:
        #     print(f"Episode {episode + 1}, Total Reward: {done_reward}")
        #     break

        # Sample a batch from the experience buffer and perform a Q-network update
        if len(experience_buffer) > BATCH_SIZE:
            batch_states, batch_actions, batch_rewards, batch_dones, batch_next_states = experience_buffer.sample(BATCH_SIZE)

            # SHOULD BE MOVED INSIDE CALC_LOSS
            # batch_states = torch.tensor(batch_states, dtype=torch.float32).to("cuda")
            # batch_actions = torch.tensor(batch_actions, dtype=torch.int64).to("cuda")
            # batch_rewards = torch.tensor(batch_rewards, dtype=torch.float32).to("cuda")
            # batch_dones = torch.BoolTensor(batch_dones).to("cuda")
            # batch_next_states = torch.tensor(batch_next_states, dtype=torch.float32).to("cuda")

            optimizer.zero_grad()
            loss = calc_loss((batch_states, batch_actions, batch_rewards, batch_dones, batch_next_states), online_net, target_net, device="cuda")
            loss.backward()
            optimizer.step()

        # Update target network periodically
        if step % TARGET_UPDATE == 0:
            target_net.load_state_dict(online_net.state_dict())

        # Check if the episode is done
        if done_reward is not None:
            last_rewards.append(done_reward)
            total_rewards_mean = np.mean(last_rewards)
            print(f"Episode {episode + 1}, Total Reward: {done_reward:0.1f} , Mean Reward: {total_rewards_mean:0.1f} , Epsilon: {epsilon:0.3f}")
            break

torch.save(online_net.state_dict(), 'models/final_online_net.pth')        
