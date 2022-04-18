from torch import nn as nn 
import torch 
import numpy as np 
import gym 
import argparse
from torch.utils.tensorboard import SummaryWriter
import ptan
import torch.optim as optim

# DEFINING HYPERPARAMETERS
GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 128
NUM_ENVS = 50
REWARD_STEPS = 4 # how many steps ahead to approximate discounted reward for each action
CLIP_GRAD = 0.1

# A2C NETWORK with actor and critic outputs 
class A2C(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(A2C, self).__init__()

        # Defining CNN layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        # Defining the policy network (actor)
        # Defines probability distribution over actions
        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        # Defining the value network (critic)
        # Returns single number for state value
        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1))
    
    def _get_conv_out(self, shape):
        """
        Function that obtains the shape of the CNN layers 
        """
        o = self.conv_layers(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        """
        Function returns two values : policy and value 
        """
        out = x.float() / 256
        conv_out = self.conv_layers(out).view(out.size()[0], -1)
        return self.policy(conv_out), self.value(conv_out)


# Batch functions

def unpack_batch_fn(batch, net, device='cpu'):
    """
    Obtains batch of environment transitions (action, reward, states)

    Returns:
    ----------
    batch_states: 
    batch_actions
    batch_qvals: 

    Notes:
    ---------
    Q_value computed using formula Q = sum(lambda^i*r_i + lambda^*V(s_n))
    Q value used to calculate MSE and Advantage of action

    """
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []

    for idx, exp in enumerate(batch):

        #Append states, actions and rewards from experience source firstlast class

        states.append(np.array(exp.state, copy=False))
        actions.append(int(exp.action))
        rewards.append(exp.reward)

        # Handle episode ending transitions, remember index 
        # of batch entry for non-terminal episodes
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(
                np.array(exp.last_state, copy=False))
    
    # Convert states into pytorch tensor and copy to gpu
    batch_states = torch.FloatTensor(
        np.array(states, copy=False)).to(device)
    batch_actions = torch.LongTensor(actions).to(device)

    # Compute Q-values
    rewards_np = np.array(rewards, dtype=np.float32)

    # for non-terminal episodes, obtain value estimation for state
    # For terminal episodes : reward value already contains
    # discounted reward for REWARD_STEPS. However, for non-terminal
    # we need to compute estimate of this 

    # Prepares variable with the last state and queries network for V(s) approximation
    # Value is multipled by discount factor, and added to immediate rewards
    if not_done_idx:
        last_states_v = torch.FloatTensor(
            np.array(last_states, copy=False)).to(device)
        last_vals_v = net(last_states_v)[1]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        last_vals_np *= GAMMA ** REWARD_STEPS
        rewards_np[not_done_idx] += last_vals_np
    
    # Computes reward 
    batch_reward = torch.FloatTensor(rewards_np).to(device)
    return batch_states, batch_actions, batch_reward

if __name__ == "__main__":

    # Argument parser: cuda and name of run 
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False,
                        action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", required=True,
                        help="Name of the run")
    args = parser.parse_args()#
    device = torch.device("cuda" if args.cuda else "cpu")

    # Make environment wrapper function
    make_env = lambda: ptan.common.wrappers.wrap_dqn(
        gym.make("PongNoFrameskip-v4"))

    # Run multiple environments 
    envs = [make_env() for _ in range(NUM_ENVS)]

    # Run summarywriter function 
    writer = SummaryWriter(comment="-pong-a2c_" + args.name)

    ## TRAINING LOOP 

    # Move network to device
    net = A2C(envs[0].observation_space.shape,
                   envs[0].action_space.n).to(device)
    print(net)

    ##  PTAN FUNCTIONS 

    # Define agent to collect experience from agent 
    agent = ptan.agent.PolicyAgent(
        lambda x: net(x)[0], apply_softmax=True, device=device)
    
    # Define experience source to collect experience from 
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS)
    
    # Define optimiser for training loop

    # eps : defined as 1e-3 as 1e-8 too small. 
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE,
                           eps=1e-3)

    # Collect batch of experiences 
    batch = []
    
    #RewardTRacker is wrapper functiont o compute mean reward for last 100 episodes,
    # and tells us whether reward exceeds desired threshold
    with common.RewardTracker(writer, stop_reward=18) as tracker:
        
        # TBMeanTracker : writes mean parameters for last 10 steps
        with ptan.common.utils.TBMeanTracker(writer,
                batch_size=10) as tb_tracker:
            
            for step_idx, exp in enumerate(exp_source):
                batch.append(exp)
                new_rewards = exp_source.pop_total_rewards()
                
                if new_rewards:
                    if tracker.reward(new_rewards[0], step_idx):
                        break
                if len(batch) < BATCH_SIZE:
                    continue
        
            states_v, actions_t, vals_ref_v = unpack_batch_fn(batch, net, device=device)
            batch.clear()

            optimizer.zero_grad()

            # Obtain prediction of action and value
            logits_v, value_v = net(states_v)

        ## Compute loss functions 

            value_loss = F.mse_loss(value_v.squeeze(-1), vals_ref_v)
            
            # Probability of each action
            log_prob_v = F.log_softmax(logits_v, dim=1)

            # Compute difference between value q value and value (Q(s,a)- V(s))

            # detach is used to make sure POLICY gradient is not propagated to value approximation 
            advantage = vals_ref_v - value_v.detach() 

            # log probability of actions taken, scale with advantage 
            log_p_a = log_prob_v[range(BATCH_SIZE), actions_t]
            log_prob_actions_v = advantage * log_p_a #advatnage * log(p(s,a))
            policy_loss = -log_prob_actions_v.mean()

            # Entropy loss : equal to scaled entropy of policy
            # h(pi) = -sum(pi*log(pi))

            prob_v = F.softmax(logits_v, dim=1)
            ent = (prob_v * log_prob_v).sum(dim=1).mean()
            entropy_loss_v = ENTROPY_BETA * ent

            # Calculate gradients from policy 
            policy_loss.backward(retain_graph=True)
            
            # Compute gradients to track maximum gradient and variance
            grads = np.concatenate([
                    p.grad.data.cpu().numpy().flatten()
                    for p in net.parameters()
                    if p.grad is not None
            ])


            # calculate gradients of value and entropy loss
            loss_v = entropy_loss_v + value_loss
            loss_v.backward()
            nn_utils.clip_grad_norm_(net.parameters(),
                                        CLIP_GRAD)
            optimizer.step()

            # Add up policy, entropy and value losses together 
            loss_v += policy_loss


            # Compute losses 
            tb_tracker.track("advantage", advantage, step_idx)
            tb_tracker.track("values", value_v, step_idx)
            tb_tracker.track("batch_rewards",vals_ref_v,
                                 step_idx)
            tb_tracker.track("loss_entropy", entropy_loss_v,
                                 step_idx)
            tb_tracker.track("loss_policy", policy_loss,
                            step_idx)
            tb_tracker.track("loss_value", value_loss,
                            step_idx)
            tb_tracker.track("loss_total", loss_v, step_idx)
            g_l2 = np.sqrt(np.mean(np.square(grads)))
            tb_tracker.track("grad_l2", g_l2, step_idx)
            g_max = np.max(np.abs(grads))
            tb_tracker.track("grad_max", g_max, step_idx)
            g_var = np.var(grads)
            tb_tracker.track("grad_var", g_var, step_idx)

