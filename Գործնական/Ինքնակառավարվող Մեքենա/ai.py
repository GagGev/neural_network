# Կցում ենք բոլոր անհրաժեշտ գրադարանները
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


# Ստեղծում ենք նեյրոնային ցանցի մոդելը
class Network(nn.Module):
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)  # մուտքային շերտ
        self.fc2 = nn.Linear(30, 50)  # 30 երկարությամբ թաքնված շերտ
        self.fc3 = nn.Linear(50, nb_action)  # 50 երկարությամբ թաքնված շերտ
    # Ստեղծում ենք ֆունկցիա, որը կանցկացնի տվյալները նեյրոնային ցնացի միջով և կտպի Q-արժեքները
    def forward(self, state):
        x = F.softmax(self.fc1(state))  # առաջին ակտիվացման ֆունկցիան՝ softmax
        y = F.relu(self.fc2(x))  # երկրորդ ակտիվացման ֆունկցիան՝ ReLU
        q_values = self.fc3(y)  # Ստանում ենք Q-արժեքները
        return q_values


# Ռեալիզացնում ենք Experience Replay-ը
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity  # Հիշողության չափը
        self.memory = []  # Հիշողությունը
    # Ստեղծում ենք ֆունկցիա, որը կգցի տեղեկություն հիշողության մեջ
    def push(self, event):
        self.memory.append(event)  # գցումն ենք հիշողության մեջ
        if len(self.memory) > self.capacity:  # եթե չափից երկար է
            del self.memory[0]  # ջնջում ենք առաջին էլեմենտը
    # Ֆունկցիա, որը կվերցնի պատահական խումբ հիշողությունից
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))  # պատահականորեն խառնում ենք
        return map(lambda x: Variable(torch.cat(x, 0)), samples)  # կիրառում ենք հատուկ torch-ի ֆունկցիաներից մեկը


# Ռեալիզացնում ենք Deep Q-learning-ը
class Dqn:
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma  # Գամմա պարամետրը
        self.reward_window = []  # Ստացած վերժին մի քանի մրցանակներ
        self.model = Network(input_size, nb_action)  # Հայտարարում ենք մեր նեյրոնային ցանցը
        self.memory = ReplayMemory(100000)  # Հայտարարում ենք մեր հիշողությունը
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)  # օգտագործելու ենք AdaM օպտիմայզերը
        self.last_state = torch.Tensor(input_size).unsqueeze(0)  # պահում ենք վերջին վիճակը,
        self.last_action = 0  # վերջին գործողությունը
        self.last_reward = 0  # և վերջին մրցանակը
    # Ֆունկցիա, որը կաշխատեցնի նեյրոնային ցանցը և կորոշի, թե ինչ գործողություն ենք կատարելու
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile=True)) * 100)  # օգտագործում ենք Softmax-ը
        action = probs.multinomial(num_samples=1)  # torch գրադարանի հատուկ ֆունկցիայի շնորհիվ ստանում ենք արժեքը
        return action.data[0, 0]  # վերադարձնում ենք կանխատեսված լավագույնը
    # Ֆունկցիա, որը հետ կգնա նեյրոնային ցանցով և կթարմացնի կշիռները
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma * next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_graph=True)
        self.optimizer.step()
    # Ֆունկցիա, որը ստանում է անցած գործողության մրցանակը, օգտագործում է և տալիս տվյալներ դրա մասին
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push(
            (self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    # Մի քանի օգտագործվող ֆունկցիա
    def score(self):
        return sum(self.reward_window) / (len(self.reward_window) + 1.)

    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    }, 'last_brain.pth')
        print("Saved!")

    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
