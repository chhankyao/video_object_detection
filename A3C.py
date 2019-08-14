import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

import os
import cv2
import random
import numpy as np


from trackers.kcftracker import *
from trackers.siamfc import *
from yolov3.models import *
from yolov3.utils.utils import *
from yolov3.utils.datasets import *
from utils_A3C import *
from pointnet import *
from VOD import *

os.environ["OMP_NUM_THREADS"] = "4"

GAMMA = 0.9
MAX_EP = 20000
UPDATE_GLOBAL_ITER = 100

N_S = 20
N_A = 2



class Net(nn.Module):
    def __init__(self, s_dim, a_dim, device):
        super(Net, self).__init__()
        self.device = device
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.pi = PointNetCls(d=s_dim, k=a_dim, feature_transform=False).to(device)
        self.v = PointNetCls(d=s_dim, k=1, feature_transform=False).to(device)
        '''self.pi1 = nn.Linear(s_dim, 50)
        self.pi2 = nn.Linear(50, 50)
        self.pi3 = nn.Linear(50, a_dim)
        self.v1 = nn.Linear(s_dim, 50)
        self.v2 = nn.Linear(50, 50)
        self.v3 = nn.Linear(50, 1)
        set_init([self.pi1, self.pi2, self.pi3, self.v1, self.v2, self.v3])'''
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        x = torch.FloatTensor(x).view(-1, self.s_dim, 10).to(self.device)
        logits, _, _ = self.pi(x)
        values, _, _ = self.v(x)
        '''pi1 = F.relu6(self.pi1(x))
        pi2 = F.relu6(self.pi2(pi1))
        logits = self.pi3(pi2)
        v1 = F.relu6(self.v1(x))
        v2 = F.relu6(self.v2(v1))
        values = self.v3(v2)'''
        return logits, values

    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)
        
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss


    
class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(N_S, N_A, torch.device("cpu"))
        data_dir = 'data/ILSVRC2015/Data/VID/train/'
        train_list = 'data/ILSVRC2015/ImageSets/VID/my_train.txt'
        self.env = VOD(scheduler='a3c', detect_interval=20)
        self.env.train_init(data_dir, train_list)

    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            if self.name == 'w0':
                print('========== ep {} =========='.format(self.g_ep.value))
            while True:
                a = self.lnet.choose_action(s)
                s_, r, done = self.env.step(a)
                if self.name == 'w0':
                    np.set_printoptions(precision=2)
                    print(s.reshape(N_S, 10)[:,0], a, np.array([r]))
                    
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1
        self.res_queue.put(None)


        
if __name__ == "__main__":
    gnet = Net(N_S, N_A, torch.device("cpu"))        # global network
    gnet.share_memory()                              # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=0.0001)
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    
    res = []
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
            if global_ep.value % 50 == 0:
                print('ep {}: reward = {}'.format(global_ep.value, r))
                torch.save(gnet.state_dict(), 'models/A3C_detect3.pth')
                np.save('models/rewards_detect3', np.array(res))
        else:
            break
            
    [w.join() for w in workers]
    torch.save(gnet.state_dict(), 'models/A3C_detect3.pth')
    np.save('models/rewards_detect3', np.array(res))
    