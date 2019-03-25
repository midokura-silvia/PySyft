import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import syft as sy
from syft.codes import MSGTYPE
from syft.workers import WebsocketClientWorker
from syft.workers import WebsocketServerWorker
from multiprocessing import Process
import threading
import asyncio

import numpy as np
from collections import ChainMap as merge
import binascii

class SimpleNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(PolicyNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.tanh(self.linear3(x))
        return x

    def get_action(self, state):
        state  = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0, 0]


def start_proc_compute(kwargs):
    def target():
        client = WebsocketClientWorker(**kwargs)
        client.ready_to_compute()
    p = Process(target=target)
    p.start()
    return p


hook = sy.TorchHook(torch)
new_loop = asyncio.new_event_loop()
kwargs = {"id": "federator", "host": "localhost", "port": 8765, "hook": hook}
server = WebsocketServerWorker(**merge({ 'loop': new_loop}, kwargs))

def go(loop):
    asyncio.set_event_loop(loop)
    server.start()
t = threading.Thread(target=go, args=(new_loop,))
t.start()
time.sleep(.1)
start_proc_compute(kwargs)
