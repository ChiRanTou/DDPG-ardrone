import random
import numpy as np
from collections import deque

BUFFER_SIZE = 20000
BATCH_SIZE = 64

# class Replay_Buffer(object):
# 	def __init__(self, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE):
# 		self.buffer_size = buffer_size
# 		self.batch_size = batch_size
# 		self.buffer = deque([], maxlen=self.buffer_size)
#
# 	def add(self, s, a, r, s2, done):
# 		exp = (s, a, r, s2, done)
# 		self.buffer.append(exp)
#
# 	def size(self):
# 		return len(self.buffer)
#
# 	def sample_batch(self):
# 		batch = []
#
# 		if len(self.buffer) < self.batch_size:
# 			batch = random.sample(list(self.buffer), len(self.buffer))
# 		else:
# 			batch = random.sample(list(self.buffer), self.batch_size)
#
# 		return batch
#
# 	def clear_buffer(self):
# 		self.buffer.clear()

class Replay_Buffer(object):

    def __init__(self, buffer_size=BUFFER_SIZE):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def sample_batch(self, batch_size=BATCH_SIZE):
        # Randomly sample batch_size examples
        #return random.sample(self.buffer, batch_size)

        if len(self.buffer) < batch_size:
            batch = random.sample(list(self.buffer), len(self.buffer))
        else:
            batch = random.sample(list(self.buffer), batch_size)

        return batch

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0