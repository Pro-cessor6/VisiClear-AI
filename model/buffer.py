from collections import deque

buffer = deque(maxlen=1000)

def add(sample):
    buffer.append(sample)

def sample(batch_size=64):
    return list(buffer)[-batch_size:]
