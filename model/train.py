import torch
import torch.nn as nn
import torch.optim as optim

loss_fn = nn.BCELoss()

def train(model, optimizer, batch):

    if len(batch) < 10:
        return

    X = torch.tensor([b[0] for b in batch])
    y = torch.tensor([b[1] for b in batch]).float().unsqueeze(1)

    pred = model(X)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

#i like trains
