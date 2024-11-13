""" TODO: module docstring for dgi/execute.py. """
import argparse
import torch
from torch import nn
import torch_geometric as tg


from .models import DGI, LogReg
from .utils.loss import DGILoss


parser = argparse.ArgumentParser(description="DGI test script")
parser.add_argument("--datapath", default="/tmp/cora")
args = parser.parse_args()

DATASET = "Cora"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

loss_fun = DGILoss()

# training params
BATCH_SIZE = 1
NB_EPOCHS = 10000
PATIENCE = 20
LR = 0.001
L2_COEF = 0.0
DROP_PROB = 0.0
HID_UNITS = 512
SPARSE = True
NONLINEARITY = "prelu"  # special name to separate parameters

data = tg.datasets.Planetoid(name=DATASET, root=args.datapath)[0]
data = data.to(device)
r_sum = data.x.sum(dim=1)
r_sum[r_sum == 0] = 1.0  # avoid division by zero
data.x /= r_sum[:, None]

# adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)
# features, _ = process.preprocess_features(features)

nb_nodes = data.num_nodes
ft_size = data.num_features
nb_classes = data.y.max().item() + 1

# adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))

# if sparse:
#     sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
# else:
#     adj = (adj + sp.eye(adj.shape[0])).todense()

# features = torch.FloatTensor(features[np.newaxis])
# if not sparse:
#     adj = torch.FloatTensor(adj[np.newaxis])
# labels = torch.FloatTensor(labels[np.newaxis])
# idx_train = torch.LongTensor(idx_train)
# idx_val = torch.LongTensor(idx_val)
# idx_test = torch.LongTensor(idx_test)

model = DGI(ft_size, HID_UNITS, NONLINEARITY)
model = model.to(device)

optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=L2_COEF)

STATE_DICT = "best_dgi.pkl"

xent = nn.CrossEntropyLoss()
# pylint: disable=invalid-name
cnt_wait = 0
best = 1e9
best_t = 0
for epoch in range(NB_EPOCHS):
    model.train()
    optimiser.zero_grad()
    loss = loss_fun(model, data)

    print("Loss:", loss)

    if loss < best:
        best = loss
        best_t = epoch
        cnt_wait = 0
        torch.save(model.state_dict(), STATE_DICT)
    else:
        cnt_wait += 1

    if cnt_wait == PATIENCE:
        print("Early stopping!")
        break

    loss.backward()
    optimiser.step()
# pylint: enable=invalid-name

print(f"Loading {best_t}th epoch")
model.load_state_dict(torch.load(STATE_DICT))

embeds = model.embed(data)
train_embs = embeds[data.train_mask]
val_embs = embeds[data.val_mask]
test_embs = embeds[data.test_mask]
#
train_lbls = data.y[data.train_mask]
val_lbls = data.y[data.val_mask]
test_lbls = data.y[data.test_mask]

tot = torch.zeros(1, device=device)
accs = []

for _ in range(50):
    log = LogReg(HID_UNITS, nb_classes).to(device)

    opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)

    best_acc = torch.zeros(1, device=device)

    for _ in range(100):
        log.train()
        opt.zero_grad()

        logits = log(train_embs)
        loss = xent(logits, train_lbls)

        loss.backward()
        opt.step()

    logits = log(test_embs)
    preds = torch.argmax(logits, dim=1)
    acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
    accs.append(acc * 100)
    print(acc)
    tot += acc

print("Average accuracy:", tot / 50)

accs = torch.stack(accs)
print(accs.mean())
print(accs.std())
