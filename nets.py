"""The two networks, the training that pulls matching pairs together, and the test that names a picture by reading captions."""
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import world

torch.set_num_threads(4)
DIM = 64
STEPS = 1500
FILES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'files')


class PictureNet(nn.Module):
    """A picture in, a list of `out` numbers out."""
    def __init__(self, out=DIM):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
                                  nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU())
        self.head = nn.Linear(128 * 16, out)

    def forward(self, x):                      # x: pictures, shape (n, 32, 32, 3)
        return self.head(self.conv(x.permute(0, 3, 1, 2)).flatten(1))


class TextNet(nn.Module):
    """A caption in, a list of `out` numbers out. A small Transformer, so word order can matter."""
    def __init__(self, out=DIM, width=64):
        super().__init__()
        self.word = nn.Embedding(len(world.WORDS), width, padding_idx=0)
        self.place = nn.Parameter(torch.randn(world.MAXLEN, width) * 0.02)
        layer = nn.TransformerEncoderLayer(width, 4, width * 2, dropout=0.0, batch_first=True)
        self.layers = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = nn.Linear(width, out)

    def forward(self, ids):                    # ids: word numbers, shape (n, 10), 0 = empty place
        pad = ids == 0
        h = self.layers(self.word(ids) + self.place, src_key_padding_mask=pad)
        h = (h * ~pad[..., None]).sum(1) / (~pad).sum(1, keepdim=True)
        return self.head(h)


class Clip(nn.Module):
    def __init__(self, temperature=0.07, learn_temperature=True):
        super().__init__()
        self.picture, self.text = PictureNet(), TextNet()
        self.scale = nn.Parameter(torch.tensor(float(np.log(1 / temperature))), requires_grad=learn_temperature)

    def see(self, imgs):
        return F.normalize(self.picture(imgs), dim=1)        # every list gets length 1

    def read(self, ids):
        return F.normalize(self.text(ids), dim=1)

    def table(self, imgs, ids):
        """How alike picture i and caption j are, for every i and j (from -1 to 1)."""
        return self.see(imgs) @ self.read(ids).T

    def loss(self, imgs, ids, hard=None, things=None):
        """Every picture must pick its own caption out of the batch, and every caption its own picture."""
        p, t = self.see(imgs), self.read(ids)
        scale = self.scale.exp().clamp(max=100)
        logits = scale * p @ t.T
        if things is not None:                 # two pairs of the same thing: do not count the other caption as wrong
            twin = (things[:, None] == things[None, :]) & ~torch.eye(len(things), dtype=torch.bool)
            logits = logits.masked_fill(twin, -1e9)
        want = torch.arange(len(imgs))
        if hard is not None:                   # one wrong caption made for each picture, added as extra columns
            rows = torch.cat([logits, scale * p @ self.read(hard).T], 1)
            return (F.cross_entropy(rows, want) + F.cross_entropy(logits.T, want)) / 2
        return (F.cross_entropy(logits, want) + F.cross_entropy(logits.T, want)) / 2


def T(a):
    return torch.from_numpy(a)


def train(model, imgs, caps, steps, batch=64, lr=1e-3, seed=0, every=0, probe=None, hard=None, things=None):
    """Plain training loop. `probe(model)` is called every `every` steps and its answers are returned."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam([q for q in model.parameters() if q.requires_grad], lr=lr)
    x, ids = T(imgs), T(world.encode(caps))
    hid = T(world.encode(hard)) if hard is not None else None
    th = T(np.asarray(things)) if things is not None else None
    log = []
    for step in range(steps + 1):
        if every and step % every == 0:
            model.eval()
            with torch.no_grad():
                log.append((step, *probe(model)))
            model.train()
        if step == steps:
            break
        i = torch.randperm(len(x), generator=g)[:batch]
        loss = model.loss(x[i], ids[i], hid[i] if hid is not None else None, th[i] if th is not None else None)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return log


@torch.no_grad()
def name(model, imgs, texts):
    """For every picture: the number of the caption in `texts` that lies closest."""
    return model.table(T(imgs), T(world.encode(texts))).argmax(1).numpy()


def prompts(combos, template=world.PROMPT):
    return [template.format(c, s) for c, s in combos]


def score(model, imgs, labels, combos=world.COMBOS, template=world.PROMPT):
    """Share of pictures whose closest caption, out of one caption for each thing in `combos`, is the right one."""
    pick = name(model, imgs, prompts(combos, template))
    return float(np.mean([world.COMBOS.index(combos[p]) == l for p, l in zip(pick, labels)]))


class Classifier(nn.Module):
    """The old way: a picture in, one score for each class on a fixed list out."""
    def __init__(self, classes):
        super().__init__()
        self.net = PictureNet(classes)

    def forward(self, x):
        return self.net(x)


def train_classifier(model, imgs, targets, steps, batch=64, lr=1e-3, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x, y = T(imgs), T(targets)
    for _ in range(steps):
        i = torch.randperm(len(x), generator=g)[:batch]
        loss = F.cross_entropy(model(x[i]), y[i])
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()


def numbers(model):
    return sum(q.numel() for q in model.parameters())


def keep(model, name):
    os.makedirs(FILES, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(FILES, name + '.pt'))


def fetch(name):
    """A trained model from step 4. Run step4_pull.py first."""
    model = Clip()
    model.load_state_dict(torch.load(os.path.join(FILES, name + '.pt')))
    return model.eval()


def save(name, data):
    os.makedirs(FILES, exist_ok=True)
    with open(os.path.join(FILES, name + '.json'), 'w') as f:
        json.dump(data, f)


def pc(x):
    return f'{100 * x:.1f} %'
