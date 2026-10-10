"""Quick checks of the parts. Runs in a few seconds: python3 tests.py"""
import math
import numpy as np
import torch
import nets
import world

rng = np.random.default_rng(0)

# The world: 36 things, 6 kept away, pictures are 32 by 32 by 3 numbers between 0 and 1.
assert len(world.COMBOS) == 36 and len(world.HELD) == 6 and len(world.SEEN) == 30
assert not set(world.HELD) & set(world.SEEN)
assert len({c for c, s in world.HELD}) == 6 and len({s for c, s in world.HELD}) == 6     # every colour and every shape is still shown
img = world.picture('red', 'circle', rng)
assert img.shape == (32, 32, 3) and 0 <= img.min() and img.max() <= 1
assert img[..., 0].mean() > img[..., 2].mean() - 0.2

# Captions: every word has a number, and the numbers give the words back.
cap = world.caption('blue', 'ring', rng)
ids = world.encode([cap])[0]
assert ' '.join(world.WORDS[i] for i in ids if i) == cap and ids.shape == (world.MAXLEN,)
imgs, caps, labels = world.pairs(world.SEEN, 2, 0)
assert len(imgs) == 60 and all(f'{world.COMBOS[l][0]} {world.COMBOS[l][1]}' in c for c, l in zip(caps, labels))

# Two things: the caption and its swapped twin use the same words in another order.
x, right, swapped = world.scene_pairs(20, 0)
assert all(sorted(r.split()) == sorted(w.split()) or {'left', 'right'} & set(r.split()) != {'left', 'right'} & set(w.split()) for r, w in zip(right, swapped))
assert all(r != w for r, w in zip(right, swapped))

# The two networks give lists of the same length, scaled to length 1.
torch.manual_seed(0)
model = nets.Clip().eval()
with torch.no_grad():
    p, t = model.see(nets.T(imgs)), model.read(nets.T(world.encode(caps)))
    assert p.shape == t.shape == (60, nets.DIM)
    assert torch.allclose(p.norm(dim=1), torch.ones(60), atol=1e-5) and torch.allclose(t.norm(dim=1), torch.ones(60), atol=1e-5)
    table = model.table(nets.T(imgs), nets.T(world.encode(caps)))
    assert table.shape == (60, 60) and table.abs().max() <= 1 + 1e-5
    # Before training the miss is near the miss of a blind guess among 60: ln 60.
    miss = model.loss(nets.T(imgs), nets.T(world.encode(caps))).item()
    assert abs(miss - math.log(60)) < 1.5, miss
    # Not counting captions of the same thing as wrong can only lower the miss.
    assert model.loss(nets.T(imgs), nets.T(world.encode(caps)), things=nets.T(labels)).item() <= miss + 1e-6
    # One more wrong caption for every picture can only raise it.
    assert model.loss(nets.T(x), nets.T(world.encode(right)), hard=nets.T(world.encode(swapped))).item() >= model.loss(nets.T(x), nets.T(world.encode(right))).item() - 1e-6

# A few steps of training lower the miss.
small = nets.Clip()
a = small.loss(nets.T(imgs), nets.T(world.encode(caps))).item()
nets.train(small, imgs, caps, 30, batch=32)
with torch.no_grad():
    assert small.loss(nets.T(imgs), nets.T(world.encode(caps))).item() < a
print('all checks passed')
