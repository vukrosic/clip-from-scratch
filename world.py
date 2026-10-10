"""The small world of this project: pictures of coloured shapes and one-line captions for them. Nothing is downloaded."""
import numpy as np

SIZE = 32
COLOURS = {'red': (0.86, 0.20, 0.20), 'green': (0.20, 0.68, 0.30), 'blue': (0.20, 0.36, 0.86),
           'yellow': (0.92, 0.80, 0.18), 'purple': (0.60, 0.28, 0.76), 'pink': (0.95, 0.50, 0.72)}
SHAPES = ['circle', 'square', 'triangle', 'cross', 'ring', 'diamond']
CNAMES = list(COLOURS)
COMBOS = [(c, s) for c in CNAMES for s in SHAPES]                     # 36 things
# Six things the training never shows: one for every colour and one for every shape.
HELD = [(CNAMES[i], SHAPES[(i + 2) % 6]) for i in range(6)]
SEEN = [k for k in COMBOS if k not in HELD]
TEMPLATES = ['a {} {}', 'the {} {}', 'a picture of a {} {}', 'this is a {} {}', 'one {} {}']
PROMPT = 'a picture of a {} {}'
WORDS = ['<pad>'] + sorted(set(' '.join(TEMPLATES).replace('{}', '').split()) | set(CNAMES) | set(SHAPES) | {'left', 'right', 'of'})
WID = {w: i for i, w in enumerate(WORDS)}
MAXLEN = 10


def mask(shape, cx, cy, r):
    """Which pixels belong to the shape. (cx, cy) is its middle, r its half size."""
    y, x = np.mgrid[0:SIZE, 0:SIZE]
    dx, dy = x - cx, y - cy
    d = np.hypot(dx, dy)
    if shape == 'circle':
        return d <= r
    if shape == 'square':
        return np.maximum(abs(dx), abs(dy)) <= r * 0.85
    if shape == 'triangle':
        return (dy <= r * 0.8) & (dy >= -r) & (abs(dx) <= (dy + r) * 0.62)
    if shape == 'cross':
        return (np.maximum(abs(dx), abs(dy)) <= r) & ((abs(dx) <= r * 0.34) | (abs(dy) <= r * 0.34))
    if shape == 'ring':
        return (d <= r) & (d >= r * 0.56)
    if shape == 'diamond':
        return abs(dx) + abs(dy) <= r * 1.15
    raise ValueError(shape)


def background(rng):
    g = rng.uniform(0.82, 1.0)
    return np.clip(g + rng.normal(0, 0.012, 3), 0, 1) * np.ones((SIZE, SIZE, 3))


def paint(img, colour, shape, cx, cy, r, rng):
    rgb = np.clip(np.array(COLOURS[colour]) + rng.normal(0, 0.05, 3), 0, 1)
    img[mask(shape, cx, cy, r)] = rgb


def picture(colour, shape, rng):
    """One thing somewhere in the picture, with a little noise on every pixel."""
    img = background(rng)
    r = rng.uniform(5, 10)
    cx, cy = rng.uniform(r + 1, SIZE - r - 2, 2)
    paint(img, colour, shape, cx, cy, r, rng)
    return np.clip(img + rng.normal(0, 0.03, img.shape), 0, 1).astype(np.float32)


def scene(a, b, rng):
    """Two things: a on the left, b on the right."""
    img = background(rng)
    for (colour, shape), lo in ((a, 0), (b, 16)):
        r = rng.uniform(4, 6.5)
        paint(img, colour, shape, lo + rng.uniform(r + 0.5, 15.5 - r), rng.uniform(r + 1, SIZE - r - 2), r, rng)
    return np.clip(img + rng.normal(0, 0.03, img.shape), 0, 1).astype(np.float32)


def caption(colour, shape, rng):
    return TEMPLATES[rng.integers(len(TEMPLATES))].format(colour, shape)


def scene_caption(a, b, rng=None):
    """'the red circle is left of the blue square', or the same fact said from the other side."""
    if rng is not None and rng.random() < 0.5:
        return 'the {} {} is right of the {} {}'.format(*b, *a)
    return 'the {} {} is left of the {} {}'.format(*a, *b)


def encode(texts):
    """Captions to rows of word numbers, padded with zeros."""
    out = np.zeros((len(texts), MAXLEN), dtype=np.int64)
    for i, t in enumerate(texts):
        ids = [WID[w] for w in t.split()]
        out[i, :len(ids)] = ids
    return out


def pairs(combos, per, seed):
    """`per` pictures of each thing, each with a caption."""
    rng = np.random.default_rng(seed)
    imgs, caps, labels = [], [], []
    for k in combos:
        for _ in range(per):
            imgs.append(picture(*k, rng)); caps.append(caption(*k, rng)); labels.append(COMBOS.index(k))
    return np.stack(imgs), caps, np.array(labels)


def scene_pairs(n, seed):
    """n pictures of two different things with a caption; also the caption with the two things swapped."""
    rng = np.random.default_rng(seed)
    imgs, caps, swapped = [], [], []
    for _ in range(n):
        i, j = rng.choice(len(COMBOS), 2, replace=False)
        a, b = COMBOS[i], COMBOS[j]
        imgs.append(scene(a, b, rng)); caps.append(scene_caption(a, b, rng)); swapped.append(scene_caption(b, a, rng))
    return np.stack(imgs), caps, swapped


def sets():
    """The pictures used everywhere: 6,000 to learn from, and fresh ones of the shown and of the never shown things."""
    return pairs(SEEN, 200, 1), pairs(SEEN, 40, 2), pairs(HELD, 200, 3)
