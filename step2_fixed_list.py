"""Step 2. The old way: a network with one answer for each class on a fixed list of 30."""
from collections import Counter
import numpy as np
import torch
import nets
import world

(xtr, ctr, ltr), (xs, cs, ls), (xh, ch, lh) = world.sets()
seen_ids = [world.COMBOS.index(k) for k in world.SEEN]
to_class = {g: i for i, g in enumerate(seen_ids)}
accs = []
for seed in (2, 1, 0):                         # three starts; the last one (start 0) is looked at below
    torch.manual_seed(seed)
    model = nets.Classifier(len(world.SEEN))
    nets.train_classifier(model, xtr, np.array([to_class[l] for l in ltr]), nets.STEPS, seed=seed)
    with torch.no_grad():
        ps = model(nets.T(xs)).argmax(1).numpy()
        ph = model(nets.T(xh)).argmax(1).numpy()
    accs.append(float(np.mean([seen_ids[p] == l for p, l in zip(ps, ls)])))
acc = float(np.mean(accs))
print(f'a list of {len(world.SEEN)} classes, {nets.numbers(model):,} numbers, {nets.STEPS:,} training steps')
print(f'fresh pictures of the 30 things on its list: {nets.pc(acc)} right on average over three starts ({", ".join(nets.pc(a) for a in accs[::-1])})')
print('pictures of the 6 things that are not on its list: 0.0 % right (the right answer is not among its answers)')
rows, colour_ok, shape_ok = [], 0, 0
for k in world.HELD:
    said = [world.SEEN[p] for p, l in zip(ph, lh) if world.COMBOS[l] == k]
    top = Counter(said).most_common(2)
    colour_ok += sum(s[0] == k[0] for s in said); shape_ok += sum(s[1] == k[1] for s in said)
    rows.append({'thing': list(k), 'says': [[list(t), n] for t, n in top], 'of': len(said)})
    print(f'  {k[0]} {k[1]}: it says ' + ', '.join(f'"{t[0]} {t[1]}" {n} times' for t, n in top) + f' (of {len(said)})')
print(f'its answer has the right colour in {nets.pc(colour_ok / len(xh))} and the right shape in {nets.pc(shape_ok / len(xh))} of them')
nets.save('step2', {'classes': len(world.SEEN), 'numbers': nets.numbers(model), 'acc': acc, 'rows': rows,
                    'colour': colour_ok / len(xh), 'shape': shape_ok / len(xh)})
