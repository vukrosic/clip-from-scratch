"""Step 5. Things the training never showed. The model names them by reading captions that we write now."""
from collections import Counter
import numpy as np
import torch
import nets
import world

(xtr, ctr, ltr), (xs, cs, ls), (xh, ch, lh) = world.sets()
models = [nets.fetch(f'clip-{seed}') for seed in range(3)]
all36 = [nets.score(m, xh, lh) for m in models]
new6 = [nets.score(m, xh, lh, world.HELD) for m in models]
print('pictures of the 6 never shown things:')
print(f'  right caption out of all 36: {nets.pc(np.mean(all36))} on average ({", ".join(nets.pc(a) for a in all36)}); a blind guess: {nets.pc(1 / 36)}')
print(f'  right caption out of the 6 new ones: {nets.pc(np.mean(new6))} on average; a blind guess: {nets.pc(1 / 6)}')
m = models[0]
pick = nets.name(m, xh, nets.prompts(world.COMBOS))
rows = []
for k in world.HELD:
    said = [world.COMBOS[p] for p, l in zip(pick, lh) if world.COMBOS[l] == k]
    wrong = Counter(s for s in said if s != k).most_common(1)
    rows.append({'thing': list(k), 'right': sum(s == k for s in said) / len(said), 'wrong': [list(wrong[0][0]), wrong[0][1]] if wrong else None})
    print(f'  {k[0]} {k[1]}: {nets.pc(rows[-1]["right"])} right' + (f'; most common mistake "{wrong[0][0][0]} {wrong[0][0][1]}" {wrong[0][1]} times' if wrong else ''))

# How the caption is written matters.
bare = [nets.score(q, xh, lh, template='{} {}') for q in models]
print(f'captions written as "red triangle", a form the training never used: {nets.pc(np.mean(bare))}')
with torch.no_grad():
    many = []
    for q in models:
        t = torch.stack([q.read(nets.T(world.encode(nets.prompts(world.COMBOS, tpl)))) for tpl in world.TEMPLATES]).mean(0)
        many.append(float(((q.see(nets.T(xh)) @ t.T).argmax(1).numpy() == lh).mean()))
print(f'the average of all {len(world.TEMPLATES)} ways to write the caption: {nets.pc(np.mean(many))}')

# Why it works: where the text network puts a caption it has never read.
with torch.no_grad():
    t = m.read(nets.T(world.encode(nets.prompts(world.COMBOS))))
    sim = (t @ t.T).numpy()
col, shp, none = [], [], []
for k in world.HELD:
    i = world.COMBOS.index(k)
    for j, o in enumerate(world.COMBOS):
        if o in world.SEEN:
            (col if o[0] == k[0] else shp if o[1] == k[1] else none).append(sim[i, j])
print(f'a never read caption against the others: {np.mean(col):.2f} alike with the same colour, {np.mean(shp):.2f} with the same shape, {np.mean(none):.2f} with neither')

# Search by text: 2,400 fresh pictures, the caption picks the 20 closest.
pool, labels = np.concatenate([xs, xh]), np.concatenate([ls, lh])
with torch.no_grad():
    table = m.table(nets.T(pool), nets.T(world.encode(nets.prompts(world.COMBOS)))).numpy()
hits = {k: float(np.mean(labels[np.argsort(-table[:, world.COMBOS.index(k)])[:20]] == world.COMBOS.index(k))) for k in world.COMBOS}
top = np.argsort(-table[:, world.COMBOS.index(world.HELD[0])])[:12]
print(f'search in {len(pool):,} fresh pictures, share of the top 20 that fit: {nets.pc(np.mean([hits[k] for k in world.HELD]))} for the 6 never shown things, '
      f'{nets.pc(np.mean([hits[k] for k in world.SEEN]))} for the 30 shown things')
np.save(nets.FILES + '/search.npy', pool[top])
nets.save('step5', {'all36': all36, 'new6': new6, 'rows': rows, 'bare': bare, 'many': many, 'alike': [float(np.mean(col)), float(np.mean(shp)), float(np.mean(none))],
                    'search_held': float(np.mean([hits[k] for k in world.HELD])), 'search_seen': float(np.mean([hits[k] for k in world.SEEN])),
                    'search_query': list(world.HELD[0]), 'search_fit': [bool(labels[i] == world.COMBOS.index(world.HELD[0])) for i in top], 'pool': len(pool)})
