"""Step 7. A limit: two things in one picture. Does the model know which one is on the left?"""
import numpy as np
import torch
import nets
import world

LONG = 4 * nets.STEPS
xtr, ctr, wtr = world.scene_pairs(6000, 11)
xte, cte, wte = world.scene_pairs(1000, 12)
print(f'{len(xtr):,} pictures with two things. A caption: "{ctr[0]}". The same with the things swapped: "{wtr[0]}"')


def check(model):
    """Right caption against 63 captions of other pictures; and right caption against its own swapped one."""
    p = model.see(nets.T(xte))
    right, wrong = model.read(nets.T(world.encode(cte))), model.read(nets.T(world.encode(wte)))
    among = np.mean([(p[i:i + 64] @ right[i:i + 64].T).argmax(1).eq(torch.arange(64)).float().mean().item() for i in range(0, 960, 64)])
    order = ((p * right).sum(1) > (p * wrong).sum(1)).float().mean().item()
    return float(among), float(order)


out = {}
for label, hard in (('plain', None), ('hard', wtr)):
    logs = []
    for seed in range(2):
        torch.manual_seed(seed)
        model = nets.Clip()
        logs.append(nets.train(model, xtr, ctr, LONG, seed=seed, hard=hard, every=500, probe=check))
    steps = [l[0] for l in logs[0]]
    among = np.mean([[l[1] for l in log] for log in logs], 0)
    order = np.mean([[l[2] for l in log] for log in logs], 0)
    at = steps.index(nets.STEPS)
    out[label] = {'steps': steps, 'among': among.tolist(), 'order': order.tolist(), 'each': [[l[2] for l in log] for log in logs]}
    print('trained as before:' if hard is None else 'trained with the swapped caption as one more wrong answer for every picture:')
    print(f'  after {nets.STEPS:,} steps: right caption against 63 captions of other pictures {nets.pc(among[at])} (a blind guess: {nets.pc(1 / 64)})')
    print(f'  after {nets.STEPS:,} steps: right caption against the swapped caption {nets.pc(order[at])} (a blind guess: 50.0 %)')
    print('  against the swapped caption by step: ' + ', '.join(f'{s}: {nets.pc(o)}' for s, o in zip(steps[::2], order[::2])))
    print(f'  after {LONG:,} steps: {nets.pc(order[-1])} against the swapped caption, {nets.pc(among[-1])} against 63 others')
mirror = 63 / (len(world.COMBOS) * (len(world.COMBOS) - 1))
print(f'chance that a batch of 64 holds the mirrored picture of a given picture: {nets.pc(mirror)}')
np.save(nets.FILES + '/scenes.npy', xte[:8])
nets.save('step7', {**out, 'train': len(xtr), 'test': len(xte), 'captions': cte[:8], 'swapped': wte[:8], 'long': LONG, 'mirror': mirror})
