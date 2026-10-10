"""Step 6. Two settings of the training: how many pairs are compared at once, and the temperature."""
import numpy as np
import torch
import nets
import world

(xtr, ctr, ltr), (xs, cs, ls), (xh, ch, lh) = world.sets()
PAIRS = 64 * nets.STEPS


def run(batch, steps, seed, things=None, **kw):
    torch.manual_seed(seed)
    model = nets.Clip(**kw)
    nets.train(model, xtr, ctr, steps, batch=batch, seed=seed, things=things)
    return nets.score(model, xs, ls), nets.score(model, xh, lh), 1 / model.scale.exp().item()


def mean(batch, steps, things=None, **kw):
    r = np.array([run(batch, steps, seed, things, **kw) for seed in range(2)])
    return [float(v) for v in r.mean(0)]


print(f'pairs compared at once, every run looks at the same {PAIRS:,} pairs (average of two starts):')
same_pairs = {}
for batch in (4, 16, 64, 256):
    same_pairs[batch] = mean(batch, PAIRS // batch)
    print(f'  {batch:3d} at once, {PAIRS // batch:6,} steps: shown things {nets.pc(same_pairs[batch][0])}, never shown things {nets.pc(same_pairs[batch][1])}')
print(f'pairs compared at once, every run takes the same {nets.STEPS:,} steps:')
same_steps = {64: same_pairs[64]}
for batch in (4, 16, 256):
    same_steps[batch] = mean(batch, nets.STEPS)
for batch in (4, 16, 64, 256):
    print(f'  {batch:3d} at once: shown things {nets.pc(same_steps[batch][0])}, never shown things {nets.pc(same_steps[batch][1])}')
# A batch of 256 drawn from 30 things holds every thing many times. The training then calls right captions wrong.
twins = {b: (b - 1) / len(world.SEEN) for b in (4, 16, 64, 256)}
print('other pairs of the same thing in a batch, on average: ' + ', '.join(f'{b} at once: {twins[b]:.1f}' for b in twins))
fixed = mean(256, nets.STEPS, things=ltr)
print(f'  256 at once, captions of the same thing are not counted as wrong: shown things {nets.pc(fixed[0])}, never shown things {nets.pc(fixed[1])}')
print('temperature, 64 at once (average of two starts):')
temp = {'learned': same_pairs[64]}
for t in (1.0, 0.07, 0.01):
    temp[str(t)] = mean(64, nets.STEPS, temperature=t, learn_temperature=False)
for k, label in (('1.0', 'fixed at 1'), ('0.07', 'fixed at 0.07'), ('0.01', 'fixed at 0.01'), ('learned', 'starts at 0.07 and is learned')):
    print(f'  {label:30s}: shown things {nets.pc(temp[k][0])}, never shown things {nets.pc(temp[k][1])}' + (f', ends at {temp[k][2]:.3f}' if k == 'learned' else ''))
# Why a temperature of 1 fails: likeness lies between -1 and 1, so even a perfect model cannot be sure.
best = {t: float(np.exp(1 / t) / (np.exp(1 / t) + 63 * np.exp(-1 / t))) for t in (1.0, 0.07)}
print(f'the most a perfect model can bet on the right caption out of 64: {nets.pc(best[1.0])} at temperature 1, {nets.pc(best[0.07])} at 0.07')
nets.save('step6', {'pairs': PAIRS, 'same_pairs': same_pairs, 'same_steps': same_steps, 'temp': temp, 'twins': twins, 'fixed': fixed, 'best': best})
