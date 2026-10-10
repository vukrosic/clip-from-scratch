"""Step 4. Training: in every batch each picture must pick its own caption, and each caption its own picture."""
import json
import numpy as np
import torch
import nets
import world

(xtr, ctr, ltr), (xs, cs, ls), (xh, ch, lh) = world.sets()
six = json.load(open(nets.FILES + '/step3.json'))['six']
six_x = np.load(nets.FILES + '/six.npy')
six_ids = nets.T(world.encode(nets.prompts([tuple(k) for k in six])))
some = np.random.default_rng(0).permutation(len(xtr))[:512]              # 512 training pairs to measure the miss on
x, ids = nets.T(xtr[some]), nets.T(world.encode([ctr[i] for i in some]))
runs = []
for seed in range(3):
    torch.manual_seed(seed)
    model = nets.Clip()
    probe = lambda m: (float(np.mean([m.loss(x[i:i + 64], ids[i:i + 64]).item() for i in range(0, 512, 64)])), nets.score(m, xs, ls),
                       m.table(nets.T(six_x), six_ids).numpy().round(3).tolist())
    log = nets.train(model, xtr, ctr, nets.STEPS, seed=seed, every=50, probe=probe)
    nets.keep(model, f'clip-{seed}')
    runs.append({'steps': [l[0] for l in log], 'loss': [l[1] for l in log], 'acc': [l[2] for l in log], 'tables': [l[3] for l in log][::3],
                 'temperature': 1 / model.scale.exp().item()})
    print(f'start {seed}: miss {log[0][1]:.2f} -> {log[-1][1]:.2f}; right caption out of 36: {nets.pc(log[0][2])} -> {nets.pc(log[-1][2])}')
r = runs[0]
print(f'batch of 64 pairs, {nets.STEPS:,} steps, {64 * nets.STEPS:,} pairs looked at')
print('start 0, right caption out of 36 by step: ' + ', '.join(f'{s}: {nets.pc(a)}' for s, a in zip(r['steps'][::5], r['acc'][::5])))
print('the six pictures against the six captions after training:')
for k, row in zip(six, r['tables'][-1]):
    print(f'  {k[0] + " " + k[1]:16s}' + ' '.join(f'{v:6.2f}' for v in row))
mean = float(np.mean([q['acc'][-1] for q in runs]))
print(f'fresh pictures of the 30 shown things, right caption out of 36: {nets.pc(mean)} on average over three starts')
print(f'the temperature moved from 0.07 to {r["temperature"]:.3f}')
nets.save('step4', {'runs': runs, 'acc': mean, 'steps': nets.STEPS, 'batch': 64})
