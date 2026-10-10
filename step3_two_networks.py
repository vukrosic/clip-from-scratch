"""Step 3. Two networks, one for pictures and one for captions. Both give a list of 64 numbers. Nothing is trained yet."""
import numpy as np
import torch
import nets
import world

(xtr, ctr, ltr), (xs, cs, ls), (xh, ch, lh) = world.sets()
torch.manual_seed(0)
model = nets.Clip().eval()
print(f'picture network: {nets.numbers(model.picture):,} numbers; text network: {nets.numbers(model.text):,} numbers')
print(f'both turn their input into a list of {nets.DIM} numbers')
SIX = [('red', 'circle'), ('green', 'square'), ('blue', 'triangle'), ('yellow', 'cross'), ('purple', 'ring'), ('pink', 'diamond')]
pick = [int(np.where(ls == world.COMBOS.index(k))[0][0]) for k in SIX]
with torch.no_grad():
    table = model.table(nets.T(xs[pick]), nets.T(world.encode(nets.prompts(SIX)))).numpy()
print('how alike picture and caption are, six pictures against six captions (1 = same direction):')
for k, row in zip(SIX, table):
    print(f'  {k[0] + " " + k[1]:16s}' + ' '.join(f'{v:6.2f}' for v in row))
acc = nets.score(model, xs, ls)
print(f'pick the right caption out of {len(world.COMBOS)}: {nets.pc(acc)} right (a blind guess: {nets.pc(1 / len(world.COMBOS))})')
np.save(nets.FILES + '/six.npy', xs[pick])
nets.save('step3', {'pic_numbers': nets.numbers(model.picture), 'text_numbers': nets.numbers(model.text), 'dim': nets.DIM,
                    'six': [list(k) for k in SIX], 'table': table.round(3).tolist(), 'acc': acc, 'chance': 1 / len(world.COMBOS)})
