"""Step 1. The world: 36 things, pictures of them, and captions. Six things are kept away from training."""
import numpy as np
import nets
import world

(xtr, ctr, ltr), (xs, cs, ls), (xh, ch, lh) = world.sets()
print(f'{len(world.CNAMES)} colours times {len(world.SHAPES)} shapes: {len(world.COMBOS)} things')
print(f'shown in training: {len(world.SEEN)} things, {len(xtr):,} pictures, each with a caption')
print('never shown:', ', '.join(f'{c} {s}' for c, s in world.HELD))
print(f'one picture: {world.SIZE} by {world.SIZE} pixels, 3 numbers for each pixel, {xtr[0].size:,} numbers')
print(f'words in all captions: {len(world.WORDS) - 1}; ways to write a caption: {len(world.TEMPLATES)}')
for i in (0, 201, 402):
    print(f'  caption of picture {i}: "{ctr[i]}"  as numbers: {[int(v) for v in world.encode([ctr[i]])[0] if v]}')
print(f'fresh test pictures: {len(xs):,} of the shown things, {len(xh):,} of the never shown things')

rng = np.random.default_rng(7)
nets.save('step1', {'things': len(world.COMBOS), 'seen': len(world.SEEN), 'held': [list(k) for k in world.HELD], 'train': len(xtr),
                    'numbers': int(xtr[0].size), 'words': len(world.WORDS) - 1, 'templates': len(world.TEMPLATES),
                    'test_seen': len(xs), 'test_held': len(xh), 'captions': [ctr[i] for i in (0, 201, 402, 603, 804)],
                    'ids': [[int(v) for v in world.encode([ctr[i]])[0] if v] for i in (0, 201, 402)], 'vocab': world.WORDS[1:]})
# Sample pictures for the video: four of every thing, and a few two-thing scenes.
np.save(nets.FILES + '/samples.npy', np.stack([[world.picture(*k, rng) for _ in range(4)] for k in world.COMBOS]))
