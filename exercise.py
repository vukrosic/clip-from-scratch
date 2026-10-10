"""Exercise. Step 7 swapped the places of two things. Now swap their colours.

The picture shows a red circle left of a blue square. Two captions:
    right:  "the red circle is left of the blue square"
    wrong:  "the blue circle is left of the red square"
Both captions hold the same words. Does the model from step 7 pick the right one?
Guess before you run it. Then train it with the wrong caption as one more wrong answer, as in step 7.
"""
import numpy as np
import torch
import nets
import world


def recolour(caption):
    """The same caption with the two colour words swapped."""
    words = caption.split()
    i, j = [k for k, w in enumerate(words) if w in world.CNAMES]
    words[i], words[j] = words[j], words[i]
    return ' '.join(words)


def different_colours(n, seed):
    imgs, caps, _ = world.scene_pairs(3 * n, seed)
    keep = [i for i, c in enumerate(caps) if recolour(c) != c][:n]
    return imgs[keep], [caps[i] for i in keep], [recolour(caps[i]) for i in keep]


xtr, ctr, wtr = different_colours(6000, 11)
xte, cte, wte = different_colours(1000, 12)
show = next(i for i, c in enumerate(cte) if c.split()[2] != c.split()[-1])  # an example with two different shapes
print(f'a caption: "{cte[show]}"; with the colours swapped: "{wte[show]}"')


def check(model):
    p = model.see(nets.T(xte))
    return (((p * model.read(nets.T(world.encode(cte)))).sum(1) > (p * model.read(nets.T(world.encode(wte)))).sum(1)).float().mean().item(),)


out = {}
for label, hard in (('plain', None), ('hard', wtr)):
    torch.manual_seed(0)
    model = nets.Clip()
    log = nets.train(model, xtr, ctr, nets.STEPS, hard=hard, every=nets.STEPS, probe=check)
    out[label] = log[-1][1]
    print(('trained as before' if hard is None else 'trained with the recoloured caption as one more wrong answer')
          + f', {nets.STEPS:,} steps: right caption against the recoloured one {nets.pc(log[-1][1])} (a blind guess: 50.0 %)')
nets.save('exercise', out)
