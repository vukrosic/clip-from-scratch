# CLIP from scratch

A small CLIP, built step by step in Python with PyTorch. It trains on a laptop CPU.
It is the code for the video "How AI Learned to Name Things It Was Never Shown".

The question: how can a model name a thing that nobody taught it to name?

## The world

Pictures are 32 by 32 pixels and drawn by a program. Each shows one thing: one of 6 colours and one of 6 shapes, so 36 things.
Six things are locked away (red triangle, green cross, blue ring, yellow diamond, purple circle, pink square).
No picture and no caption of them goes into training. The other 30 things give 6,000 pairs of a picture and its caption.

## Run it

```
pip install torch numpy
sh run_all.sh
```

All steps together take about 45 minutes on a laptop CPU. Each step also runs alone, for example `python3 step4_pull.py`.
`demo-output.txt` holds the full output of one run. No downloads: no dataset, no trained model.

## The steps

| File | What it does |
| --- | --- |
| `world.py` | draws the pictures and writes the captions |
| `nets.py` | the picture network, the text network, the training rule |
| `step1_world.py` | the 36 things, the 6 locked ones, the 6,000 pairs |
| `step2_fixed_list.py` | the old way: a network with a fixed list of 30 answers |
| `step3_two_networks.py` | two networks, 64 numbers each, and the table of likeness before training |
| `step4_pull.py` | contrastive training: true pairs together, all other pairs apart |
| `step5_never_shown.py` | the 6 locked things: naming them and searching for them |
| `step6_what_matters.py` | batch size and temperature |
| `step7_word_order.py` | two things in a picture, and the left/right trap |
| `exercise.py` | swap the two colours in the caption instead of the places |
| `tests.py` | checks of the pieces |

## What we measured

One laptop run. Your numbers will differ a little.

| Measurement | Result |
| --- | --- |
| Fixed list of 30 answers, new pictures of its 30 things | 75.6 % |
| Fixed list, on the 6 locked things | 0 % (colour right 99.2 % of the time) |
| Two networks, 1,500 steps, 30 shown things (average of 3 runs) | 95.3 % |
| Two networks, 6 locked things, one caption out of 36 | 79.8 % (a blind guess: 2.8 %) |
| Same, choosing only among the 6 new captions | 99.7 % |
| Caption without "a picture of" | 78.3 % |
| Average of all 5 caption forms | 80.9 % |
| Search by caption, locked things, top 20 pictures that fit | 100 % |
| Batch of 4 / 16 / 64 / 256 pairs, 1,500 steps each | 18.8 / 53.6 / 95.3 / 86.8 % |
| Batch of 256, captions of the same thing not counted as wrong | 97.4 % |
| Temperature fixed at 1 / fixed at 0.07 / learned | 18.3 / 93.4 / 95.3 % |
| Two things per picture, right caption out of 64, 1,500 steps | 82.6 % |
| True caption against left/right swapped, 3,000 steps | 50.9 % |
| Same, with the swapped caption added as a wrong answer, 3,000 steps | 97.5 % |
| Without that fix, 6,000 steps | 93.8 % |

The exercise result is at the end of `demo-output.txt`. Guess first.

## Limits

- This is a toy. 36 things drawn by a program say nothing about photographs.
- A network with one answer for colour and one for shape would also name a red triangle. The point shown here is that captions supply the answers, and nobody designs a list.
- Neither network was tuned. Do not read the scores as a fair race between the two methods.
- One run per number, except where the table says an average.

## Sources

- Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", 2021, arXiv 2103.00020 (CLIP).
- Yuksekgonul et al., "When and why vision-language models behave like bags-of-words, and what to do about it?", 2022, arXiv 2210.01936.

## License

MIT
