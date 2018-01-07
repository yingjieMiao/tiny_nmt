# Toy Neural Machine Translation with TensorFlow

## Introduction
This is an oversimplified, no-frill refactor of the [tensorflow/nmt](https://github.com/tensorflow/nmt) library. In just about 500 lines of code, it illustrates the core idea of encoder-decoder network, and provides an end-to-end training and inference example. A basic attention mechanism (Luong style) is also included as a training option.

Training and inference is based on synthetic data. Each source sentence is a sequence of numbers in {1, ..., 26}, and the target sentence is the sequence of corresponding ascii letters. For example, "3 2 7 26" translates to "c b g z". The source sentence is generated in two steps: 1) Sample sentence length `n` (e.g. randomly between 1 to 10); 2) Independently sample `n` numbers from {1, ..., 26}.

## Run the code

Running the code is simple:
```
python train.py
tensorboard --logdir /tmp/tf/log/nmt/test/
```

With the default parameters, you should see some debug info after each epoch, which contains: 1) a few sampled (src, tgt, nmt) tuples; 2) the translation of each single number to letter.

```
Done with epoch 4. Global step = 400
src:  20
tgt:  t
nmt:  
src:  16 20 1 10 14 18
tgt:  p t a j n r
nmt:  n t n n n r
char prediction: a, , c, d, e, , g, g, l, c, , , , , , , , , , , u, d, , , ,  

...

Done with epoch 30. Global step = 3000
src:  25 9
tgt:  y i
nmt:  y i
src:  15 10 2 21 4 16 14 15 21
tgt:  o j b u d p n o u
nmt:  o j j r i r r o r
char prediction:  a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z

...

Done with epoch 100. Global step = 10000
src:  8 22 17 11 11 21 8 21 5 8
tgt:  h v q k k u h u e h
nmt:  h v q k k q w u g a
src:  3 2 4 13 6 22 22
tgt:  c b d m f v v
nmt:  c b d m f v v
char prediction:  a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z

...

Done with epoch 180. Global step = 18000
src:  21 2 18 19 3 2 13 15 2
tgt:  u b r s c b m o b
nmt:  u b r s c b m o b
src:  10 19 18 12 1 10 17 17 25
tgt:  j s r l a j q q y
nmt:  j s r l a j q q y
char prediction:  a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z

```

The NMT model gradually learned a few things:
- Correct mapping from single number in {1..26} to single letter in {a..z}.
- Correct target sequence length.
- Good translation of short sentences.
- Good translation of longer sentences.

It would be interesting to look at the trained model parameters and analyze what happened under the hood. 

## Things not covered
A few important components that are not covered here (yet):
- Real language dataset
- Beam Search
- Smarter learning rate scheduling
- Perplexity and BLEU metrics
