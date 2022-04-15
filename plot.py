#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update({'font.size': 22})
plt.rcParams["figure.figsize"] = (8, 6)

with open('cifar10.txt') as f:
    lines = f.readlines()

iters = []
mask_ratio = []
acc = []
for line in lines:
    splits = line.split(',')
    for split in splits:
        if 'iteration' in split:
            iters.append(int(split.split(' ')[2]))
        if 'mask' in split:
            mask_ratio.append(1.0 - float(split.split(':')[1][8:]))
        if 'top-1-acc' in split:
            acc.append(float(split.split(':')[1]))

df = pd.DataFrame({'iter': iters, 'mask_ratio': mask_ratio, 'acc': acc})

fig1, ax1 = plt.subplots(1, 1)

ax1.plot(iters, mask_ratio, label='mask_ratio')
ax1.plot(iters, acc, label='top-1-acc')
ax1.legend(loc=4)
ax1.grid()
ax1.set(ylim=(0.0, 1.0))
ax1.xaxis.set_major_formatter(ticker.EngFormatter())
ax1.set_xlabel('Iter.')
ax1.set_ylabel('Accuracy/Ratio')
ax1.set_title('CIFAR-10-40')

plt.tight_layout()
plt.savefig('mask-10.pdf', format='pdf', dpi=1000, tight_layout=True)


with open('cifar100.txt') as f:
    lines = f.readlines()

iters = []
mask_ratio = []
acc = []
for line in lines:
    splits = line.split(',')
    for split in splits:
        if 'iteration' in split:
            iters.append(int(split.split(' ')[2]))
        if 'mask' in split:
            mask_ratio.append(1.0 - float(split.split(':')[1][8:]))
        if 'top-1-acc' in split:
            acc.append(float(split.split(':')[1]))

df = pd.DataFrame({'iter': iters, 'mask_ratio': mask_ratio, 'acc': acc})

fig2, ax2 = plt.subplots(1, 1)

ax2.plot(iters, mask_ratio, label='mask_ratio')
ax2.plot(iters, acc, label='top-1-acc')
ax2.legend(loc=4)
ax2.grid()
ax2.set(ylim=(0.0, 1.0))
ax2.xaxis.set_major_formatter(ticker.EngFormatter())
ax2.set_xlabel('Iter.')
ax2.set_ylabel('Accuracy/Ratio')
ax2.set_title('CIFAR-100-400')

plt.tight_layout()
plt.savefig('mask-100.pdf', format='pdf', dpi=1000, tight_layout=True)
