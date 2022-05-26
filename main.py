import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import argparse
import os
from utils import net_builder
from models.fixmatch.fixmatch import FixMatch
from models.flexmatch.flexmatch import FlexMatch
from models.refixmatch.refixmatch import ReFixMatch
from models.sequencematch.sequencematch import SequenceMatch
import numpy as np
import pandas as pd
# from tsne import bh_sne
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

parser = argparse.ArgumentParser(description='PyTorch t-SNE for STL10')
parser.add_argument('--load_path', type=str, default=None)
parser.add_argument('--save-dir', type=str, default='./results', help='path to save the t-sne image')
parser.add_argument('--batch-size', type=int, default=128, help='batch size (default: 128)')
parser.add_argument('--seed', type=int, default=1, help='random seed value (default: 1)')

args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# set seed
torch.manual_seed(args.seed)
if device == 'cuda':
    torch.cuda.manual_seed(args.seed)

# set dataset
transform = transforms.Compose([    
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

dataset = torchvision.datasets.STL10(root='data', split='test', download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

# set model
# net = models.resnet18(pretrained=True)
_net_builder = net_builder("WideResNetVar",
                            False,
                            {'first_stride': 2,
                            'depth': 28,
                            'widen_factor': 2,
                            'leaky_slope': 0.1,
                            'bn_momentum': 0.9,
                            'dropRate': 0,
                            'use_embed': False,
                            'is_remix': False},
                            )

net = SequenceMatch(_net_builder,
                10,
                0.999,
                0.5,
                0.95,
                1.0,
                True,
                num_eval_iter=5000)
net.model = net.model.to(device)
if device == 'cuda':
    net.model = torch.nn.DataParallel(net.model)
    cudnn.benchmark = True

net.load_model(args.load_path)
net = net.model

def gen_features():
    net.eval()
    targets_list = []
    outputs_list = []

    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            targets_np = targets.data.cpu().numpy()

            outputs = net(inputs)
            outputs_np = outputs.data.cpu().numpy()
            
            targets_list.append(targets_np[:, np.newaxis])
            outputs_list.append(outputs_np)
            
            if ((idx+1) % 10 == 0) or (idx+1 == len(dataloader)):
                print(idx+1, '/', len(dataloader))

    targets = np.concatenate(targets_list, axis=0)
    outputs = np.concatenate(outputs_list, axis=0).astype(np.float64)

    return targets, outputs

def tsne_plot(save_dir, targets, outputs):
    print('generating t-SNE plot...')
    # tsne_output = bh_sne(outputs)
    tsne = TSNE(random_state=0)
    tsne_output = tsne.fit_transform(outputs)

    df = pd.DataFrame(tsne_output, columns=['x', 'y'])
    df['targets'] = targets

    plt.rcParams['figure.figsize'] = 10, 10
    sns.scatterplot(
        x='x', y='y',
        hue='targets',
        palette=sns.color_palette("hls", 10),
        data=df,
        marker='o',
        legend="full",
    )

    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    plt.tight_layout(pad=0.4)
    plt.savefig(os.path.join(save_dir,'SequenceMatch.pdf'), format='pdf', dpi=1000, tight_layout=True)
    print('done!')

targets, outputs = gen_features()
tsne_plot(args.save_dir, targets, outputs)
