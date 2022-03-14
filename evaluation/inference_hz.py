import torch
from tqdm import tqdm

from motion import Motion, Skeleton
import yaml

with open('./config/default.yaml', 'r') as stream:
    params = yaml.safe_load(stream)

with open(params['graph_config'], 'r') as stream:
    skeleton = Skeleton(**yaml.safe_load(stream))
model = Motion(graph_representation=skeleton, **params).to('cuda:0')

x = torch.rand(1, 2, 32, 4)
x = torch.nn.functional.normalize(x, dim=-1).to('cuda:0')
with torch.no_grad():
    for i in tqdm(range(1000)):
        model(x)