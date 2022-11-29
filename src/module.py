import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from pdb import set_trace

class FeatureEmbedder(nn.Module):
    def __init__(
        self,
        num_categorical,
        num_numerical,
        num_categories,
        emb_dim=32,
        hidden_dim=256,
        feature_dropout=0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_categorical = num_categorical
        if num_categorical:
            self.cat_embedding = nn.Sequential(
                nn.Embedding(num_categories, emb_dim, scale_grad_by_freq=True),
                nn.Flatten(start_dim=1)
            )
        self.num_embedding = nn.Sequential(
            nn.Dropout(.1),
            nn.Linear(num_numerical + 1, (num_numerical + 1) * emb_dim),
        )
        self.date_params = nn.Parameter(torch.Tensor([0,1]))
        self.net = nn.Sequential(
            nn.BatchNorm1d((num_categorical + num_numerical + 1) * emb_dim),
            weight_norm(nn.Linear((num_categorical + num_numerical + 1) * emb_dim, hidden_dim)),
            nn.ReLU(),
            weight_norm(nn.Linear(hidden_dim, hidden_dim)),
        )
        
    def forward(self, feats):
        cats, nums, dates = feats['cat'], feats['num'], feats['date']
        cats, nums, dates = cats.nan_to_num(), nums.nan_to_num(), dates.nan_to_num()
        if dates.size(0) == 0:
            return torch.empty((0, self.hidden_dim), dtype=torch.float32, device=cats.device)
        w = torch.sigmoid(self.date_params[0])
        dates = w * torch.cos(self.date_params[1] * dates) + (1-w) * dates
        nums = self.num_embedding(torch.cat((nums, dates.view(-1,1)), dim=1))
        if self.num_categorical:
            cats = self.cat_embedding(cats)
            return self.net(torch.cat((cats, nums), dim=1))
        else:
            return self.net(nums)
        

if __name__ == "__main__":
    import pickle
    with open("data/data.pkl", 'rb') as f:
        data = pickle.load(f)
    for cust_id, cust_data in data.items():
        break
    print(len(list(filter(lambda x: x[0] == "cdtx", cust_data["event_index"]))))
    # model = FeatureEmbedder()