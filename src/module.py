import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class FeatureEmbedder(nn.Module):
    def __init__(
        self,
        num_categorical,
        num_numerical,
        num_categories,
        emb_dim=64,
        hidden_dim=512,
        feature_dropout=0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_categorical = num_categorical
        if num_categorical:
            self.cat_embedding = nn.Sequential(
                nn.Embedding(num_categories+1, emb_dim, 0, scale_grad_by_freq=True),
                nn.Dropout2d(feature_dropout),
                nn.Flatten(start_dim=1)
            )
        self.num_embedding = nn.Sequential(
            nn.Dropout(feature_dropout),
            nn.Linear(num_numerical + 1, (num_numerical + 1) * emb_dim),
        )
        self.date_params = nn.Parameter(torch.Tensor([0,1,0]))
        self.net = nn.Sequential(
            nn.BatchNorm1d((num_categorical + num_numerical + 1) * emb_dim),
            weight_norm(nn.Linear((num_categorical + num_numerical + 1) * emb_dim, hidden_dim)),
            nn.ReLU(),
            weight_norm(nn.Linear(hidden_dim, hidden_dim)),
        )
        
    def forward(self, feats):
        cats, nums, dates = feats['cat'], feats['num'], feats['date']
        if dates.size(0) == 0:
            return torch.empty((0, self.hidden_dim), dtype=torch.float32, device=cats.device)
        w = torch.sigmoid(self.date_params[0])
        dates = w * torch.cos(self.date_params[1] * dates + self.date_params[2]) + (1-w) * dates
        nums = self.num_embedding(torch.cat((nums, dates.view(-1,1)), dim=1))
        if self.num_categorical:
            cats = self.cat_embedding(cats)
            return self.net(torch.cat((cats, nums), dim=1))
        else:
            return self.net(nums)
        
class SingleEventPredictor(nn.Module):
    def __init__(
        self,
        num_numerical,
        num_categories,
        hidden_dim=256,
    ):
        super().__init__()
        
        self.num_numerical = num_numerical
        self.num_categories = num_categories
        self.num_loss = nn.MSELoss()
        self.cat_loss = nn.CrossEntropyLoss(label_smoothing=0)
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, sum(num_categories)+num_numerical),
        )
        
    def forward(self, feats, num_true, cat_true):
        out = self.predictor(feats)
        num_out, cat_out = out[:self.num_numerical], out[self.num_numerical:]
        num_loss = self.num_loss(num_out[:self.num_numerical], num_true)
        cat_loss = 0
        for out, true in zip(cat_out.split(self.num_categories), cat_true):
            cat_loss = cat_loss + self.cat_loss(out, true)
        
        return num_loss + cat_loss
            
        

if __name__ == "__main__":
    import pickle
    with open("data/data.pkl", 'rb') as f:
        data = pickle.load(f)
    for cust_id, cust_data in data.items():
        break
    print(len(list(filter(lambda x: x[0] == "cdtx", cust_data["event_index"]))))
    # model = FeatureEmbedder()