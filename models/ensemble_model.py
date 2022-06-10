from .bci2021_model import BCI2021
import torch
import torch.nn as nn

class EnsembleModel(nn.Module):
    def __init__(
            self,
            n_classes,
            n_band,
            cnn_params,
            sub_band_att_params,
            lstm_params,
            segment_att_params,
            cnn_params2,
            sub_band_att_params2,
            lstm_params2,
            segment_att_params2,
            **kwargs
    ):
        super(EnsembleModel, self).__init__()
        self.model_fine=BCI2021(n_classes,
            n_band,
            cnn_params,
            sub_band_att_params,
            lstm_params,
            segment_att_params)
        self.model_raw=BCI2021(n_classes,
            n_band,
            cnn_params2,
            sub_band_att_params2,
            lstm_params2,
            segment_att_params2)
        
        # self.fusion=nn.Sequential(
        #     nn.Linear(segment_att_params[0]*2,segment_att_params[0]),
        #     nn.LeakyReLU()
        # )
        self.dropout=nn.Dropout(p=0.2)
        self.fc = nn.Linear(segment_att_params[0]*2, n_classes)
    
    def forward(self,X):
        # if len(X[0].size())==4 or len(X[1].size())==4:
        #     import ipdb;ipdb.set_trace()
        out_fine=self.model_fine(X[0])
        out_raw=self.model_raw(X[1])
        # out=out_fine+out_raw
        # import ipdb;ipdb.set_trace()
        out=torch.cat([out_fine,out_raw],dim=-1)
        # out=self.fusion(torch.cat([out_fine,out_raw],dim=-1))
        
        out=self.dropout(out)
        out = self.fc(out)

        return out
    

