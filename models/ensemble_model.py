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
    
    def forward(self,X):
        # if len(X[0].size())==4 or len(X[1].size())==4:
        #     import ipdb;ipdb.set_trace()
        out_fine=self.model_fine(X[0])
        out_raw=self.model_raw(X[1])
        out=out_fine+out_raw

        return out
    

