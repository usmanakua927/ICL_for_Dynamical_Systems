import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, Lasso
import warnings
from sklearn import tree
import xgboost as xgb

from base_models import NeuralNetwork, ParallelNetworks



def build_model(conf, y_dim=2, discard=True, discard_mode='All', control=True, Non_Linear=False, non_lin_mode=1, state_est=False):
    
    if conf.family == "gpt2":

        model = TransformerModelOneStepPredControl(
            n_dims=y_dim * (conf.n_dims + 1),
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
            y_dim=y_dim,
            discard = discard,
            discard_mode = discard_mode,
            control=control,
            Non_Linear=Non_Linear,
            non_lin_mode=non_lin_mode,
            state_est=state_est
        )

    else:
        raise NotImplementedError

    return model


class TransformerModelOneStepPredControl(nn.Module):
    def __init__(self, n_dims, n_positions, discard_mode='All', discard=False, n_embd=128, n_layer=12, n_head=4, y_dim=2, control=True, Non_Linear=False, non_lin_mode=2, state_est=False):
        super(TransformerModelOneStepPredControl, self).__init__()

        if not Non_Linear or (Non_Linear and (non_lin_mode==6 or non_lin_mode==7 or non_lin_mode==8 or non_lin_mode==11)):
            if control:
                if not discard:
                    pos=int(3*n_positions+3*(n_dims/y_dim-1)+1)
                elif discard and discard_mode=='All':
                    pos=int(3 * n_positions)
                    
                elif discard and discard_mode=='AllEM':
                    pos=int(n_positions)
                elif discard and discard_mode == 'Noise':
                    pos = int(3 * n_positions + 2*(n_dims / y_dim - 1))
            else:
                if not discard:
                    pos=int(2*n_positions+2*(n_dims/y_dim-1)+1)
                elif discard and discard_mode=='All':
                    pos=int(2 * n_positions)
                elif discard and discard_mode=='AllEM':
                    pos=int(n_positions)
                    
                elif discard and discard_mode == 'Noise':
                    pos = int(2 * n_positions + (n_dims / y_dim - 1))
        else:
            if discard:
                if control:
                    if discard_mode=='AllEM':
                        pos = int(n_positions)
                    else:
                        pos = int(3 * n_positions)
                else:
                    # if (non_lin_mode==3 or non_lin_mode==4): #Legacy
                    if (non_lin_mode==3):
                        pos = int(n_positions)
                    else:
                        if discard_mode=='AllEM':
                            pos = int(n_positions);
                        else:
                            pos = int(2 * n_positions)

            else:
                if control:
                    if non_lin_mode==1:
                        pos = int(3 * n_positions+ 2*(n_dims/y_dim-1)+1)
                    elif non_lin_mode==2 or non_lin_mode==4 or non_lin_mode==5 or non_lin_mode==9 or non_lin_mode==10:
                        pos = int(3 * n_positions+ 2*(n_dims/y_dim-1)+1+2)


                else:
                    if non_lin_mode==1:
                        pos = int(2 * n_positions +  (n_dims / y_dim - 1) + 1)
                    elif non_lin_mode == 2 or non_lin_mode==4 or non_lin_mode==5 or non_lin_mode==9 or non_lin_mode==10:
                        pos = int(2 * n_positions + (n_dims / y_dim - 1) + 1+2)

                    elif non_lin_mode==3:
                        pos = int(n_positions+5+1);

                    # elif non_lin_mode==4:
                    #     pos = int(n_positions + 5 + 1+1);  # Legacy non_lin_mode 4 previously extended non_lin_mode==3 by inserting del_t

        configuration = GPT2Config(
            n_positions=pos,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = pos
        self.state_est=state_est


        # if Non_Linear and (non_lin_mode==3 or non_lin_mode==4): # Legacy
        if Non_Linear and (non_lin_mode==3):
            self.state_dim = 5;
            if discard:
                self.n_dims = 2
            else:
                self.n_dims = 7
            self.y_dim = 2
        else:
            self.n_dims = n_dims
            self.y_dim = y_dim
            self.state_dim = int((n_dims / y_dim - 1))
        self._read_in = nn.Linear(self.n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        if self.state_est:
            self._read_out = nn.Linear(n_embd, self.state_dim)
        else:
            self._read_out = nn.Linear(n_embd, self.y_dim)
        self.discard=discard
        self.discard_mode=discard_mode
        self.control=control
        self.Non_Linear=Non_Linear
        self.non_lin_mode=non_lin_mode


    def forward(self, H, inds=None):
        
        if self.discard and self.discard_mode == 'AllEM' and not (self.Non_Linear and self.non_lin_mode==3):
            H=H[:, :-1, :]
            
        
        embeds = self._read_in(H)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)


        if not self.Non_Linear or (self.Non_Linear and (self.non_lin_mode==6 or self.non_lin_mode==7 or self.non_lin_mode==8 or self.non_lin_mode==11)):

            if self.control:
                if not self.discard:
                    return prediction[:, (3 * (self.state_dim) + 1)::3, :]  # predict only on xs

                elif self.discard and self.discard_mode == 'All':
                    return prediction[:, ::3, :]
                elif self.discard and self.discard_mode == 'AllEM':
                    return prediction
                elif self.discard and self.discard_mode == 'Noise':
                    return prediction[:, 2*(self.state_dim)::3, :]
            else:
                if not self.discard:
                    return prediction[:, (2*(self.state_dim)+1)::2, :]  # predict only on xs

                elif self.discard and self.discard_mode == 'All':
                    return prediction[:, ::2, :]
                
                elif self.discard and self.discard_mode == 'AllEM':
                    return prediction
                elif self.discard and self.discard_mode == 'Noise':
                    return prediction[:, (self.state_dim)::2, :]

        else:
            if self.control:
                if self.discard:
                    return prediction[:, ::3, :]
                else:
                    if self.non_lin_mode==1:
                        return prediction[:, (2 * (self.state_dim) + 1)::3, :]  # predict only on xs

                    elif self.non_lin_mode==2 or self.non_lin_mode==4 or self.non_lin_mode==5 or self.non_lin_mode==9 or self.non_lin_mode==10:
                        return prediction[:, (2 * (self.state_dim) + 1+2)::3, :]  # predict only on xs
            else:
                if self.discard:
                    # if (self.non_lin_mode==3 or self.non_lin_mode==4): # Legacy non_lin_mode 4 previously extended non_lin_mode==3 by inserting del_t
                    if (self.non_lin_mode==3):  
                        return prediction[:, ::1, :]
                    else:
                        return prediction[:, ::2, :]
                else:
                    if self.non_lin_mode==1:
                        return prediction[:, (1 * (self.state_dim) + 1)::2, :]  # predict only on xs
                    elif self.non_lin_mode==2 or self.non_lin_mode==4 or self.non_lin_mode==5 or self.non_lin_mode==9 or self.non_lin_mode==10:
                        return prediction[:, (1* (self.state_dim) + 1+2)::2, :]  # predict only on xs

                    elif self.non_lin_mode==3:
                        return prediction[:, 6::1, :]

                    # elif self.non_lin_mode==4:  # Legacy non_lin_mode 4 previously extended non_lin_mode==3 by inserting del_t
                    #     return prediction[:, 7::1, :]



