'''
This script is used to define the trainable blocks and 
their functions for the OCO training procedure.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# TODO: add interfaces for trainable blocks into get_paras() for easier and more flexible modification

# %% trainable blocks
class Seq2Seq(nn.Module):
    def __init__(self, channel_in, height, width, filter_size):     
        super(Seq2Seq, self).__init__()

        l = height # 270

        self.conv1 = nn.Sequential(  nn.Conv2d( in_channels=channel_in,
                                                out_channels=2*channel_in,
                                                kernel_size=(filter_size, 3),
                                                stride=(1, 1),
                                                padding=(0, 1),
                                                bias=True),
                                     nn.LeakyReLU()
                                  )
        l = (l - filter_size + 1) # 240

        self.conv2 = nn.Sequential( nn.AvgPool2d(kernel_size=(2,1),
                                                 stride=(2,1)), 
                                    nn.Conv2d( in_channels=2*channel_in,
                                                out_channels=2*channel_in,
                                                kernel_size=(filter_size, 3),
                                                stride=(1, 1),
                                                padding=(0, 1),
                                                bias=True),
                                    nn.LeakyReLU()

                                   )
        l = int(l/2 - filter_size + 1) # 90

        self.conv3 = nn.Sequential( nn.AvgPool2d(kernel_size=(2,1),
                                                 stride=(2,1)),  
                                    nn.Conv2d( in_channels=2*channel_in,
                                                out_channels=channel_in,
                                                kernel_size=(filter_size, 3),
                                                stride=(1, 1),
                                                padding=(0, 0),
                                                bias=True),
                                     nn.LeakyReLU()

                                   )
        l = int(l/2 - filter_size + 1) # 15

        # TODO: add hidden size of CNN's FCN into API too
        self.fc = nn.Sequential(nn.Linear( channel_in*l*1, 8, bias=True),    
                                nn.LeakyReLU(),
                                nn.Linear( 8, channel_in*l*1, bias=True),    
                                nn.LeakyReLU(),
                                )
        
        self.Tconv1 = nn.Sequential(nn.ConvTranspose2d( in_channels=channel_in,
                                                        out_channels=2*channel_in,
                                                        kernel_size=(filter_size, 3),
                                                        stride=(1, 1),
                                                        padding=(0, 1),
                                                        bias=True),
                                    nn.LeakyReLU(),
                                    nn.Upsample(scale_factor=(2,1),
                                                mode = 'bilinear')
                                  )
        
        self.Tconv2 = nn.Sequential(nn.ConvTranspose2d( in_channels=2*channel_in,
                                                        out_channels=2*channel_in,
                                                        kernel_size=(filter_size, 3),
                                                        stride=(1, 1),
                                                        padding=(0, 1),
                                                        bias=True),
                                    nn.LeakyReLU(),
                                    nn.Upsample(scale_factor=(2,1),
                                                mode = 'bilinear')
                                  )
        
        self.Tconv3 = nn.Sequential(nn.ConvTranspose2d( in_channels=2*channel_in,
                                                        out_channels=channel_in,
                                                        kernel_size=(filter_size, 3),
                                                        stride=(1, 1),
                                                        padding=(0, 1),
                                                        bias=True),
                                    nn.Hardtanh(-4000, 4000)
                                  )
        
    def forward(self, inputs):        
        preds = None

        inputs_ = torch.mul(inputs, 255*2/np.pi)
   
        out1 = self.conv1(inputs_)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        
        batch_size, channels, height, width = out3.shape 
        out3 = out3.view(-1, channels * height * width)
        newinputs = self.fc(out3)
        newinputs = newinputs.view(batch_size, channels, height, width)

        out4 = self.Tconv1(newinputs)
        out5 = self.Tconv2(out4)
        preds = self.Tconv3(out5)
        batch_size, channels, height, width = preds.shape 
        preds = preds.view(-1, channels * height * width)
      
        return preds.float()

class LinearMap(nn.Module):
    def __init__(self, channel_in, height, width):     
        super(LinearMap, self).__init__()

        self.fc = nn.Sequential(nn.Linear( channel_in*height*width, 270, bias=True),    
                                        )
        
    def forward(self, inputs):        
        preds = None
        preds = self.fc(inputs)
        return preds.float()

class CNN(nn.Module):
    def __init__(self, channel_in, height, width, filter_size, nobias=False):     
        super(CNN, self).__init__()

        l = height # 76

        self.conv1 = nn.Sequential(  nn.Conv2d( in_channels=channel_in,
                                                out_channels=3*channel_in,
                                                kernel_size=(filter_size, 3),
                                                stride=(1, 1),
                                                padding=(0, 1),
                                                bias=True),
                                     nn.ReLU()
                                  )
        l = (l - filter_size + 1) # 67

        self.conv2 = nn.Sequential(  nn.Conv2d( in_channels=3*channel_in,
                                                out_channels=3*channel_in,
                                                kernel_size=(filter_size, 3),
                                                stride=(1, 1),
                                                padding=(0, 1),
                                                bias=True),
                                     nn.ReLU()

                                   )
        l = (l - filter_size + 1) # 58

        self.conv3 = nn.Sequential(  nn.Conv2d( in_channels=3*channel_in,
                                                out_channels=channel_in,
                                                kernel_size=(filter_size, 3),
                                                stride=(1, 1),
                                                padding=(0, 0),
                                                bias=True),
                                     nn.ReLU()

                                   )
        l = (l - filter_size + 1) # 49

        # TODO: add hidden size of CNN's FCN into API too
        if not nobias:
            self.fc = nn.Sequential(  nn.Linear( channel_in*l*1, 32, bias=True),    
                                    nn.ReLU(),
                                    nn.Linear( 32, 16, bias=True),    
                                    nn.ReLU(),    
                                    nn.Linear( 16, 8, bias=True),    
                                    nn.ReLU(), 
                                    nn.Linear( 8, 1, bias=False), 
                                    # nn.Tanh(),
                                    )
        else:
            self.fc = nn.Sequential(  nn.Linear( channel_in*l*1, 32, bias=True),    
                                    nn.ReLU(),    
                                    nn.Linear( 32, 16, bias=True),    
                                    nn.ReLU(), 
                                    nn.Linear( 16, 1, bias=False), 
                                    # nn.Tanh(),
                                    )
        
    def forward(self, inputs):        
        preds = None

        inputs_ = torch.mul(inputs, 255*2/np.pi)
   
        out1 = self.conv1(inputs_)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        
        batch_size, channels, height, width = out3.shape 
        out3 = out3.view(-1, channels * height * width)
        preds = self.fc(out3)
      
        return preds.float()

class FCN(nn.Module):
    def __init__(self, channel_in, height, width, hidden_size, out_size, last_act, linear, nobias):     
        super(FCN, self).__init__()

        self.last_act = last_act
        if self.last_act is not None:
            self.fc = nn.Sequential(  nn.Linear( channel_in*height*width, hidden_size[0], bias=True),    
                                    nn.LeakyReLU(),
                                    # TODO
                                    # nn.Linear( hidden_size[0], 16, bias=True),    
                                    # nn.ReLU(),    
                                    nn.Linear( hidden_size[0], out_size, bias=True), 
                                    nn.Tanh(),
                                    )
        else:
            if not linear:
                if not nobias:
                    self.fc = nn.Sequential(nn.Linear( channel_in*height*width, hidden_size[0], bias=True),    
                                            nn.ReLU(),
                                            # TODO
                                            # nn.Linear( hidden_size[0], hidden_size[1], bias=True),    
                                            # nn.ReLU(),     
                                            nn.Linear( hidden_size[0], out_size, bias=True),
                                            )
                else:
                    self.fc = nn.Sequential(nn.Linear( channel_in*height*width, hidden_size[0], bias=False),    
                                            nn.ReLU(),
                                            # TODO
                                            # nn.Linear( hidden_size[0], hidden_size[1], bias=True),    
                                            # nn.ReLU(),     
                                            nn.Linear( hidden_size[0], out_size, bias=False),
                                            )
            else:
                self.fc = nn.Sequential(nn.Linear( channel_in*height*width, out_size, bias=True),    
                                        )
        
    def forward(self, inputs):        
        preds = None
        preds = self.fc(inputs)
        if self.last_act is not None:
            preds = torch.mul(preds, self.last_act)
        return preds.float()

# %% related functions
def weight_init(layer):
    '''
    to initialize the weights using a chosen method
    '''
    if isinstance(layer,nn.Conv2d) or isinstance(layer,nn.Linear):
        nn.init.zeros_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

def trainable_blocks_init(flag_dof, nn_type, nr_channel, height, width, filter_size, device, hidden_size, model=None, disturbance=[None,None,None], linear=False, init_weight=False, nobias=False):
    '''
    to initialize the trainable blocks
    '''
    block_list   = []
    name_list  = []
    shape_list = []
    idx_list   = []
    idx = 0
    idx_list.append(idx)

    for index, dof in enumerate(flag_dof):
        if not dof:
            block_list.append(None)
        else:
            if nn_type=='FCN':
                block = FCN(channel_in=nr_channel, height=height, width=width, hidden_size=hidden_size, out_size=1, last_act=disturbance[index], linear=linear, nobias=nobias)
            elif nn_type=='CNN':
                block = CNN(channel_in=nr_channel, height=height, width=width, filter_size=filter_size, nobias=nobias)
            elif nn_type=='Seq2Seq':
                block = Seq2Seq(channel_in=nr_channel, height=height, width=width, filter_size=filter_size)
            elif nn_type=='LinearMap':
                block = LinearMap(channel_in=nr_channel, height=height, width=width)
            ### for pre-trained weights loading
            if model is not None:
                temp_model = model + str(index)
                temp = torch.load(temp_model)
                # block.load_state_dict(temp)
                block.load_state_dict({k.replace('FCN', 'fc'):v for k,v in temp.items()})
            ### for self-defined weights initialization
            if init_weight:
                block.apply(weight_init)
            block.to(device)
            block_list.append(block)

    for name, param in block.named_parameters():
        name_list.append(name)
        shape_list.append(param.shape)
        d_idx = len(param.data.view(-1))
        idx += d_idx
        idx_list.append(idx)

    print('the type of the trainable block: {}'.format(nn_type))
    print('the length of named parameters: {}'.format(len(shape_list)))
    print('the number of trainable parameters: {}'.format(idx_list[-1]))

    W_list = []
    for block in block_list:
        if block is None:
            W_list.append(None)
        else:    
            W = []
            [W.append(param.data.view(-1)) for param in block.parameters()]
            W = torch.cat(W)
            W_list.append(W.cpu().numpy().reshape(-1, 1))

    return block_list, shape_list, idx_list, W_list

def trainable_block_init(nn_type, nr_channel, height, width, filter_size, device, hidden_size, model=None):
    '''
    to initialize the trainable blocks
    '''
    block_list   = []
    name_list  = []
    shape_list = []
    idx_list   = []
    idx = 0
    idx_list.append(idx)

    if nn_type=='FCN':
        block = FCN(channel_in=nr_channel, height=height, width=width, hidden_size=hidden_size, out_size=3)
    elif nn_type=='Seq2Seq':
        block = Seq2Seq(channel_in=nr_channel, height=height, width=width, filter_size=filter_size)
    ### for pre-trained weights loading
    if model is not None:
        temp_model = model
        temp = torch.load(temp_model)
        block.load_state_dict(temp)
    ### for self-defined weights initialization
    # block.apply(weight_init)
    block.to(device)
    block_list.append(block)

    for name, param in block.named_parameters():
        name_list.append(name)
        shape_list.append(param.shape)
        d_idx = len(param.data.view(-1))
        idx += d_idx
        idx_list.append(idx)

    print('the type of the trainable block: {}'.format(nn_type))
    print('the length of named parameters: {}'.format(len(shape_list)))
    print('the number of trainable parameters: {}'.format(idx_list[-1]))

    W_list = []
    for block in block_list:
        W = []
        [W.append(param.data.view(-1)) for param in block.parameters()]
        W = torch.cat(W)
        W_list.append(W.cpu().numpy().reshape(-1, 1))

    return block_list, shape_list, idx_list, W_list

