'''
This script is used to define the trainable blocks and 
their functions for the OCO training procedure.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: add interfaces for trainable blocks into get_paras() for easier and more flexible modification

# %% trainable blocks
class CNN(nn.Module):
    def __init__(self, channel_in, height, width, filter_size):     
        super(CNN, self).__init__()

        l = height # 201

        self.conv1 = nn.Sequential(  nn.Conv2d( in_channels=channel_in,
                                                out_channels=3*channel_in,
                                                kernel_size=(filter_size, 3),
                                                stride=(1, 1),
                                                padding=(0, 1),
                                                bias=True),
                                     nn.ReLU()
                                  )
        l = (l - filter_size + 1) # 171

        self.conv2 = nn.Sequential(  nn.Conv2d( in_channels=3*channel_in,
                                                out_channels=3*channel_in,
                                                kernel_size=(filter_size, 3),
                                                stride=(1, 1),
                                                padding=(0, 1),
                                                bias=True),
                                     nn.ReLU()

                                   )
        l = (l - filter_size + 1) # 141

        self.conv3 = nn.Sequential(  nn.Conv2d( in_channels=3*channel_in,
                                                out_channels=channel_in,
                                                kernel_size=(filter_size, 3),
                                                stride=(1, 1),
                                                padding=(0, 0),
                                                bias=True),
                                     nn.ReLU()

                                   )
        l = (l - filter_size + 1) # 111

        self.fc = nn.Sequential(  nn.Linear( channel_in*l*1, 32, bias=True),    
                                  nn.ReLU(),    
                                  nn.Linear( 32, 1, bias=True), 
                                  # nn.Tanh(),
                                )
        
    def forward(self, inputs):        
        preds = None
   
        out1 = self.conv1(inputs)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        
        batch_size, channels, height, width = out3.shape 
        out3 = out3.view(-1, channels * height * width)
        preds = self.fc(out3)
      
        return preds.float()

class FCN(nn.Module):
    def __init__(self, channel_in, height, width, hidden_size, out_size):     
        super(FCN, self).__init__()

        self.fc = nn.Sequential(  nn.Linear( channel_in*height*width, hidden_size[0], bias=True),    
                                  nn.ReLU(),
                                  nn.Linear( hidden_size[0], hidden_size[0], bias=True),    
                                  nn.ReLU(),     
                                  nn.Linear( hidden_size[0], out_size, bias=True), 
                                  # nn.Tanh(),
                                )
        
    def forward(self, inputs):        
        preds = None
        preds = self.fc(inputs)
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

def trainable_blocks_init(flag_dof, nn_type, nr_channel, height, width, filter_size, device, hidden_size, model=None):
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
              block = FCN(channel_in=nr_channel, height=height, width=width, hidden_size=hidden_size, out_size=1)
            elif nn_type=='CNN':
              block = CNN(channel_in=nr_channel, height=height, width=width, filter_size=filter_size)
            ### for pre-trained weights loading
            if model is not None:
                temp_model = model + str(index)
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
        if block is None:
            W_list.append(None)
        else:    
            W = []
            [W.append(param.data.view(-1)) for param in block.parameters()]
            W = torch.cat(W)
            W_list.append(W.cpu().numpy().reshape(-1, 1))

    return block_list, shape_list, idx_list, W_list

def trainable_block_init(nn_type, nr_channel, height, width, device, hidden_size, model=None):
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

