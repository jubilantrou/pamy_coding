import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self, in_channel=1, out_channel=1, dropout=0.0,
                 height=3, filter_size=21, width=100, min_val=-1.0, max_val=1.0):
     
        super(CNN, self).__init__()

        l = width
        # self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channel,
        #                                      out_channels=2*in_channel,
        #                                      kernel_size=(1, filter_size),
        #                                      stride=(1, 1),
        #                                      padding=(0, 0),
        #                                      bias=True),
        #                             nn.ELU()
        #                           )
        # l = l-filter_size+1
        # self.conv2 = nn.Sequential(nn.Conv2d(in_channels=2*in_channel,
        #                                      out_channels=3*in_channel,
        #                                      kernel_size=(1, filter_size),
        #                                      stride=(1, 1),
        #                                      padding=(0, 0),
        #                                      bias=True),
        #                             # nn.MaxPool2d((1, 5), stride=(1, 5)),
        #                             nn.ELU()
        #                            )
        # l = (l-filter_size+1)
        # self.conv3 = nn.Sequential( nn.Conv2d(in_channels=4*in_channel,
        #                                       out_channels=8*in_channel,
        #                                       kernel_size=(1, filter_size),
        #                                       stride=(1, 1),
        #                                       padding=(0, 0),
        #                                       bias=True),
        #                             # nn.MaxPool2d((1, 5), stride=(1, 5)),
        #                             nn.ELU()
        #                            )
        # l = (l-filter_size+1)
        # self.conv4 = nn.Sequential(nn.Conv2d( in_channels=8*in_channel,
        #                                       out_channels=16*in_channel,
        #                                       kernel_size=(1, filter_size),
        #                                         stride=(1, 1),
        #                                         padding=(0, 0),
        #                                         bias=True),
        #                             nn.ELU()
        #                            )
        # l = (l - filter_size + 1)

        # self.conv5 = nn.Sequential(  nn.Conv2d( in_channels=1*in_channel,
        #                                         out_channels=1*in_channel,
        #                                         kernel_size=(2, filter_size),
        #                                         stride=(1, 1),
        #                                         padding=(0, 0),
        #                                         bias=True),
        #                              nn.ReLU()
        #                            )
        # l = (l-filter_size+1)

        # self.conv6 = nn.Sequential(  nn.Conv2d( in_channels=1*in_channel,
        #                                         out_channels=1*in_channel,
        #                                         kernel_size=(1, filter_size),
        #                                         stride=(1, 1),
        #                                         padding=(0, 0),
        #                                         bias=True),
        #                              nn.ReLU()

        #                            )
        # l = (l-filter_size+1)

        self.fc = nn.Sequential(nn.Linear(in_channel*height*width, 1, bias=True) ,
                                # nn.Dropout(dropout),     
                                # nn.ELU(),           
                                # nn.Linear(64, 1, bias=True),
                                # nn.Dropout(dropout),
                                # nn.ReLU(),
                                # nn.Tanh()
                                nn.Hardtanh(min_val=min_val, max_val=max_val)
                                )
        
    def forward(self, inputs):
        """
        Forward pass of the Neural Network. Should not be called manually but by
        calling a model instance directly.

        Inputs:
        - inputs: PyTorch input Variable
        """
        
        preds = None
   
        # out = self.conv1(inputs)
        # out = self.conv2(out)
        # out = self.conv3(out)
        # out = self.conv4(out)
        # out = self.conv5(out)
        # out = self.conv6(out)
        
        batch_size, channels, height, width = inputs.shape 
        out = inputs.view(-1, channels*height*width)
        preds = self.fc(out)
      
        return preds.float()


