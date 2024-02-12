import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):

    def __init__(self, channel_in, filter_size, height, width, channel_out=1, dropout=0.0,
                 length_label=1):
     
        super(CNN, self).__init__()

        # l = height #21
        # self.conv1 = nn.Sequential(  nn.Conv2d( in_channels=channel_in,
        #                                         out_channels=3*channel_in,
        #                                         kernel_size=(filter_size, 3),
        #                                         stride=(1, 1),
        #                                         padding=(0, 1),
        #                                         bias=True),
        #                              nn.ReLU()
        #                           )
        # l = (l - filter_size + 1) #19
        # self.conv2 = nn.Sequential(  nn.Conv2d( in_channels=3*channel_in,
        #                                         out_channels=channel_in,
        #                                         kernel_size=(filter_size, 3),
        #                                         stride=(1, 1),
        #                                         padding=(0, 0),
        #                                         bias=True),
        #                              nn.ReLU()

        #                            )
        # l = (l - filter_size + 1) #17
        # self.conv3 = nn.Sequential(  nn.Conv2d( in_channels=3*channel_in,
        #                                         out_channels=3* channel_in,
        #                                         kernel_size=(filter_size, 2),
        #                                         stride=(1, 1),
        #                                         padding=(0, 0),
        #                                         bias=True),
        #                              nn.ReLU()

        #                            )
        # l = (l - filter_size + 1) #15

        # self.conv4 = nn.Sequential(  nn.Conv2d( in_channels=3*channel_in,
        #                                         out_channels=1*channel_in,
        #                                         kernel_size=(filter_size, 2),
        #                                         stride=(1, 1),
        #                                         padding=(0, 0),
        #                                         bias=True),
        #                              nn.ReLU()

        #                            )
        # l = (l - filter_size + 1) #13

        # self.conv5 = nn.Sequential(  nn.Conv2d( in_channels=1*channel_in,
        #                                         out_channels=1*channel_in,
        #                                         kernel_size=(filter_size, 1),
        #                                         stride=(1, 1),
        #                                         padding=(0, 0),
        #                                         bias=True ),
        #                              nn.ReLU()

        #                            )
        # l = (l - filter_size + 1)

        # self.conv6 = nn.Sequential(  nn.Conv2d( in_channels=1*channel_in,
        #                                         out_channels=1*channel_in,
        #                                         kernel_size=(filter_size, 1),
        #                                         stride=(1, 1),
        #                                         padding=(0, 0),
        #                                         bias=True ),
        #                              nn.ReLU()

        #                            )
        # l = (l - filter_size + 1)

        # self.conv7 = nn.Sequential(  nn.Conv2d( in_channels=1,
        #                                         out_channels=1,
        #                                         kernel_size=(filter_size, 1),
        #                                         stride=(1, 1),
        #                                         padding=(0, 0),
        #                                         bias=True ),
        #                              nn.ReLU()

        #                            )
        # l = (l - filter_size + 1)

        self.fc = nn.Sequential(  nn.Linear( 1*height*width*channel_in, 64, bias=True),
                                  # nn.Dropout(dropout),     
                                  nn.ReLU(),
                                  # nn.Linear( 32, 16, bias=True),
                                  # # nn.Dropout(dropout),     
                                  # nn.ReLU(),     
                                  nn.Linear( 64, 1, bias=True),
                                  # nn.Dropout(dropout),     
                                  # nn.ReLU(),        
                                  # nn.Linear( 8, 4, bias=True) ,
                                  # nn.Dropout(dropout),
                                  # nn.ReLU(),
                                  # nn.Linear( 32, length_label * channel_out, bias=True ), 
                                  # nn.Tanh(),
                                #  nn.Hardtanh(-0.9,0.9)
                                )
        
    def forward(self, inputs):
        """
        Forward pass of the Neural Network. Should not be called manually but by
        calling a model instance directly.

        Inputs:
        - inputs: PyTorch input Variable
        """
        
        preds = None
   
        # out1 = self.conv1(inputs)
        # out2 = self.conv2(out1)
        # out = self.conv3(out)
        # out = self.conv4(out)
        # out = self.conv5(out)
        # out = self.conv6(out)
        # out = self.conv7(out)
        
        # batch_size, channels, height, width = inputs.shape 
        # out2 = out2.view(-1, channels * height * width)
        # inputs = inputs.view(-1, channels * height * width)
        preds = self.fc(inputs)
      
        return preds.float()


