# Heather Fryling
# 1/8/2023
# Neural net used for initial 64x64 fish/no fish experiments.

import torch.nn as nn
import torch.nn.functional as F

class FishNet64(nn.Module):

    # the init method defines the layers of the network
    def __init__(self, dropout_perc = 0.7):
        self.dropout_percent = dropout_perc

        # create all of the layers that have to store information
        super(FishNet64, self).__init__() # input are 64 x 64 images
        self.conv0 = nn.Conv2d( 3, 16, kernel_size=5 ) # reduce to 60 x 60 images, 3 input channels, 16 output channels
        # will insert a max pooling layer in between to get to 30 x 30
        self.conv1 = nn.Conv2d( 16, 32, kernel_size=5 ) # reduce to 26 x 26 images, 16 input channels, 32 output channels
        # will insert a max pooling layer in between to get to 13 x 13
        self.conv2 = nn.Conv2d( 32, 32, kernel_size=5 ) # reduce to 9 x 9 images, 32 input channels, 32 output channels
        # will insert a max pooling layer in between to get to 4 x  4 x 32 channels = 512 outputs
        self.conv2_drop = nn.Dropout2d(self.dropout_percent) # dropout layer (default % dropout?)
        self.fc1 = nn.Linear( 512, 64 ) # 512 input signals, 64 nodes
        self.fc2 = nn.Linear( 64, 2 ) # 64 inputs, 2 outputs

        return

    # execute a forward pass
    def forward( self, x ):
        x = F.relu( F.max_pool2d( self.conv0(x), 2 ) ) # relu on max pooled results of conv0
        x = F.relu( F.max_pool2d( self.conv1(x), 2 ) ) # relu on max pooled results of conv1
        x = F.relu( F.max_pool2d( self.conv2_drop( self.conv2(x)), 2 ) ) # relu on max pooled results of dropout of conv2       
        x = x.view( -1, 512 ) # converts the array to a linear 512 nodes
        x = F.relu( self.fc1(x) ) # relu of fully connected layer
        x = self.fc2( x ) # final output layer
        return F.log_softmax( x, dim=1 ) # return softmax of the output layer (batch x 1 dim array)
