import torch.nn as nn
import torch 

class RandomConvolution(nn.Module):
    def __init__(self, *args, **kwargs):
        super(RandomConvolution, self).__init__()

    def forward(self, imgs):
        '''
        random covolution in "network randomization"
        
        (imbs): B x (C x stack) x H x W, note: imgs should be normalized and torch tensor
        '''
        
        img_h, img_w = imgs.shape[2], imgs.shape[3]
        num_stack_channel = imgs.shape[1]
        num_batch = imgs.shape[0]
        num_trans = num_batch
        batch_size = int(num_batch / num_trans)
        
        # initialize random covolution
        rand_conv = nn.Conv2d(3, 3, kernel_size=3, bias=False, padding=1)
        
        for trans_index in range(num_trans):
            torch.nn.init.xavier_normal_(rand_conv.weight.data)
            temp_imgs = imgs[trans_index*batch_size:(trans_index+1)*batch_size]
            temp_imgs = temp_imgs.reshape(-1, 3, img_h, img_w) # (batch x stack, channel, h, w)
            rand_out = rand_conv(temp_imgs)
            if trans_index == 0:
                total_out = rand_out
            else:
                total_out = torch.cat((total_out, rand_out), 0)
        total_out = total_out.reshape(-1, num_stack_channel, img_h, img_w)
        return total_out