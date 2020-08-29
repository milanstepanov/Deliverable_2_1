# Implementation of the view synthesis approach following the paper 
# "Learning Occlusion-Aware View Synthesis for Light Fields" by Navarro et
# al.
try:
    import os
    del os.environ['MKL_NUM_THREADS']
except:
    print("")
    
import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
cuda = torch.cuda.is_available()


def conv_elu_bn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.ELU(),
                         nn.BatchNorm2d(out_planes))


class FeatureExtraction(nn.Module):

    def __init__(self, num_filters=32, dilation=1):
        super(FeatureExtraction, self).__init__()

        self.layer0 = nn.Sequential(conv_elu_bn(5, num_filters, 3, 1, 1, dilation))
        self.layer1 = nn.Sequential(conv_elu_bn(num_filters, num_filters, 3, 1, 1, dilation))
        self.layer2 = nn.Sequential(conv_elu_bn(num_filters, num_filters, 3, 1, 1, dilation))
        self.layer3 = nn.Sequential(conv_elu_bn(num_filters, num_filters, 3, 1, 1, dilation))
        self.layer4 = nn.Sequential(conv_elu_bn(num_filters, num_filters, 3, 1, 1, dilation))

        self.pool16 = nn.AvgPool2d(kernel_size=(17, 17), padding=(8, 8), stride=(1, 1))
        self.pool8 = nn.AvgPool2d(kernel_size=(9, 9), padding=(4, 4), stride=(1, 1))

        self.layer5 = nn.Sequential(conv_elu_bn(4 * num_filters, num_filters, 3, 1, 1, dilation))

    def forward(self, x, to_print=False):
        if to_print:
            print(x.shape)

        output = self.layer0(x)
        output = self.layer1(output)
        output_raw = self.layer2(output)
        if to_print:
            print("Layer 2:", output_raw.shape)
        output = self.layer3(output_raw)
        output_skip = self.layer4(output)
        if to_print:
            print("Layer 4:", output_skip.shape)

        output_pool16 = self.pool16(output_skip)
        if to_print:
            print("Pool 0:", output_pool16.shape)
        output_pool8 = self.pool8(output_skip)
        if to_print:
            print("Pool 1:", output_pool8.shape)

        output = torch.cat((output_raw, output_skip, output_pool16, output_pool8), dim=1)
        if to_print:
            print("Concatenated:", output.shape)
        output = self.layer5(output)
        if to_print:
            print("Out:", output.shape)

        return output

    def test(self):
        # test the forward pass
        img_size = 64
        x = torch.as_tensor(np.random.randint(0, 255 + 1,
                                              (1, 5, img_size, img_size)).astype(np.float32))

        if torch.cuda.is_available() and use_cuda:
            x = x.cuda()

        return self.forward(x)


def conv_tanh(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.Tanh())


class DisparityEstimation(nn.Module):

    def __init__(self, dmax=4., num_inp_channels=130, num_dilated_channels=128, num_channels=64, num_out_channels=4):
        super(DisparityEstimation, self).__init__()

        self.dmax = dmax

        self.layer0 = nn.Sequential(conv_elu_bn(num_inp_channels, num_dilated_channels, 3, 1, 1, dilation=2))
        self.layer1 = nn.Sequential(conv_elu_bn(num_dilated_channels, num_dilated_channels, 3, 1, 1, dilation=4))
        self.layer2 = nn.Sequential(conv_elu_bn(num_dilated_channels, num_dilated_channels, 3, 1, 1, dilation=8))
        self.layer3 = nn.Sequential(conv_elu_bn(num_dilated_channels, num_dilated_channels, 3, 1, 1, dilation=16))

        self.layer4 = nn.Sequential(conv_elu_bn(num_dilated_channels, num_channels, 3, 1, 1, dilation=1))
        self.layer5 = nn.Sequential(conv_elu_bn(num_channels, num_channels, 3, 1, 1, dilation=1))

        self.layer6 = nn.Sequential(conv_tanh(num_channels, num_out_channels, 3, 1, 1, dilation=1))

    def forward(self, x, to_print=False):
        if to_print:
            print(x.shape)

        output = self.layer0(x)
        if to_print:
            print("Layer 0:", output.shape)
        output = self.layer1(output)
        if to_print:
            print("Layer 1:", output.shape)
        output = self.layer2(output)
        if to_print:
            print("Layer 2:", output.shape)
        output = self.layer3(output)
        if to_print:
            print("Layer 3:", output.shape)
        output = self.layer4(output)
        if to_print:
            print("Layer 4:", output.shape)
        output = self.layer5(output)
        if to_print:
            print("Layer 5:", output.shape)
        output = self.layer6(output)
        if to_print:
            print("Layer 6:", output.shape)

        output = self.dmax * output

        return output

    def test(self, use_cuda):
        # test the forward pass
        img_size = 64
        x = torch.as_tensor(np.random.randint(0, 255 + 1, (1, 130, img_size, img_size)))
        if torch.cuda.is_available() and use_cuda:
            x = x.cuda()

        return self.forward(x)


class SelectionEstimation(nn.Module):

    def __init__(self, num_inp_channels=18, num_filters=128, num_out_channels=4):
        super(SelectionEstimation, self).__init__()

        self.layer0 = nn.Sequential(conv_elu_bn(num_inp_channels, num_filters // 2, 3, 1, 1, dilation=1))
        self.layer1 = nn.Sequential(conv_elu_bn(num_filters // 2, num_filters, 3, 1, 1, dilation=1))
        self.layer2 = nn.Sequential(conv_elu_bn(num_filters, num_filters, 3, 1, 1, dilation=1))
        self.layer3 = nn.Sequential(conv_elu_bn(num_filters, num_filters, 3, 1, 1, dilation=1))
        self.layer4 = nn.Sequential(conv_elu_bn(num_filters, num_filters // 2, 3, 1, 1, dilation=1))
        self.layer5 = nn.Sequential(conv_elu_bn(num_filters // 2, num_filters // 4, 3, 1, 1, dilation=1))

        self.layer6 = nn.Sequential(conv_tanh(num_filters // 4, num_out_channels, 3, 1, 1, dilation=1))

        self.softmax = nn.Softmax(dim=1)

        self.beta = nn.Parameter(data=torch.tensor(1.))  # requires_grad is True by default for Parameter

    def forward(self, x, to_print=False):
        if to_print:
            print(x.shape)

        output = self.layer0(x)
        if to_print:
            print("Layer 0:", output.shape)
        output = self.layer1(output)
        if to_print:
            print("Layer 1:", output.shape)
        output = self.layer2(output)
        if to_print:
            print("Layer 2:", output.shape)
        output = self.layer3(output)
        if to_print:
            print("Layer 3:", output.shape)
        output = self.layer4(output)
        if to_print:
            print("Layer 4:", output.shape)
        output = self.layer5(output)
        if to_print:
            print("Layer 5:", output.shape)
        output = self.layer6(output)
        if to_print:
            print("Layer 6:", output.shape)

        output = self.softmax(self.beta * output)

        return output

    def test(self):
        # test the forward pass
        img_size = 64
        x = torch.as_tensor(np.random.randint(0, 255 + 1, (1, 18, img_size, img_size)))

        if torch.cuda.is_available() and use_cuda:
            x = x.cuda()

        return self.forward(x)


class OcclusionAwareVS(nn.Module):

    def __init__(self, angular, dmax, use_cuda):
        super(OcclusionAwareVS, self).__init__()

        self.fen = FeatureExtraction()
        self.den = DisparityEstimation(dmax=dmax)
        self.sen = SelectionEstimation()

        self.use_cuda = use_cuda

        self.min_view = 0
        self.max_view = angular - 1


        if torch.cuda.is_available() and use_cuda:
            self.fen = self.fen.cuda()
            self.den = self.den.cuda()
            self.sen = self.sen.cuda()

    def forward(self, pp, qq, c1, c2, c3, c4):

        batch, channels, height, width = c1.shape

        # prepare inputs
        pp = np.array(pp).astype(np.float32)
        qq = np.array(qq).astype(np.float32)
        pp = torch.as_tensor(pp).reshape((-1, 1, 1, 1))
        qq = torch.as_tensor(qq).reshape((-1, 1, 1, 1))
        p = pp*torch.ones(size=(batch,1,height,width))
        q = qq*torch.ones(size=(batch,1,height,width))

        c1 = torch.from_numpy(np.array(c1).astype(np.float32))
        c2 = torch.from_numpy(np.array(c2).astype(np.float32))
        c3 = torch.from_numpy(np.array(c3).astype(np.float32))
        c4 = torch.from_numpy(np.array(c4).astype(np.float32))
        

        if torch.cuda.is_available() and self.use_cuda:
            p = p.cuda()
            q = q.cuda()
            c1 = c1.cuda()
            c2 = c2.cuda()
            c3 = c3.cuda()
            c4 = c4.cuda()

        # Feature extraction
        fen_input_1 = torch.cat([p, q, c1], dim=1)
        fen_input_2 = torch.cat([p, q, c2], dim=1)
        fen_input_3 = torch.cat([p, q, c3], dim=1)
        fen_input_4 = torch.cat([p, q, c4], dim=1)

        # Get features from each view
        features_1 = self.fen(fen_input_1)
        features_2 = self.fen(fen_input_2)
        features_3 = self.fen(fen_input_3)
        features_4 = self.fen(fen_input_4)

        # Depth estimation
        depth_features = torch.cat([p, q, features_1, features_2, features_3, features_4], dim=1)
        disparities = self.den(depth_features)

        # warping
        views = torch.cat([pp, qq], dim=3)
        mesh = torch.nn.functional.affine_grid(torch.as_tensor(batch*[[[1,0,0],[0,1,0]]]).type(torch.FloatTensor),
                                               size=(batch,1,height,width),
                                               align_corners=True)
        
        if torch.cuda.is_available() and self.use_cuda:
            mesh = mesh.cuda()

        # relation  (x, y)
        #           (p, q)
        # disparity (0, 0)
        disparity_map_idx = 0
        disparity = disparities[:, disparity_map_idx:disparity_map_idx + 1, :, :]
        disparity = disparity.permute(0, 2, 3, 1)
        # Seen as a way to scale pixels in the same way they were scaled when generating the grid
        # Image of width N has pixels in range [0, N-1] which translates to [0, 1] range by dividing with N-1
        # As we move to [-1, 1] range there is additional multiplication with 2
        # Final translation to the desired range is obtained by substracting 1 which does not change the distance
        # between pixels
        disparity = torch.cat([disparity/(width-1)*2, disparity/(height-1)*2], dim=3)
        
        view_offset = torch.as_tensor(
            np.array([self.min_view, self.min_view], dtype=np.float32).reshape(1, 1, 1, -1)).repeat(batch, 1, 1, 1) - views

        if torch.cuda.is_available() and self.use_cuda:
            view_offset = view_offset.cuda()
            
        c1_w = torch.nn.functional.grid_sample(c1,
                                               mesh + view_offset * disparity,
                                               align_corners=True)

        # disparity (0, N-1)
        disparity_map_idx = 1
        disparity = disparities[:, disparity_map_idx:disparity_map_idx + 1, :, :]
        disparity = disparity.permute(0, 2, 3, 1)
        disparity = torch.cat([disparity/(width-1)*2, disparity/(height-1)*2], dim=3)
        
        view_offset = torch.as_tensor(
            np.array([self.min_view, self.max_view], dtype=np.float32).reshape(1, 1, 1, -1)).repeat(batch, 1, 1, 1) - views
        
        if torch.cuda.is_available() and self.use_cuda:
            view_offset = view_offset.cuda()
        c2_w = torch.nn.functional.grid_sample(c2,
                                               mesh + view_offset * disparity,
                                               align_corners=True)

        # disparity (N-1, 0)
        disparity_map_idx = 2
        disparity = disparities[:, disparity_map_idx:disparity_map_idx + 1, :, :]
        disparity = disparity.permute(0, 2, 3, 1)
        disparity = torch.cat([disparity/(width-1)*2, disparity/(height-1)*2], dim=3)
        
        view_offset = torch.as_tensor(
            np.array([self.max_view, self.min_view], dtype=np.float32).reshape(1, 1, 1, -1)).repeat(batch, 1, 1, 1) - views
        if torch.cuda.is_available() and self.use_cuda:
            view_offset = view_offset.cuda()
            
        c3_w = torch.nn.functional.grid_sample(c3,
                                               mesh + view_offset * disparity,
                                               align_corners=True)

        # disparity (N-1, N-1)
        disparity_map_idx = 3
        disparity = disparities[:, disparity_map_idx:disparity_map_idx + 1, :, :]
        disparity = disparity.permute(0, 2, 3, 1)
        disparity = torch.cat([disparity/(width-1)*2, disparity/(height-1)*2], dim=3)
        
        view_offset = torch.as_tensor(
            np.array([self.max_view, self.max_view], dtype=np.float32).reshape(1, 1, 1, -1)).repeat(batch, 1, 1, 1) - views
        if torch.cuda.is_available() and self.use_cuda:
            view_offset = view_offset.cuda()
            
        c4_w = torch.nn.functional.grid_sample(c4,
                                               mesh + view_offset * disparity,
                                               align_corners=True)

        warped_views = torch.cat([c1_w, c2_w, c3_w, c4_w], dim=1)

        #print("Disparity")

        # Selection estimation
        m = self.sen(torch.cat([warped_views, disparities, p, q], dim=1))

        w_n, w_c, w_h, w_w = warped_views.size()
        prediction = (warped_views.reshape(w_n, 4, w_c // 4, w_h, w_w) * m.reshape((w_n,4,1,w_h,w_w))).sum(1)

        #print("Selection")

        return prediction, warped_views, m, disparities
