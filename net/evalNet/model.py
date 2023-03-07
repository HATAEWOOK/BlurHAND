import sys
sys.path.append("/root/workplace/backup/blurHand")
from net.evalNet.config import cfg
import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import pickle
import numpy as np
from utils.util_loss import normalize, proj_func
from utils.manopth.manolayer import ManoLayer
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    AmbientLights,
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex,
    PerspectiveCameras
)
import transforms3d as t3d

def rodrigues(r):       
    theta = torch.sqrt(torch.sum(torch.pow(r, 2),1))  

    def S(n_):   
        ns = torch.split(n_, 1, 1)     
        Sn_ = torch.cat([torch.zeros_like(ns[0]),-ns[2],ns[1],ns[2],torch.zeros_like(ns[0]),-ns[0],-ns[1],ns[0],torch.zeros_like(ns[0])], 1)
        Sn_ = Sn_.view(-1, 3, 3)      
        return Sn_    

    n = r/(theta.view(-1, 1))   
    Sn = S(n) 

    #R = torch.eye(3).unsqueeze(0) + torch.sin(theta).view(-1, 1, 1)*Sn\
    #        +(1.-torch.cos(theta).view(-1, 1, 1)) * torch.matmul(Sn,Sn)
    
    I3 = Variable(torch.eye(3).unsqueeze(0).cuda())

    R = I3 + torch.sin(theta).view(-1, 1, 1)*Sn\
        +(1.-torch.cos(theta).view(-1, 1, 1)) * torch.matmul(Sn,Sn)

    Sr = S(r)
    theta2 = theta**2     
    R2 = I3 + (1.-theta2.view(-1,1,1)/6.)*Sr\
        + (.5-theta2.view(-1,1,1)/24.)*torch.matmul(Sr,Sr)
    
    idx = np.argwhere((theta<1e-30).data.cpu().numpy())

    if (idx.size):
        R[idx,:,:] = R2[idx,:,:]

    return R,Sn

def get_poseweights(poses, bsize):
    # pose: batch x 24 x 3                                                    
    pose_matrix, _ = rodrigues(poses[:,1:,:].contiguous().view(-1,3))
    #pose_matrix, _ = rodrigues(poses.view(-1,3))    
    pose_matrix = pose_matrix - Variable(torch.from_numpy(np.repeat(np.expand_dims(np.eye(3, dtype=np.float32), 0),bsize*(keypoints_num-1),axis=0)).cuda())
    pose_matrix = pose_matrix.view(bsize, -1)
    return pose_matrix

#-------------------
# Resnet + Mano
#-------------------

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class DeconvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=2, stride=1, upsample=None):
        super(DeconvBottleneck, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if stride == 1:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                   stride=stride, bias=False, padding=1)
        else:
            self.conv2 = nn.ConvTranspose2d(out_channels, out_channels,
                                            kernel_size=3,
                                            stride=stride, bias=False,
                                            padding=1,
                                            output_padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.upsample = upsample

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.upsample is not None:
            shortcut = self.upsample(x)

        out += shortcut
        out = self.relu(out)

        return out

class ResNet_Mano(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        self.device = torch.device(cfg.device)
        super(ResNet_Mano, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)       

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)

        self.fc = nn.Linear(512 * block.expansion, num_classes)                        
        self.mano = ManoLayer(side='right', mano_root="/root/workplace/backup/blurHand/models/", use_pca=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        # ------loss
        self.smoothl1_loss = torch.nn.SmoothL1Loss()
        self.mse_loss = torch.nn.MSELoss()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def rendering(self, vert, faces):
        verts_rgb = torch.ones_like(vert)[None] # (1, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb.squeeze().to(vert.device))
        vert = vert - vert.mean(1).unsqueeze(1)
        meshes = Meshes(verts=vert, faces=faces, textures=textures)
        lights = PointLights(device=vert.device, location=[[0.0, 0.0, -3.0]])
        bs = vert.shape[0]
        M_corr = np.eye(4)
        M_corr[:3, :3] = t3d.euler.euler2mat(0.0, .0, np.pi)
        M_obj2cam = np.matmul(M_corr, np.eye(4)[None])
        R = torch.FloatTensor(np.transpose(M_obj2cam[:, :3, :3], [0, 2, 1])).repeat(bs, 1, 1)
        T = torch.FloatTensor(M_obj2cam[:, :3, 3] + [[0., 0., 200]]).repeat(bs, 1)
        camera = FoVPerspectiveCameras(device=vert.device, R=R, T=T, zfar=300)
        # camera = cameras_from_opencv_projection(R = R, tvec= T, camera_matrix=Ks.to(vert.device), image_size=image_size.to(vert.device))
        raster_settings = RasterizationSettings(
                image_size=224, 
                blur_radius=0.0, 
                faces_per_pixel=1, 
        )

        renderer_img = MeshRenderer(
                rasterizer=MeshRasterizer(
                        cameras=camera, 
                        raster_settings=raster_settings
                ),
                shader=SoftPhongShader(
                        device=vert.device, 
                        cameras=camera,
                        lights=lights,
                )
        )

        re_img = renderer_img(meshes, cameras=camera, lights=lights,)

        sigma = 1e-4
        raster_settings_silhouette = RasterizationSettings(
                image_size=224, 
                blur_radius=0, 
                faces_per_pixel=50, 
        )

        renderer_silhouette = MeshRenderer(
                rasterizer=MeshRasterizer(
                        cameras=camera, 
                        raster_settings=raster_settings_silhouette
                ),
                shader=SoftSilhouetteShader()
        )
        
        re_sil = renderer_silhouette(meshes, cameras=camera, lights=lights)

        return re_img, re_sil


    def forward(self, x, target=None):
        output = {}
        if x.shape[3] != 256:
            pad = nn.ZeroPad2d(padding= (0, 32, 0, 32))
            x = pad(x)
        x = self.conv1(x) #[bs, 3, 256, 256]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) 

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)            

        x = self.avgpool(x)
        x = x.view(x.size(0), -1) 
        xs = self.fc(x)
        scale = xs[:,0]
        trans = xs[:,1:3] 
        rot = xs[:,3:6]    
        beta = xs[:,6:16]
        theta = xs[:,16:] 
        output['rot'] = rot
        output['theta'] = theta
        output['beta'] = beta

        verts, joint = self.mano(torch.concat([theta, rot], dim=1), th_betas = beta)
        # verts = x3d[:, 21:, :]
        # joint = x3d[:, :21, :]
        output['vert'] = verts
        faces = self.mano.th_faces.repeat(verts.shape[0], 1, 1)
        output['joint'] = joint
        keypt = joint[:, :, :2]
        output['keypt'] = keypt
        re_img, re_sil = self.rendering(verts, faces)
        re_sil = re_sil[:, :, :, 3]
        re_sil[re_sil != 0] = 1
        output['re_img'] = re_img
        output['re_sil'] = re_sil

        if self.training:
            loss = {}
            j3d_tar_norm = normalize(target['j3d'])
            # keypt_tar = proj_func(target['j3d'], target['Ks']) #perspective projection
            keypt_tar = normalize(target['j3d'][:, :, :2], scale_norm=False) #ortho

            loss['j3d'] = self.mse_loss(normalize(output['joint']).to(self.device), j3d_tar_norm.to(self.device))
            loss['j2d'] = self.smoothl1_loss(normalize(output['keypt'], scale_norm=False).to(self.device), keypt_tar.to(self.device))
            loss['mask'] = self.smoothl1_loss(output['re_sil'].to(self.device), target['mask'][:, 0, :, :].float().to(self.device))
            loss['reg'] = self.mse_loss(output['theta'], torch.zeros_like(output['theta']).to(self.device))
            loss['beta'] = self.mse_loss(output['beta'], torch.zeros_like(output['beta']).to(self.device))
            return loss
        else:
            return output
        
        # x = trans.unsqueeze(1) + scale.unsqueeze(1).unsqueeze(2) * x3d[:,:,:2] 
        # x = x.view(x.size(0),-1)      
              
        #x3d = scale.unsqueeze(1).unsqueeze(2) * x3d
        #x3d[:,:,:2]  = trans.unsqueeze(1) + x3d[:,:,:2] 

def resnet34_Mano(pretrained=False, **kwargs):
    
    model = ResNet_Mano(BasicBlock, [3, 4, 6, 3], **kwargs)    
    model.fc = nn.Linear(512 * 1, 61)

    return model

if __name__ == "__main__":
    model = resnet34_Mano()
    input = torch.randn(1, 3, 256, 256)
    model = model.cuda()
    input = input.cuda()
    x, x3d = model(input)
    u = np.zeros(21)   
    v = np.zeros(21)   
    for ii in range(21): 
        u[ii] = x[0,2*ii]
        v[ii] = x[0,2*ii+1]  
    print(u.shape)
    print(v.shape)


