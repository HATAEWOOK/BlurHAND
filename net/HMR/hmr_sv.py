from pickle import NONE
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
sys.path.append(".")
sys.path.append("..")
from net.NET_HG.net_hg import Net_HM_HG
from utils.linear_model import LinearModel
from net.HMR.config import cfg
from utils.resnet import resnet34, resnet50
from utils.layer import Conv1dLayer
from utils.v2v import V2FModel
from utils import op, volumetric
from utils.manopth.manopth.manolayer import ManoLayer
from utils.utils_net import compute_uv_from_integral
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

class RGB2HM(nn.Module):
    def __init__(self):
        super(RGB2HM, self).__init__()
        num_joints = 21
        self.net_hm = Net_HM_HG(num_joints,
                                num_stages=2,
                                num_modules=2,
                                num_feats=256)
    def forward(self, images):
        # 1. Heat-map estimation
        est_hm_list, encoding = self.net_hm(images)
        return est_hm_list, encoding

class Regressor(LinearModel):
    def __init__(self, fc_layers, use_dropout, drop_prob, use_ac_func, num_param, num_iters, max_batch_size):
        super(Regressor, self).__init__(fc_layers, use_dropout, drop_prob, use_ac_func)

        self.num_param = num_param
        self.num_iters = num_iters
        mean = np.zeros(self.num_param, dtype=np.float32)
        mean_param = np.tile(mean, max_batch_size).reshape((max_batch_size, -1)) #[bs, num_param]
        self.register_buffer('mean_param', torch.from_numpy(mean_param).float())

    def forward(self, inputs):
        """
        input : output of encoder which has 2048 features
        return: list of params, 
        """
        params = []
        bs = inputs.shape[0] #batch size
        param = self.mean_param[:bs, :] #[bs, num_param]

        for _ in range(self.num_iters):
            total = torch.cat([inputs, param], dim=1)
            param = param + self.fc_blocks(total)
            params.append(param)
            if torch.any(torch.isnan(param)):
                print('regressor')

        return params

class PoseNet(nn.Module):
    def __init__(self, joint_num):
        super(PoseNet, self).__init__()
        self.joint_num = joint_num
        # # self.deconv = make_deconv_layers([2048,256,256,256])
        # self.deconv = make_deconv_layers([512,256,256,256])
        # self.conv_x = make_conv1d_layers([256,self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        # self.conv_y = make_conv1d_layers([256,self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        # # self.conv_z_1 = make_conv1d_layers([2048,256*64], kernel=1, stride=1, padding=0)
        # self.conv_z_1 = make_conv1d_layers([512,256*64], kernel=1, stride=1, padding=0)
        # self.conv_z_2 = make_conv1d_layers([256,self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)

        # self.deconv = DeconvLayer([512,256,256,256])
        self.conv_x = Conv1dLayer([256,self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.conv_y = Conv1dLayer([256,self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.conv_z_1 = Conv1dLayer([512,256*64], kernel=1, stride=1, padding=0)
        self.conv_z_2 = Conv1dLayer([256,self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)

    def soft_argmax_1d(self, heatmap1d):
        heatmap1d = F.softmax(heatmap1d, 2)
        heatmap_size = heatmap1d.shape[2]
        coord = heatmap1d * torch.arange(heatmap_size).float().cuda()
        coord = coord.sum(dim=2, keepdim=True)
        return coord

    def forward(self, img_feat, encoding):
        # img_feat_xy = self.deconv(img_feat) #[bs, 256, 64, 64]
        img_feat_xy = encoding #[bs, 256, 64, 64]

        # x axis
        img_feat_x = img_feat_xy.mean((2)) #[bs, 256, 64]
        heatmap_x = self.conv_x(img_feat_x) #[bs, 21, 64]
        coord_x = self.soft_argmax_1d(heatmap_x)
        
        # y axis
        img_feat_y = img_feat_xy.mean((3))
        heatmap_y = self.conv_y(img_feat_y) #[bs, 21, 64]
        coord_y = self.soft_argmax_1d(heatmap_y)
        
        # z axis
        img_feat_z = img_feat.mean((2,3))[:,:,None] #[bs, 2048, 1]
        img_feat_z = self.conv_z_1(img_feat_z) #[bs, 16384, 1]
        img_feat_z = img_feat_z.view(-1,256,64) #[bs, 256, 64]
        heatmap_z = self.conv_z_2(img_feat_z) #[bs, 21, 64]
        coord_z = self.soft_argmax_1d(heatmap_z)

        joint_coord = torch.cat((coord_x, coord_y, coord_z),2)
        # print(joint_coord.shape)
        img_feat_xyz = img_feat_xy.unsqueeze(-1)
        img_feat_xyz = img_feat_xyz.repeat(1, 1, 1, 1, 64)
        for b_idx in range(img_feat_xyz.shape[0]):
            for ch in range(img_feat_xyz.shape[1]):
                img_feat_xyz[b_idx, ch, :, :, :] *= img_feat_z[b_idx, ch, :] #[bs, 256, 64, 64, 64]

        return joint_coord, img_feat_xyz

class HMR_SV(nn.Module):
    def __init__(self):
        super(HMR_SV, self).__init__()

        self.num_param = 61
        self.max_batch_size = 300
        if cfg.pretrained:
            self.resnet_backbone = resnet34(pretrained=True)
        else:
            self.resnet_backbone = resnet34()
        num_features = 1024
        self.posenet = PoseNet(21)

        self.process_features = nn.Sequential(
            nn.Conv3d(256, 32, 1)
        )

        self.volume_net = V2FModel(32, 512)
        self.volume_aggregation_method = "softmax"
        self.volume_softmax = True
        self.volume_multiplier = 1.0
        self.volume_size = 64
        self.cuboid_side = 400

        self.regressor = Regressor(
            fc_layers  =[num_features+self.num_param, 
                        int(num_features),
                        int(num_features), 
                        self.num_param],
            use_dropout=[True,True,False], 
            drop_prob  =[0.5, 0.5, 0.0],
            use_ac_func=[True,True,False],
            num_param  =self.num_param,
            num_iters  =3,
            max_batch_size=self.max_batch_size
        )

        self.mano = ManoLayer(side='right', mano_root="/root/workplace/backup/blurhand/model", use_pca=False)
        self.rgb2hm = RGB2HM()

    def compute_results(self, param):
        scale = param[:, 0].contiguous()    # [bs]    Scaling 
        trans = param[:, 1:3].contiguous()  # [bs,2]  Translation in x and y only
        rvec  = param[:, 3:6].contiguous()  # [bs,3]  Global rotation vector
        beta  = param[:, 6:16].contiguous() # [bs,10] Shape parameters
        theta   = param[:, 16:].contiguous()  # [bs,45] Angle parameters

        # mano_output = self.mano(beta, rvec, theta)
        th_verts, th_jtr = self.mano(torch.concat([theta, rvec], dim=1), th_betas=beta)

        verts = th_verts
        faces = self.mano.th_faces.repeat(verts.shape[0], 1, 1)
        joint = th_jtr
        pose = self.mano.tsa_poses
        pose = pose.view(-1, 16, 3)
        # verts = torch.bmm(verts, r_y.repeat(verts.shape[0],1,1).to(verts.device))
        scale_ = torch.abs(scale).contiguous().view(-1, 1, 1)
        verts = verts * scale_
        joint = joint * scale_
        joint = joint - joint[:, 9, :].unsqueeze(1)

        return joint, verts, faces, theta, beta, scale, trans, rvec, pose

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

    def get_volume(self, target_j3d, batch_size, device):
        cuboids = []
        base_points = torch.zeros(batch_size, 3, device=device)
        coord_volumes = torch.zeros(batch_size, self.volume_size, self.volume_size, self.volume_size, 3, device=device)
        for batch_i in range(batch_size):
            keypoints_3d = target_j3d[batch_i]
            base_point = keypoints_3d[9, :3]
            base_points[batch_i] = base_point.to(device)
            sides = np.array([self.cuboid_side, self.cuboid_side, self.cuboid_side])
            position = base_point.cpu().numpy() - sides/2
            cuboid = volumetric.Cuboid3D(position, sides)
            cuboids.append(cuboid)
            xxx, yyy, zzz = torch.meshgrid(torch.arange(self.volume_size, device=device), torch.arange(self.volume_size, device=device), torch.arange(self.volume_size, device=device))
            grid = torch.stack([xxx, yyy, zzz], dim=-1).type(torch.float)
            grid = grid.reshape((-1, 3))
            grid_coord = torch.zeros_like(grid)
            grid_coord[:, 0] = position[0] + (sides[0] / (self.volume_size - 1)) * grid[:, 0]
            grid_coord[:, 1] = position[1] + (sides[1] / (self.volume_size - 1)) * grid[:, 1]
            grid_coord[:, 2] = position[2] + (sides[2] / (self.volume_size - 1)) * grid[:, 2]
            coord_volume = grid_coord.reshape(self.volume_size, self.volume_size, self.volume_size, 3)
            coord_volumes[batch_i] = coord_volume

        return coord_volumes

    def forward(self, input, target_j3d):
        if input.shape[3] != 256:
            pad = nn.ZeroPad2d(padding= (0, 32, 0, 32))
            input_hm = pad(input)
        output = {}
        _, img_feature, _ = self.resnet_backbone(input_hm) #[bs, 64, 64, 64], [bs, 2048, 8, 8], [bs, 2048, 1, 1]
        del _
        hm_list, encoding = self.rgb2hm(input_hm) #{[bs,21,64,64]}, {[bs,256,64,64]}
        hm_keypt = compute_uv_from_integral(hm_list[-1], input.shape[2:4]) #[bs, 21, 3] (0, 224)
        output['hm_keypt'] = hm_keypt
        joint_coords, feature_xyz = self.posenet(img_feature, encoding[-1]) #[bs, 21, 3], [bs, 256, 64, 64, 64]
        # joint_coords = self.posenet(img_feature) #[bs, 21, 3], [bs, 256, 64, 64, 64]
        feature_xyz = self.process_features(feature_xyz)
        # # print("posenet")
        # # GPUtil.showUtilization()
        regressor_feature, heatmap_3d = self.volume_net(feature_xyz)
        # # print("volumenet")
        # # GPUtil.showUtilization()
        coord_volumes = self.get_volume(target_j3d, img_feature.shape[0], img_feature.device)
        vol_joint_3d, _ = op.integrate_tensor_3d_with_coordinates(heatmap_3d * self.volume_multiplier, coord_volumes, softmax=self.volume_softmax)
        del _
        output['posenet_joint'] = joint_coords
        output['vol_joint'] = vol_joint_3d

        return output

if __name__ == "__main__":
    input = torch.randn(2, 3, 224, 224).to("cuda")
    target_3d = torch.randn(2, 21, 3).to("cuda")
    net = HMR_SV().to("cuda")
    output = net(input, target_3d)
    for k in output.keys():
        print(output[k].shape)
