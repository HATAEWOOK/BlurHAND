import torch

def normalize(param, position_norm = True, scale_norm=True, skip=False):
        if skip:
                return param
        if scale_norm:
                norm_scale = torch.norm(param[:,10,:] - param[:,9,:], dim=1) #[bs]
                if torch.any(norm_scale == 0.0):
                        norm_scale[(norm_scale == 0.0).nonzero(as_tuple=True)[0]] = 1e-5
                        print("norm_scale zero")
                param = param / norm_scale.unsqueeze(1).unsqueeze(2)
        if position_norm:
                param = param - param[:,9,:].unsqueeze(1)

        return param

def proj_func(xyz, K):
    '''
    xyz: N x num_points x 3
    K: N x 3 x 3
    '''
    uv = torch.bmm(K,xyz.permute(0,2,1))
    uv = uv.permute(0, 2, 1)
    out_uv = torch.zeros_like(uv[:,:,:2]).to(device=uv.device)
    out_uv = torch.addcdiv(out_uv, uv[:,:,:2], uv[:,:,2].unsqueeze(-1).repeat(1,1,2), value=1)
    return out_uv