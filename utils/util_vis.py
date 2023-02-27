from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import os
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from opendr.camera import ProjectPoints
from opendr.renderer import ColoredRenderer, TexturedRenderer
from opendr.lighting import LambertianPointLight, SphericalHarmonics
import torch
import trimesh

class Renderer(object):
    """
    Render mesh using OpenDR for visualization.
    """

    def __init__(self, width=600, height=600, near=0.5, far=1000, faces=None):
        self.colors = {'hand': [.9, .9, .9], 'pink': [.9, .7, .7], 'light_blue': [0.65098039, 0.74117647, 0.85882353] }
        self.width = width
        self.height = height
        self.faces = faces
        self.renderer = ColoredRenderer()

    def render(self, vertices, faces=None, img=None,
               camera_t=np.zeros([3], dtype=np.float32),
               camera_rot=np.zeros([3], dtype=np.float32),
               camera_center=None,
               use_bg=False,
               bg_color=(0.0, 0.0, 0.0),
               body_color=None,
               focal_length=5000,
               disp_text=False,
               gt_keyp=None,
               pred_keyp=None,
               **kwargs):
        if img is not None:
            height, width = img.shape[:2]
        else:
            height, width = self.height, self.width

        if faces is None:
            faces = self.faces

        if camera_center is None:
            camera_center = np.array([width * 0.5,
                                      height * 0.5])

        self.renderer.camera = ProjectPoints(rt=camera_rot,
                                             t=camera_t,
                                             f=focal_length * np.ones(2),
                                             c=camera_center,
                                             k=np.zeros(5))
        dist = np.abs(self.renderer.camera.t.r[2] -
                      np.mean(vertices, axis=0)[2])
        far = dist + 20

        self.renderer.frustum = {'near': 1.0, 'far': far,
                                 'width': width,
                                 'height': height}

        if img is not None:
            if use_bg:
                self.renderer.background_image = img
            else:
                self.renderer.background_image = np.ones_like(
                    img) * np.array(bg_color)

        if body_color is None:
            color = self.colors['light_blue']
        else:
            color = self.colors[body_color]

        if isinstance(self.renderer, TexturedRenderer):
            color = [1.,1.,1.]

        self.renderer.set(v=vertices, f=faces,
                          vc=color, bgcolor=np.ones(3))
        albedo = self.renderer.vc
        # Construct Back Light (on back right corner)
        yrot = np.radians(120)

        self.renderer.vc = LambertianPointLight(
            f=self.renderer.f,
            v=self.renderer.v,
            num_verts=self.renderer.v.shape[0],
            light_pos=rotateY(np.array([-200, -100, -100]), yrot),
            vc=albedo,
            light_color=np.array([1, 1, 1]))

        # Construct Left Light
        self.renderer.vc += LambertianPointLight(
            f=self.renderer.f,
            v=self.renderer.v,
            num_verts=self.renderer.v.shape[0],
            light_pos=rotateY(np.array([800, 10, 300]), yrot),
            vc=albedo,
            light_color=np.array([1, 1, 1]))

        #  Construct Right Light
        self.renderer.vc += LambertianPointLight(
            f=self.renderer.f,
            v=self.renderer.v,
            num_verts=self.renderer.v.shape[0],
            light_pos=rotateY(np.array([-500, 500, 1000]), yrot),
            vc=albedo,
            light_color=np.array([.7, .7, .7]))

        return self.renderer.r

def concat_meshes(mesh_list):
    '''manually concat meshes'''
    cur_vert_number = 0
    cur_face_number = 0
    verts_list = []
    faces_list = []
    for idx, m in enumerate(mesh_list):
        verts_list.append(m.vertices)
        faces_list.append(m.faces + cur_vert_number)
        cur_vert_number += len(m.vertices)

    combined_mesh = trimesh.Trimesh(np.concatenate(verts_list),
        np.concatenate(faces_list), process=False
    )
    return combined_mesh

def create_skeleton_mesh(joints):
    mesh_list = []
    # Sphere for joints
    for idx, j in enumerate(joints):
        joint_sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.005)
        joint_sphere.vertices += j.detach().cpu().numpy()
        mesh_list.append(joint_sphere)
    
    parent = np.array([0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])
    # Cylinder for bones
    for idx in range(1, 21):
        ed = joints[idx].detach().cpu().numpy()
        st = joints[parent[idx]].detach().cpu().numpy()
        skel = trimesh.creation.cylinder(0.003, segment=(st, ed))
        mesh_list.append(skel)

    skeleton_mesh = concat_meshes(mesh_list)
    return skeleton_mesh

def rotateY(points, angle):
    ry = np.array([
        [np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
        [-np.sin(angle), 0., np.cos(angle)]
    ])
    return np.dot(points, ry)

def render_handjoint(kps,  renderer, out_img_path=None):
    hand_joints = torch.from_numpy(kps)
    # hand_joints = kps

    # render HALO
    camera_t = np.array([0., 0.05, 70])
    skeleton_mesh = create_skeleton_mesh(hand_joints)

    final = renderer.render(skeleton_mesh.vertices, skeleton_mesh.faces, camera_t=camera_t)

    return final