import trimesh
import numpy as np
import os
import torch
from torch.utils.data import Dataset
import glob
from tqdm import tqdm
from ripple_tokenizer.tokenizer import discretize, undiscretize
import logging
import random
from pathlib import Path
import open3d
from scipy.spatial import cKDTree

def dataset_exists(base_path):
    """Check whether any sharded dataset files exist."""
    base_path = base_path.replace(".npz", "")  # Ensure it matches sharded file naming
    files = glob.glob(f"{base_path}_*.npz")  # Get all matching npz files
    return len(files) > 0  # Return True if at least one file exists

def normalize_vertices(vertices):
    bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
    vertices = vertices - (bounds[0] + bounds[1])[None, :] / 2
    vertices = vertices / (bounds[1] - bounds[0]).max()
    return vertices

    
def dec(mesh, target_face):
    open3d_mesh = open3d.geometry.TriangleMesh(
            vertices=open3d.utility.Vector3dVector(mesh.vertices),
            triangles=open3d.utility.Vector3iVector(mesh.faces))
    simple = open3d_mesh.simplify_quadric_decimation(target_face)
    return trimesh.Trimesh(vertices=simple.vertices, faces=simple.triangles)

def normalize(mesh):
    bounds = np.array([mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)])
    mesh.vertices = mesh.vertices - (bounds[0] + bounds[1])[None, :] / 2
    mesh.vertices = mesh.vertices / (bounds[1] - bounds[0]).max()
    mesh.vertices = mesh.vertices.clip(-0.5, 0.5)  # Clamp vertex coordinates to [-0.5, 0.5]

def merge_duplicate_vertices(mesh, tolerance=0):
    vertices = mesh.vertices
    faces = mesh.faces
    
    # Use KD-tree to find nearby vertices
    tree = cKDTree(vertices)
    
    # Find vertex pairs within the tolerance
    pairs = tree.query_pairs(tolerance, output_type='ndarray')
    
    if len(pairs) == 0:
        return mesh  # No vertices need to be merged
    
    # Create vertex mapping: map duplicate vertices to the first one
    vertex_map = {i: i for i in range(len(vertices))}
    
    for i, j in pairs:
        # Always map to the smaller index
        if i < j:
            vertex_map[j] = i
        else:
            vertex_map[i] = j
    
    # Apply the vertex mapping
    new_faces = []
    for face in faces:
        new_face = [vertex_map[v] for v in face]
        new_faces.append(new_face)
    
    # Create a new mesh
    new_mesh = trimesh.Trimesh(vertices=vertices, faces=new_faces)
    
    # Remove unreferenced vertices
    new_mesh.remove_unreferenced_vertices()
    
    return new_mesh

def discrete_and_clean(mesh, n_discrete):
    vertices = discretize(mesh.vertices, num_discrete=n_discrete)
    dis_mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces)
    new_mesh = merge_duplicate_vertices(dis_mesh)
    new_mesh.merge_vertices()
    new_mesh.update_faces(new_mesh.nondegenerate_faces())
    new_mesh.update_faces(new_mesh.unique_faces())
    new_mesh.remove_unreferenced_vertices()
    new_mesh.fix_normals()
    return new_mesh

def sample_pc(mesh, pc_num, total_pc_num = 50000, with_normal=True, aug=True, overfit=False):
    if overfit:
        np.random.seed(42)
    if not with_normal:
        points, _ = mesh.sample(pc_num, return_index=True)
        return points

    points, face_idx = mesh.sample(total_pc_num, return_index=True)
    if aug and random.random() < 0.5:
        points += np.random.randn(*points.shape) * 0.01
    normals = mesh.face_normals[face_idx]
    pc_normal = np.concatenate([points, normals], axis=-1, dtype=np.float16)

    # random sample point cloud
    ind = np.random.choice(pc_normal.shape[0], pc_num, replace=False)
    pc_normal = pc_normal[ind]
    
    return pc_normal

class MeshDataset_infer(Dataset):
    """
    Refactored version where initialization parameters are passed via the opt object.
    """
    def __init__(self, opt, data, token_map, version):
        # Store the configuration object and core data in the instance
        self.opt = opt
        self.data = data
        self.token_map = token_map
        self.version = version
        self.y_up = self.opt.data_processing.y_up
        self.n_discrete_size = self.opt.data_processing.n_discrete_size
        self.vertex_order = self.opt.data_processing.vertex_order
        self.max_length = self.opt.data_processing.max_len 
        self.min_length = self.opt.data_processing.min_len 
        self.pc_num = self.opt.data_processing.pc_num
        self.total_pc_num = self.opt.data_processing.total_pc_num
        self.dec_to_facenum = self.opt.data_processing.dec_to_facenum
        self.overfit = self.opt.data_processing.overfit if hasattr(self.opt.data_processing, 'overfit') else False
        
        logging.info(f"[MeshDataset] Created from {len(self.data)} entries")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        idx = item
        file_path = self.data[idx]
        
        try:
            mesh = trimesh.load(file_path, force="mesh")
            if isinstance(mesh, trimesh.Scene):
                if len(mesh.geometry) > 0:
                    mesh = trimesh.util.concatenate(
                        tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces) for g in mesh.geometry.values())
                    )
            
            raw_vertices = mesh.vertices
            raw_faces = mesh.faces
            
            # Convert from Y-up to Z-up
            if self.y_up:
                mesh.vertices = mesh.vertices[:, [2, 0, 1]]
            
            # Normalize
            normalize(mesh)

            # Discretize
            mesh = discrete_and_clean(mesh, n_discrete=self.n_discrete_size)
            if self.dec_to_facenum != -1 and mesh.faces.shape[0]>self.dec_to_facenum:
                mesh = dec(mesh=mesh, target_face=self.dec_to_facenum)
            
            if self.version == "v1-nsa":
                undiscrete_clean_mesh = trimesh.Trimesh(
                    vertices=undiscretize(mesh.vertices, num_discrete=self.n_discrete_size), 
                    faces=mesh.faces
                    )
            elif self.version == "full_attn":
                undiscrete_clean_mesh = trimesh.Trimesh(
                    vertices=undiscretize(mesh.vertices), 
                    faces=mesh.faces
                    )
            else:
                raise ValueError(f"Unknown version: {self.version}")
                
            # Sample point cloud
            pc = sample_pc(undiscrete_clean_mesh, pc_num=self.pc_num, total_pc_num=self.total_pc_num, 
                            with_normal=True, aug=False, overfit=self.overfit)
        except Exception as e:
            logging.warning(f"Error loading {file_path}: {e}", exc_info=True)
            pc = np.zeros((self.pc_num, 6), dtype=np.float32)
            raw_vertices, raw_faces = [], []

        data_dict = {}
        data_dict["pc"] = torch.from_numpy(pc)
        data_dict['name'] = Path(file_path).stem
        return data_dict
    
    @classmethod
    def load(cls, opt, path, token_map, accelerator, version):
        """ 
        Load data and initialize a Dataset instance using the opt object.
        """
        data = []
        data.extend(glob.glob(os.path.join(path, '**', '*.glb'), recursive=True))
        data.extend(glob.glob(os.path.join(path, '**', '*.obj'), recursive=True))

        if accelerator.is_main_process:
            logging.info(f"[MeshDataset] Loaded {len(data)} entries from {path}")
        return cls(opt,
                   data=data,
                   token_map=token_map,
                   version=version)
