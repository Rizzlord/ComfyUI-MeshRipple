import os
import sys
import torch
import yaml
import numpy as np
import trimesh
from PIL import Image
# Add the MeshRipple directory to sys.path
EXTENSION_DIR = os.path.dirname(os.path.realpath(__file__))
MESH_RIPPLE_PATH = os.path.join(EXTENSION_DIR, "MeshRipple")

try:
    from folder_paths import get_filename_list, get_full_path, models_dir
except ImportError:
    # Fallback for testing or if running in an environment where folder_paths is not available
    def get_filename_list(category):
        return []
    def get_full_path(category, name):
        return None
    models_dir = os.path.join(EXTENSION_DIR, "models")

from huggingface_hub import hf_hub_download

if MESH_RIPPLE_PATH not in sys.path:
    sys.path.append(MESH_RIPPLE_PATH)
MICHE_PATH = os.path.join(MESH_RIPPLE_PATH, "miche")
if MICHE_PATH not in sys.path:
    sys.path.append(MICHE_PATH)

from model_compile.transformer import FaceBoundary
from model_nsa_compile.transformer_nsa import NSAFaceBoundary
from ripple_tokenizer.tokenizer import undiscretize_tensor
from ripple_utils.data_process import process_predictions

# Michelangelo Imports
from michelangelo.utils.misc import get_config_from_file, instantiate_from_config
from michelangelo.models.tsal.inference_utils import extract_geometry

class MockAccelerator:
    def __init__(self):
        self.is_local_main_process = True
        self.is_main_process = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def wait_for_everyone(self):
        pass

    def prepare(self, *args):
        if len(args) == 1:
            return args[0]
        return args

class DictToObject:
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, DictToObject(value))
            else:
                setattr(self, key, value)

def download_miche_model():
    miche_dir = os.path.join(models_dir, "miche")
    os.makedirs(miche_dir, exist_ok=True)
    
    ckpt_path = os.path.join(miche_dir, "shapevae-256.ckpt")
    config_path = os.path.join(miche_dir, "shapevae-256.yaml")
    
    repo_id = "Maikou/Michelangelo"
    
    if not os.path.exists(config_path):
        print(f"Downloading Miche config to {config_path}...")
        hf_hub_download(repo_id=repo_id, filename="configs/aligned_shape_latents/shapevae-256.yaml", local_dir=miche_dir, local_dir_use_symlinks=False)
        # Move it to the flat miche_dir if it was downloaded into a subfolder
        downloaded_path = os.path.join(miche_dir, "configs/aligned_shape_latents/shapevae-256.yaml")
        if os.path.exists(downloaded_path):
            os.rename(downloaded_path, config_path)
        
    if not os.path.exists(ckpt_path):
        print(f"Downloading Miche checkpoint to {ckpt_path}...")
        hf_hub_download(repo_id=repo_id, filename="checkpoints/aligned_shape_latents/shapevae-256.ckpt", local_dir=miche_dir, local_dir_use_symlinks=False)
        # Move it to the flat miche_dir if it was downloaded into a subfolder
        downloaded_path = os.path.join(miche_dir, "checkpoints/aligned_shape_latents/shapevae-256.ckpt")
        if os.path.exists(downloaded_path):
            os.rename(downloaded_path, ckpt_path)
    
    return ckpt_path, config_path

class MichelangeloModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        miche_dir = os.path.join(models_dir, "miche")
        os.makedirs(miche_dir, exist_ok=True)
        files = [f for f in os.listdir(miche_dir) if f.endswith(".ckpt")]
        return {
            "required": {
                "model_name": (files,),
            }
        }
    
    RETURN_TYPES = ("MICHE_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "MeshRipple/Michelangelo"

    def load_model(self, model_name):
        ckpt_path = os.path.join(models_dir, "miche", model_name)
        config_path = ckpt_path.replace(".ckpt", ".yaml")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found for model: {model_name}")
            
        model_config = get_config_from_file(config_path)
        if hasattr(model_config, "model"):
            model_config = model_config.model

        print(f"Loading Michelangelo model from {ckpt_path}...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = instantiate_from_config(model_config, ckpt_path=ckpt_path)
        model.to(device).eval()
        
        return ({"model": model, "config": model_config, "device": device},)

class MichelangeloImageToPoints:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "miche_model": ("MICHE_MODEL",),
                "image": ("IMAGE",),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "num_steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "sample_points": ("INT", {"default": 16384, "min": 1024, "max": 65536}),
            }
        }
    
    RETURN_TYPES = ("POINTS",)
    FUNCTION = "generate_points"
    CATEGORY = "MeshRipple/Michelangelo"

    def generate_points(self, miche_model, image, guidance_scale, num_steps, seed, sample_points):
        model = miche_model["model"]
        device = miche_model["device"]
        
        torch.manual_seed(seed)
        
        # Preprocess image
        # image is (B, H, W, C), range [0, 1]
        img = image[0].cpu().numpy()
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img).convert("RGB")
        img = img.resize((224, 224), Image.LANCZOS)
        
        img_np = np.array(img).astype(np.float32) / 255.0
        img_np = img_np * 2.0 - 1.0  # Normalize to [-1, 1]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
        
        sample_inputs = {"image": img_tensor}
        
        print(f"Sampling 3D shape from image with guidance_scale={guidance_scale}...")
        # Check if the model is the diffuser or just the VAE
        if hasattr(model, "sample"):
            # It's the diffuser
            mesh_outputs = model.sample(
                sample_inputs,
                sample_times=1,
                steps=num_steps,
                guidance_scale=guidance_scale,
                bounds=[-1.1, -1.1, -1.1, 1.1, 1.1, 1.1],
                octree_depth=7
            )[0]
        else:
            # It's likely just the VAE, try to encode/decode (though this node is for image generation)
            raise ValueError("The loaded Michelangelo model does not support sampling (it is likely a VAE, not a Diffuser). Please load an 'image-ASLDM' model.")

        # Extract mesh from the first output
        mesh_out = mesh_outputs[0]
        if mesh_out is None:
            raise ValueError("Michelangelo failed to generate a surface for this image.")
            
        # Michelangelo mesh faces are often inverted for some reason in their inference.py
        faces = mesh_out.mesh_f[:, ::-1]
        tm_mesh = trimesh.Trimesh(mesh_out.mesh_v, faces, process=False)
        
        # Sample points and normals
        sampled_points, face_idx = tm_mesh.sample(sample_points, return_index=True)
        normals = tm_mesh.face_normals[face_idx]
        
        points_6d = np.concatenate([sampled_points, normals], axis=-1).astype(np.float32)
        points_tensor = torch.from_numpy(points_6d).unsqueeze(0) # (1, N, 6)
        
        return (points_tensor,)

class MeshRippleModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (["meshRipple_10k.pth", "meshRipple_nsa.pth"],),
            }
        }
    
    RETURN_TYPES = ("MESH_RIPPLE_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "MeshRipple"

    def load_model(self, model_name):
        ckpt_path = os.path.join("/Apps/ComfyUI/models/meshripple/", model_name)
        
        if "10k" in model_name:
            config_file = os.path.join(MESH_RIPPLE_PATH, "config_loader/config_10k_full_dense_mesh.yaml")
        else:
            config_file = os.path.join(MESH_RIPPLE_PATH, "config_loader/config_20k_nsa.yaml")
            
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = DictToObject(config_dict)
        
        # Override model path in config to use the absolute path
        config.model.model_path = ckpt_path
        
        pad_id = config.data_processing.n_discrete_size + 3
        token_map = {
            's': torch.tensor([config.data_processing.n_discrete_size] * 9),
            'n': torch.tensor([config.data_processing.n_discrete_size + 1] * 9),
            'eos': torch.tensor([config.data_processing.n_discrete_size + 2] * 9),
            "pad": torch.tensor([pad_id] * 9),
            "b1": torch.tensor([pad_id + 1] * 9),
        }
        num_categories = config.data_processing.n_discrete_size + len(token_map)
        model_args = (
            config.data_processing.windowing.split_method,
            config.data_processing.windowing.window_size,
            config.data_processing.windowing.window_stride
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Ensure Miche model is available (fallback)
        download_miche_model()
        
        if config.model.model_version == "full_attn":
            model = FaceBoundary(
                args=model_args,
                embed_dim=config.model.feature_dim,
                num_heads=config.model.num_heads,
                num_categories=num_categories,
                context_embedding_dim=config.model.context_embedding_dim,
                max_len=config.data_processing.max_len,
                num_classify=config.model.num_classify,
                root_pred_depth=config.model.root_pred_depth,
                hourglass_depth=config.model.hourglass_depth,
                conditioned_on_pc=config.model.conditioned_on_pc,
                encoder_freeze=config.model.encoder_freeze,
            )
        elif config.model.model_version == "v1-nsa":
            model = NSAFaceBoundary(
                args=model_args,
                embed_dim=config.model.feature_dim,
                num_heads=config.model.num_heads,
                root_pred_depth=config.model.root_pred_depth,
                num_categories=num_categories,
                context_embedding_dim=config.model.context_embedding_dim,
                max_len=config.data_processing.max_len,
                num_classify=config.model.num_classify,
                hourglass_depth=config.model.hourglass_depth,
                conditioned_on_pc=config.model.conditioned_on_pc,
                encoder_freeze=config.model.encoder_freeze,
            )
        
        print(f"Loading MeshRipple weights from {ckpt_path}...")
        state_dict = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        model.to(device).eval()
        
        return ({"model": model, "config": config, "token_map": token_map, "device": device},)

class MeshRippleGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh_ripple_model": ("MESH_RIPPLE_MODEL",),
                "points": ("POINTS",), # Expects (N, 3) or (N, 6) tensor
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "top_k": ("INT", {"default": 50, "min": 1, "max": 100}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.01, "max": 2.0}),
                "max_faces": ("INT", {"default": 5000, "min": 100, "max": 20000}),
            }
        }
    
    RETURN_TYPES = ("MESH",)
    FUNCTION = "generate_mesh"
    CATEGORY = "MeshRipple"

    def generate_mesh(self, mesh_ripple_model, points, seed, top_k, top_p, temperature, max_faces):
        # Prepare points
        # If points is (B, N, 3/6), take first batch
        if points.ndim == 3:
            points = points[0]
            
        # Ensure points are (N, 6)
        if points.shape[1] == 3:
            # Add zero normals if missing
            normals = torch.zeros_like(points)
            points = torch.cat([points, normals], dim=1)
        
        # Normalize points to [-0.5, 0.5]
        p_min = points[:, :3].min(dim=0)[0]
        p_max = points[:, :3].max(dim=0)[0]
        p_center = (p_min + p_max) / 2
        p_scale = (p_max - p_min).max()
        
        norm_points = points.clone()
        norm_points[:, :3] = (points[:, :3] - p_center) / p_scale
        norm_points[:, :3] = norm_points[:, :3].clamp(-0.5, 0.5)
        
        return common_generate(mesh_ripple_model, norm_points, seed, top_k, top_p, temperature, max_faces)

class MeshRippleRemesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh_ripple_model": ("MESH_RIPPLE_MODEL",),
                "mesh": ("*", {"forceInput": True}), # Use '*' to allow various 3D node outputs (MESH, TRIMESH, etc.)
                "sample_points": ("INT", {"default": 16384, "min": 1024, "max": 65536}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "top_k": ("INT", {"default": 50, "min": 1, "max": 100}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.01, "max": 2.0}),
                "max_faces": ("INT", {"default": 5000, "min": 100, "max": 20000}),
            }
        }
    
    RETURN_TYPES = ("MESH",)
    FUNCTION = "remesh"
    CATEGORY = "MeshRipple"

    def remesh(self, mesh_ripple_model, mesh, sample_points, seed, top_k, top_p, temperature, max_faces):
        # Convert input to trimesh
        # Handle dictionary format (ComfyUI standard)
        if isinstance(mesh, dict) and "vertices" in mesh and "faces" in mesh:
            vertices = mesh["vertices"][0].cpu().numpy()
            faces = mesh["faces"][0].cpu().numpy()
            tm_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        # Handle raw trimesh object (common in some 3D extensions)
        elif hasattr(mesh, "vertices") and hasattr(mesh, "faces") and not isinstance(mesh, dict):
            # It's likely already a trimesh-like object
            # We normalize its scale/center later anyway
            vertices = np.array(mesh.vertices)
            faces = np.array(mesh.faces)
            tm_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        else:
            # Final fallback: try to see if it's some other object with vertices/faces
            try:
                vertices = np.array(mesh.vertices)
                faces = np.array(mesh.faces)
                tm_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            except:
                raise ValueError(f"Unsupported mesh format: {type(mesh)}. Expected ComfyUI dictionary or Trimesh object.")
        
        # Normalize mesh to [-0.5, 0.5] locally first for better sampling or just sample raw
        # The model logic expects normalized points, so we'll normalize our sampled points.
        
        # Sample points and normals
        # Using logic similar to MeshDataset_infer.sample_pc
        points, face_idx = tm_mesh.sample(sample_points, return_index=True)
        normals = tm_mesh.face_normals[face_idx]
        
        points_norm = np.concatenate([points, normals], axis=-1).astype(np.float32)
        norm_points_tensor = torch.from_numpy(points_norm)
        
        # Shift and scale to [-0.5, 0.5]
        p_min = norm_points_tensor[:, :3].min(dim=0)[0]
        p_max = norm_points_tensor[:, :3].max(dim=0)[0]
        p_center = (p_min + p_max) / 2
        p_scale = (p_max - p_min).max()
        
        norm_points_tensor[:, :3] = (norm_points_tensor[:, :3] - p_center) / p_scale
        norm_points_tensor[:, :3] = norm_points_tensor[:, :3].clamp(-0.5, 0.5)
        
        return common_generate(mesh_ripple_model, norm_points_tensor, seed, top_k, top_p, temperature, max_faces)

def common_generate(mesh_ripple_model, norm_points, seed, top_k, top_p, temperature, max_faces):
    model = mesh_ripple_model["model"]
    config = mesh_ripple_model["config"]
    token_map = mesh_ripple_model["token_map"]
    device = mesh_ripple_model["device"]
    
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    
    # Sample or pad to config.data_processing.pc_num (default 16384)
    target_pc_num = config.data_processing.pc_num
    current_pc_num = norm_points.shape[0]
    
    if current_pc_num > target_pc_num:
        indices = torch.randperm(current_pc_num)[:target_pc_num]
        norm_points = norm_points[indices]
    elif current_pc_num < target_pc_num:
        padding = torch.zeros((target_pc_num - current_pc_num, 6), device=norm_points.device)
        norm_points = torch.cat([norm_points, padding], dim=0)
        
    pc = norm_points.unsqueeze(0).to(device)
    batch_size = 1
    
    s_token = token_map['s'].to(device=device, dtype=torch.long)
    n_token = token_map['n'].to(device=device, dtype=torch.long)

    init_context = torch.stack([s_token, n_token], dim=0).unsqueeze(0).repeat(batch_size, 1, 1)
    initial_input = init_context.view(batch_size, -1)

    base_attention_mask = torch.tensor([[False, True], [True, False]], dtype=torch.bool, device=device)
    init_attention_mask = base_attention_mask.unsqueeze(0).repeat(batch_size, 1, 1)

    init_cur_root_index = torch.ones(batch_size, dtype=torch.long, device=device)
    init_cur_root_move = torch.cat([
        torch.zeros(batch_size, 1, dtype=torch.long, device=device),
        torch.ones(batch_size, 1, dtype=torch.long, device=device)
    ], dim=-1)
    init_cur_root_index_total = torch.cumsum(init_cur_root_move, dim=-1)
    
    # Effective max len
    max_seq_len = min(max_faces, config.data_processing.max_len) * 9
    accelerator = MockAccelerator()
    
    with torch.no_grad():
        if config.model.model_version == "full_attn":
            call_kwargs = dict(
                initial_input=initial_input,
                init_context=init_context,
                init_attention_mask=init_attention_mask,
                init_cur_root_index=init_cur_root_index,
                accelerator=accelerator,
                token_map=token_map,
                max_seq_len=max_seq_len,
                device=device,
                init_cur_root_index_total=init_cur_root_index_total,
                pc=pc,
                use_toppk=True,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                eos_aug=config.generate.eos_aug,
                wr_fix=config.generate.wr_fix,
                root_connect_constrain=True,
            )
        else:
            call_kwargs = dict(
                initial_input=initial_input,
                init_context=init_context,
                init_attention_mask=init_attention_mask,
                init_cur_root_index=init_cur_root_index,
                accelerator=accelerator,
                token_map=token_map,
                max_seq_len=max_seq_len,
                device=device,
                init_cur_root_index_total=init_cur_root_index_total,
                pc=pc,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                return_cur_root=True,
                eos_aug=config.generate.eos_aug,
                wr_fix=config.generate.wr_fix,
                root_connect_constrain=True,
            )
        
        total_pred_token = model.generate(**call_kwargs)
        
    pred_token = total_pred_token[:, 9:]
    pred_token_unflatten = pred_token.view(pred_token.shape[0], -1, 9)
    
    # Token processing logic
    eos_id = token_map["eos"]
    pad_id = token_map["pad"]
    
    eos_mask = torch.eq(pred_token_unflatten.to(device), eos_id.to(device)).any(dim=2)
    pad_mask = torch.eq(pred_token_unflatten, pad_id.to(device)).any(dim=2)
    stop_mask = eos_mask | pad_mask
    
    stop_indices = stop_mask.float().argmax(dim=1)
    no_stop_token = ~stop_mask.any(dim=1)
    stop_indices[no_stop_token] = pred_token_unflatten.size(1)
    
    lenth = stop_indices[0].item()
    tokens = pred_token_unflatten[0, :lenth]
    
    identifiers = torch.stack(list(token_map.values())).to(device)
    valid_mask = ~(tokens.unsqueeze(1) == identifiers).all(dim=2).any(dim=1)
    valid_indices = torch.where(valid_mask)[0]
    vertices_faces = tokens[valid_indices]
    
    if config.data_processing.vertex_order == "zyx":
        vertices_faces_unflatten = vertices_faces.reshape(vertices_faces.shape[0], 3, 3)[:, :, [2,1,0]]
        vertices_faces = vertices_faces_unflatten.reshape(vertices_faces.shape[0], 9)
        
    faces_coords = vertices_faces.reshape([vertices_faces.shape[0], 3, 3])
    all_vertices = faces_coords.reshape([faces_coords.shape[0] * 3, 3])
    
    # Undiscretize
    all_vertices = undiscretize_tensor(all_vertices, num_discrete=config.data_processing.n_discrete_size)
    all_vertices = all_vertices[:, [1, 2, 0]]
    
    unique_vertices, inverse_indices = torch.unique(all_vertices, sorted=False, dim=0, return_inverse=True)
    faces_indices = inverse_indices.view(-1, 3)
    
    return ({"vertices": unique_vertices.unsqueeze(0), "faces": faces_indices.unsqueeze(0)},)

NODE_CLASS_MAPPINGS = {
    "MeshRippleModelLoader": MeshRippleModelLoader,
    "MeshRippleGenerator": MeshRippleGenerator,
    "MeshRippleRemesh": MeshRippleRemesh,
    "MichelangeloModelLoader": MichelangeloModelLoader,
    "MichelangeloImageToPoints": MichelangeloImageToPoints,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MeshRippleModelLoader": "MeshRipple Model Loader",
    "MeshRippleGenerator": "MeshRipple Generator",
    "MeshRippleRemesh": "MeshRipple Remesh",
    "MichelangeloModelLoader": "Michelangelo Model Loader",
    "MichelangeloImageToPoints": "Michelangelo Image to Points",
}
