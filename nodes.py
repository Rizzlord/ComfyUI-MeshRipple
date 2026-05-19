import os
import sys
import torch
import yaml
import numpy as np
import trimesh
from PIL import Image
import comfy.utils

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



if MESH_RIPPLE_PATH not in sys.path:
    sys.path.append(MESH_RIPPLE_PATH)
MICHE_PATH = os.path.join(MESH_RIPPLE_PATH, "miche")
if MICHE_PATH not in sys.path:
    sys.path.append(MICHE_PATH)

from model_compile.transformer import FaceBoundary
from model_nsa_compile.transformer_nsa import NSAFaceBoundary
from ripple_tokenizer.tokenizer import undiscretize_tensor
from ripple_utils.data_process import process_predictions

def make_progress_callback(unique_id, config):
    from server import PromptServer
    if PromptServer.instance is None:
        return None
    node_str = str(unique_id[0]) if isinstance(unique_id, list) else str(unique_id)
    def callback(generated, token_map):
        try:
            pred_token = generated[:, 9:]
            pred_token_unflatten = pred_token.view(pred_token.shape[0], -1, 9)
            eos_id = token_map["eos"].cpu()
            pad_id = token_map["pad"].cpu()
            pred_token_unflatten_cpu = pred_token_unflatten.detach().cpu()
            eos_mask = torch.eq(pred_token_unflatten_cpu, eos_id).any(dim=2)
            pad_mask = torch.eq(pred_token_unflatten_cpu, pad_id).any(dim=2)
            stop_mask = eos_mask | pad_mask
            stop_indices = stop_mask.float().argmax(dim=1)
            no_stop_token = ~stop_mask.any(dim=1)
            stop_indices[no_stop_token] = pred_token_unflatten.size(1)
            lenth = stop_indices[0].item()
            if lenth == 0:
                return
            tokens = pred_token_unflatten[0, :lenth]
            identifiers = torch.stack([v.detach().cpu() for v in token_map.values()])
            valid_mask = ~(tokens.unsqueeze(1).cpu() == identifiers).all(dim=2).any(dim=1)
            valid_indices = torch.where(valid_mask)[0]
            if len(valid_indices) == 0:
                return
            vertices_faces = tokens[valid_indices]
            if config.data_processing.vertex_order == "zyx":
                vertices_faces_unflatten = vertices_faces.reshape(vertices_faces.shape[0], 3, 3)[:, :, [2,1,0]]
                vertices_faces = vertices_faces_unflatten.reshape(vertices_faces.shape[0], 9)
            faces_coords = vertices_faces.reshape([vertices_faces.shape[0], 3, 3])
            all_vertices = faces_coords.reshape([faces_coords.shape[0] * 3, 3])
            all_vertices = undiscretize_tensor(all_vertices, num_discrete=config.data_processing.n_discrete_size)
            all_vertices = all_vertices[:, [1, 2, 0]]
            unique_vertices, inverse_indices = torch.unique(all_vertices, sorted=False, dim=0, return_inverse=True)
            faces_indices = inverse_indices.view(-1, 3).cpu().numpy()
            PromptServer.instance.send_sync("meshripple_preview", {
                "node_id": node_str,
                "vertices": unique_vertices.cpu().numpy().tolist(),
                "faces": faces_indices.tolist()
            })
        except Exception:
            pass
    return callback


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



class TrimeshToPoints:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            }
        }

    RETURN_TYPES = ("POINTS",)
    RETURN_NAMES = ("points",)
    FUNCTION = "sample"
    CATEGORY = "MeshRipple"

    def sample(self, trimesh):
        import trimesh as tm_module
        import torch

        N = 16384
        print(f"Sampling {N} points from Trimesh for MeshRipple...")
        
        # Consistent sampling logic using tm_module to avoid shadowing
        points, face_indices = tm_module.sample.sample_surface(trimesh, N)
        sampled_normals = trimesh.face_normals[face_indices]
        
        points_tensor = torch.from_numpy(points).float()
        normals_tensor = torch.from_numpy(sampled_normals).float()

        # Combine to (N, 6)
        pc_normal = torch.cat([points_tensor, normals_tensor], dim=1) # (N, 6)
        
        return (pc_normal.unsqueeze(0),) # (1, N, 6)


class MeshRippleModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (["meshRipple_10k.pth", "meshRipple_nsa.pth"],),
                "compile_model": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("MESH_RIPPLE_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "MeshRipple"

    def load_model(self, model_name, compile_model):
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
        
        if compile_model:
            print(f"Compiling MeshRipple math kernels...")
            model.compile_math_kernels()
        
        return ({"model": model, "config": config, "token_map": token_map, "device": device},)

class MeshRippleGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh_ripple_model": ("MESH_RIPPLE_MODEL",),
                "points": ("POINTS", {"tooltip": "Input point cloud with normals."}),
                "sample_points": ("INT", {"default": 16384, "min": 1024, "max": 65536, "tooltip": "Target number of points for the generator. Default 16384 is recommended by the paper."}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Random seed for generation."}),
                "top_k": ("INT", {"default": 50, "min": 1, "max": 100, "tooltip": "Limits sampling to the top K most likely tokens. Higher = more diverse but riskier."}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "tooltip": "Nucleus sampling threshold. Filters out lower probability noise."}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.01, "max": 2.0, "tooltip": "Controls randomness. <1.0 is safer/tighter, >1.0 is more experimental."}),
                "max_faces": ("INT", {"default": 5000, "min": 100, "max": 60000, "tooltip": "Maximum number of faces to generate before auto-stopping."}),
                "use_kv_cache": ("BOOLEAN", {"default": True, "tooltip": "Enables Key-Value caching for massive speedup. This is mathematically lossless."}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }
    
    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "generate_mesh"
    CATEGORY = "MeshRipple"

    def generate_mesh(self, mesh_ripple_model, points, sample_points, seed, top_k, top_p, temperature, max_faces, use_kv_cache, unique_id=None):
        # Prepare points
        # If points is (B, N, 3/6), take first batch
        if points.ndim == 3:
            points = points[0]
            
        # Ensure points are (N, 6)
        if points.shape[1] == 3:
            # Add zero normals if missing
            normals = torch.zeros_like(points)
            points = torch.cat([points, normals], dim=1)
        
        # Ensure exact number of points required by model
        num_in = points.shape[0]
        if num_in != sample_points:
            print(f"Resampling point cloud from {num_in} to {sample_points}...")
            if num_in > sample_points:
                # Randomly sample
                torch.manual_seed(seed)
                indices = torch.randperm(num_in)[:sample_points]
                points = points[indices]
            else:
                # Padding or repetition
                repeats = (sample_points // num_in) + 1
                points = points.repeat(repeats, 1)[:sample_points]

        # Normalize points to [-0.5, 0.5]
        p_min = points[:, :3].min(dim=0)[0]
        p_max = points[:, :3].max(dim=0)[0]
        p_center = (p_min + p_max) / 2
        p_scale = (p_max - p_min).max()
        
        norm_points = points.clone()
        norm_points[:, :3] = (points[:, :3] - p_center) / p_scale
        norm_points[:, :3] = norm_points[:, :3].clamp(-0.5, 0.5)
        
        return common_generate(mesh_ripple_model, norm_points, seed, top_k, top_p, temperature, max_faces, use_kv_cache, unique_id=unique_id)


def common_generate(mesh_ripple_model, norm_points, seed, top_k, top_p, temperature, max_faces, use_kv_cache, unique_id=None):
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
        
    pc_np = norm_points.cpu().numpy()
    
    # Research code Y-up to Z-up conversion logic (if input is Y-up)
    if config.data_processing.y_up:
        # Permute (x, y, z) -> (z, x, y)
        pc_np_new = pc_np.copy()
        pc_np_new[:, 0] = pc_np[:, 2] # x_new = z_old
        pc_np_new[:, 1] = pc_np[:, 0] # y_new = x_old
        pc_np_new[:, 2] = pc_np[:, 1] # z_new = y_old (up)
        
        pc_np_new[:, 3] = pc_np[:, 5] # nx_new = nz_old
        pc_np_new[:, 4] = pc_np[:, 3] # ny_new = nx_old
        pc_np_new[:, 5] = pc_np[:, 4] # nz_new = ny_old
        pc_np = pc_np_new
        
    pc = torch.from_numpy(pc_np).unsqueeze(0).to(device)
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
    
    # Initialize ComfyUI progress bar
    pbar = comfy.utils.ProgressBar(max_faces)
    
    # Effective max len dynamically grows to max_faces without being clamped
    max_seq_len = max_faces * 9
    accelerator = MockAccelerator()
    
    callback = make_progress_callback(unique_id, config) if unique_id is not None else None

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
                use_kv_cache=use_kv_cache,
                max_faces=max_faces,
                pbar=pbar,
                callback=callback
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
                use_kv_cache=use_kv_cache,
                max_faces=max_faces,
                pbar=pbar,
                callback=callback
            )
        
        total_pred_token = model.generate(**call_kwargs)
        
    pred_token = total_pred_token[:, 9:]
    pred_token_unflatten = pred_token.view(pred_token.shape[0], -1, 9)
    
    # Token processing logic - ensure all on CPU for processing
    eos_id = token_map["eos"].cpu()
    pad_id = token_map["pad"].cpu()
    
    # Process on CPU to avoid device mismatch with the IDs from token_map
    pred_token_unflatten_cpu = pred_token_unflatten.detach().cpu()
    eos_mask = torch.eq(pred_token_unflatten_cpu, eos_id).any(dim=2)
    pad_mask = torch.eq(pred_token_unflatten_cpu, pad_id).any(dim=2)
    stop_mask = eos_mask | pad_mask
    
    stop_indices = stop_mask.float().argmax(dim=1)
    no_stop_token = ~stop_mask.any(dim=1)
    stop_indices[no_stop_token] = pred_token_unflatten.size(1)
    
    lenth = stop_indices[0].item()
    tokens = pred_token_unflatten[0, :lenth]
    
    # Use CPU for identifier comparison
    identifiers = torch.stack([v.detach().cpu() for v in token_map.values()])
    valid_mask = ~(tokens.unsqueeze(1).cpu() == identifiers).all(dim=2).any(dim=1)
    valid_indices = torch.where(valid_mask)[0]
    vertices_faces = tokens[valid_indices]
    
    if config.data_processing.vertex_order == "zyx":
        vertices_faces_unflatten = vertices_faces.reshape(vertices_faces.shape[0], 3, 3)[:, :, [2,1,0]]
        vertices_faces = vertices_faces_unflatten.reshape(vertices_faces.shape[0], 9)
        
    faces_coords = vertices_faces.reshape([vertices_faces.shape[0], 3, 3])
    all_vertices = faces_coords.reshape([faces_coords.shape[0] * 3, 3])
    
    # Undiscretize
    all_vertices = undiscretize_tensor(all_vertices, num_discrete=config.data_processing.n_discrete_size)
    # Research code output permutation: [1, 2, 0]
    all_vertices = all_vertices[:, [1, 2, 0]]
    
    unique_vertices, inverse_indices = torch.unique(all_vertices, sorted=False, dim=0, return_inverse=True)
    faces_indices = inverse_indices.view(-1, 3).cpu().numpy()
    
    # Create trimesh object
    mesh = trimesh.Trimesh(vertices=unique_vertices.cpu().numpy(), faces=faces_indices)
        
    return (mesh,)


class MeshHoleSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "sample_points": ("INT", {"default": 16384, "min": 1, "max": 65536}),
            }
        }

    RETURN_TYPES = ("POINTS",)
    RETURN_NAMES = ("points",)
    FUNCTION = "sample"
    CATEGORY = "MeshRipple"

    def sample(self, trimesh, sample_points):
        import trimesh as tm
        import torch
        
        mesh_copy = trimesh.copy()
        original_faces_count = len(mesh_copy.faces)
        
        tm.repair.fill_holes(mesh_copy)
        
        new_faces_count = len(mesh_copy.faces)
        
        if new_faces_count <= original_faces_count:
            print("No holes detected in the mesh.")
            return (torch.zeros((1, 1, 6)),)
            
        patch_faces = mesh_copy.faces[original_faces_count:]
        patch_mesh = tm.Trimesh(vertices=mesh_copy.vertices, faces=patch_faces)
        
        points, face_indices = tm.sample.sample_surface(patch_mesh, sample_points)
        normals = patch_mesh.face_normals[face_indices]
        
        points_tensor = torch.from_numpy(points).float()
        normals_tensor = torch.from_numpy(normals).float()
        
        pc_normal = torch.cat([points_tensor, normals_tensor], dim=1)
        
        return (pc_normal.unsqueeze(0),)


class MeshMerge:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh_1": ("TRIMESH",),
                "mesh_2": ("TRIMESH",),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "merge"
    CATEGORY = "MeshRipple"

    def merge(self, mesh_1, mesh_2):
        import trimesh as tm
        merged_mesh = tm.util.concatenate([mesh_1, mesh_2])
        return (merged_mesh,)


NODE_CLASS_MAPPINGS = {
    "MeshRippleModelLoader": MeshRippleModelLoader,
    "MeshRippleGenerator": MeshRippleGenerator,
    "TrimeshToPoints": TrimeshToPoints,
    "MeshHoleSampler": MeshHoleSampler,
    "MeshMerge": MeshMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MeshRippleModelLoader": "MeshRipple Model Loader",
    "MeshRippleGenerator": "MeshRipple Generator",
    "TrimeshToPoints": "Trimesh to Points (16k)",
    "MeshHoleSampler": "Mesh Hole Sampler",
    "MeshMerge": "Mesh Merge",
}
