import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from functools import partial
from accelerate import Accelerator, InitProcessGroupKwargs
from data_load.mesh_dataset_more_aug import  MeshDataset_infer
from utils.data_process import process_predictions
from utils.utils import count_model_params
from model_nsa_compile.transformer_nsa import NSAFaceBoundary
from model_compile.transformer import FaceBoundary
from config_loader.load_config import load_config
from datetime import timedelta
import random
import logging
torch._dynamo.config.suppress_errors = True
torch._logging.set_logs(dynamo=logging.ERROR)
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def collate_fn(batch):
    padded_batch = {}
    padded_batch["name"] = [x['name'] for x in batch]
    pc = [x['pc'] for x in batch]
    padded_batch["pc"] = torch.stack(pc)
    return padded_batch

def main(config):
    accelerator = Accelerator(
        mixed_precision=config.accelerator.mixed_precision,
    )
    device = accelerator.device

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
    else:
        raise ValueError(f"Unknown version: {config.model.model_version}")
    if accelerator.is_main_process:
        if accelerator.is_main_process:
            count_model_params(model)
    
    val_mesh_datasets = MeshDataset_infer.load(
        opt=config, 
        path=config.data.eval_dataset_path,
        accelerator=accelerator,
        token_map=token_map,
        version=config.model.model_version,
    )

    val_mesh_loader = DataLoader(
        val_mesh_datasets,
        batch_size=config.generate.batch_size,
        shuffle=False,
        num_workers=0,
        persistent_workers=False,
        collate_fn = partial(collate_fn)
    )

    output_folder = os.path.join(config.output_folder_base, config.project_name)
    print(f"Output will be saved to: {output_folder}")
    ckpt_path = config.model.model_path
    print(f"Loading weights from {ckpt_path}...")
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)

    model, val_mesh_loader = accelerator.prepare(model, val_mesh_loader)

    model_for_generate = model.module if hasattr(model, 'module') else model
    generate_fn = model_for_generate.generate
    use_compile = getattr(config.model, "use_compile", True)
    
    if use_compile:
        default_compile_mode = "max-autotune-no-cudagraphs"
        compile_mode = getattr(config.model, "compile_mode", default_compile_mode)
        model_for_generate.compile_math_kernels(compile_mode=compile_mode)
        generate_fn = model_for_generate.generate
        
    with torch.no_grad():
        for eval_batch_idx, data in enumerate(val_mesh_loader):
            name = data["name"]
            pc = data["pc"].to(device)
            batch_size = len(name)
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
            if config.model.model_version == "full_attn":
                call_kwargs = dict(
                    initial_input=initial_input,
                    init_context=init_context,
                    init_attention_mask=init_attention_mask,
                    init_cur_root_index=init_cur_root_index,
                    accelerator=accelerator,
                    token_map=token_map,
                    max_seq_len=config.data_processing.max_len * 9,
                    device=device,
                    init_cur_root_index_total=init_cur_root_index_total,
                    pc=pc,
                    use_toppk=True,
                    top_k=config.generate.top_k,
                    top_p=config.generate.top_p,
                    temperature=config.generate.temperature,
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
                    max_seq_len=config.data_processing.max_len * 9,
                    device=device,
                    init_cur_root_index_total=init_cur_root_index_total,
                    pc=pc,
                    top_k=config.generate.top_k,
                    top_p=config.generate.top_p,
                    temperature=config.generate.temperature,
                    return_cur_root=True,
                    eos_aug=config.generate.eos_aug,
                    wr_fix=config.generate.wr_fix,
                    root_connect_constrain=True,
                )
            total_pred_token = generate_fn(**call_kwargs)
            pred_token = total_pred_token[:, 9:]
            params_str = f"k{config.generate.top_k}_p{config.generate.top_p}_t{config.generate.temperature}"
            output_folder_ply = f'{output_folder}/_val_generate_{params_str}/'
            if not os.path.exists(output_folder_ply):
                os.makedirs(output_folder_ply, exist_ok=True)

            suffix = ""
            pred_token_unflatten = pred_token.view(pred_token.shape[0], -1, 9)
            process_predictions(pred_token_unflatten, config.data_processing.n_discrete_size, 
                output_folder_ply, name, token_map,
                vertex_order=config.data_processing.vertex_order, 
                suffix=suffix, return_mesh=True, clean_mesh=True)
            accelerator.wait_for_everyone()

if __name__ == "__main__":
    set_seed(42)
    config = load_config()
    main(config)