
import torch
from ripple_tokenizer.tokenizer import undiscretize_tensor
import trimesh
import os
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import math

def process_gt(raw_vertrices, raw_faces, name, output_folder_ply):
    """
    This function processes the predictions from the model, detokenizes the sequences, and saves the results as .obj files.
    Args:
        pred_discrete (torch.Tensor): The predicted token sequences of shape (batch_size, seq_len, vocab_size).
        pred_labels (torch.Tensor): The predicted edges of shape (batch_size, seq_len, seq_len).
        n_discrete_size (int): The size of the discrete vocabulary.
        output_folder_ply (str): Path where the .obj files will be saved.
        epoch (int): The current epoch number.
        eval_batch_idx (int): The batch index for evaluation.
        batch_size (int): The number of samples in a batch.
    """
    batch_size = len(raw_vertrices)
    mesh_list = []
    for each_idx in range(batch_size):
        vertices = raw_vertrices[each_idx]
        faces = raw_faces[each_idx]
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh_list.append(mesh)
        mesh.export(os.path.join(output_folder_ply,f'{name[each_idx]}_raw_gt.obj'))

def process_predictions(pred_discrete, n_discrete_size, output_folder_ply, 
                        name, token_map, vertex_order="zyx", suffix="", return_mesh=False, clean_mesh=True):
    """
    This function processes the predictions from the model, detokenizes the sequences, and saves the results as .obj files.
    Args:
        pred_discrete (torch.Tensor): The predicted token sequences of shape (batch_size, seq_len, vocab_size).
        pred_labels (torch.Tensor): The predicted edges of shape (batch_size, seq_len, seq_len).
        n_discrete_size (int): The size of the discrete vocabulary.
        output_folder_ply (str): Path where the .obj files will be saved.
        epoch (int): The current epoch number.
        eval_batch_idx (int): The batch index for evaluation.
        batch_size (int): The number of samples in a batch.
    """
    
    visual_batch_seq = []
    visual_edge_batch = []
    
    # Token IDs for EOS and PAD
    eos_id = token_map["eos"]
    pad_id = token_map["pad"]
    
    # Find eos_id or pad_id positions
    eos_mask = torch.eq(pred_discrete, eos_id.to('cpu')).any(dim=2)
    pad_mask = torch.eq(pred_discrete, pad_id.to('cpu')).any(dim=2)
    stop_mask = eos_mask | pad_mask
    
    stop_indices = stop_mask.float().argmax(dim=1)
    
    # Handle cases with no eos_id or pad_id
    no_stop_token = ~stop_mask.any(dim=1)
    stop_indices[no_stop_token] = pred_discrete.size(1)
    
    # Process each sample in the batch
    for b in range(pred_discrete.shape[0]):
        lenth = stop_indices[b].item()
        visual_seq = pred_discrete[b, :lenth]
        visual_batch_seq.append(visual_seq)

    mesh_list = []
    for each_idx, tokens in enumerate(visual_batch_seq):
        identifiers = torch.stack(list(token_map.values()))

        valid_mask = ~(tokens.unsqueeze(1) == identifiers).all(dim=2).any(dim=1)
        valid_indices = torch.where(valid_mask)[0]
        vertices_faces = tokens[valid_indices]
        if vertex_order=="zyx":
            vertices_faces_unflatten = vertices_faces.reshape(vertices_faces.shape[0], 3, 3)[:, :, [2,1,0]] # n,3,3
            vertices_faces = vertices_faces_unflatten.reshape(vertices_faces.shape[0], 9) # n,9
        os.makedirs(output_folder_ply, exist_ok=True)
        save_mesh_as_obj(vertices_faces, os.path.join(output_folder_ply,f'{name[each_idx]}{suffix}.obj'))

        faces = vertices_faces.reshape([vertices_faces.shape[0],3,3])
        all_vertices = faces.reshape([faces.shape[0]*3, 3])
        all_vertices = undiscretize_tensor(all_vertices, num_discrete=n_discrete_size)
        all_vertices = all_vertices[:, [1,2,0]]
        unique_vertices, inverse_indices = torch.unique(all_vertices, sorted =False, dim=0, return_inverse=True)
        faces_indices = inverse_indices.view(-1, 3)

        mesh = trimesh.Trimesh(vertices=unique_vertices, faces=faces_indices, process=False)
        mesh_list.append(mesh)
        mesh.export(os.path.join(output_folder_ply,f'{name[each_idx]}{suffix}_y.obj'))
    if return_mesh:
        return mesh_list

def save_mesh_as_obj(vertices_faces, file_path):
    """
    Save mesh data with shape (n, 9) to a .obj file.
    Each row represents a face with 3 vertices (3 coordinates per vertex).

    :param vertices_faces: The mesh data with shape (n, 9) where each row contains 9 values (3 vertices, each with 3 coordinates)
    :param file_path: The path where the .obj file will be saved
    """
    with open(file_path, 'w') as f:
        # Write vertices (each vertex consists of 3 values)
        vertex_count = 0  # To keep track of the vertex index
        # vertices_faces=vertices_faces.reshape(-1,9)
        
        for row in vertices_faces:
            for i in range(0, 9, 3):
                # Extract each vertex's x, y, z coordinates
                x, y, z = row[i], row[i+1], row[i+2]
                f.write(f"v {x} {y} {z}\n")
                vertex_count += 1
        
        # Write faces (each face is made up of 3 vertices, 1-based index)
        for i in range(vertex_count // 3):
            # Each face is made by three consecutive vertices
            v1 = 3 * i + 1
            v2 = 3 * i + 2
            v3 = 3 * i + 3
            f.write(f"f {v1} {v2} {v3}\n")