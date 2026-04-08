import trimesh
import numpy as np
from collections import deque
from tqdm import tqdm
import numpy as np
import sys
import os
import torch
from collections import defaultdict
import trimesh

# from visualize import save_mesh_animation_faces
class Vertex:
    def __init__(self, x, y, z, index):
        self.x = x
        self.y = y
        self.z = z
        self.index = index
        self.vis = False  # Whether it has been visited

class HalfEdge:
    def __init__(self, v, s, e):
        self.v = v  # Corresponding vertex
        self.s = s  # Start vertex
        self.e = e  # End vertex
        self.t = None  # Belonging triangle face
        self.n = None  # Next half-edge
        self.p = None  # Previous half-edge
        self.o = []  # Twin half-edges (opposite)

class TriangleFace:
    def __init__(self):
        self.halfedges = None
        self.face_idx = -1
        self.vis = False  # Whether it has been visited

def create_half_edges(input_vertices, input_faces):
    # Store half-edges corresponding to each vertex pair (edge)
    edge_dict = defaultdict(list)
    
    # Initialize all vertices
    vertices = [Vertex(v[0], v[1], v[2], idx) for idx, v in enumerate(input_vertices)]
    triangle_faces =  [TriangleFace() for f in input_faces]
    # Initialize half-edges for all triangle faces
    half_edges = []  # Store all created half-edges

    # Iterate through all triangle faces to create half-edges
    for face_idx, face in enumerate(input_faces):
        face_halfedges = []
        # Create three half-edges for the current face
        for i in range(3):
            v_start = vertices[face[i]]  # Start vertex of the current half-edge
            v_end = vertices[face[(i + 1) % 3]]  # End vertex of the current half-edge
            v_v = vertices[face[(i + 2) % 3]]  # Opposite vertex
            he = HalfEdge(v_v, v_start, v_end)  # Create half-edge
            face_halfedges.append(he)
            half_edges.append(he)
            edge_dict[(v_start.index, v_end.index)].append(he)  # Store half-edge

        # Set the previous and next relationships for the three half-edges of the current face
        face_halfedges[0].n, face_halfedges[0].p = face_halfedges[1], face_halfedges[2]
        face_halfedges[1].n, face_halfedges[1].p = face_halfedges[2], face_halfedges[0]
        face_halfedges[2].n, face_halfedges[2].p = face_halfedges[0], face_halfedges[1]

        triangle_faces[face_idx].halfedges = face_halfedges
        triangle_faces[face_idx].face_idx = face_idx
        for he in face_halfedges:
            he.t = triangle_faces[face_idx]
        

    for (v_start, v_next), he_list in edge_dict.items():
        if (v_next, v_start) in edge_dict:
            # Find the half-edge in the opposite direction of the current edge (twin half-edge)
            for he in he_list:
                for o_he in edge_dict[(v_next, v_start)]:
                    he.o.append(o_he)
                    o_he.o.append(he)

    return vertices, half_edges, triangle_faces

class FaceBfsTokenizer:
    def __init__(self, mesh, n_discrete_size, token_map, vertex_order="zyx"):
        """
        Initialize the graph tokenizer
        
        Args:
            end_id: Base ID for the end token
        """
        self.source_mesh = mesh
        self.n_discrete_size = n_discrete_size
        self.vertex_order = vertex_order
        self.tokenized_token = []
        self.tokenized_token_coord = []
        self.start_id = token_map["s"]
        self.new_id = token_map["n"]
        self.special_tokens = torch.stack([self.start_id, self.new_id])
        self.token_idx = 0
        self.mask_l = 0
        self.mask_r = 0
        self.process()
    
    def process(self, sort=True):
        if sort:
            sort_inds = np.lexsort(self.source_mesh.vertices.T)
            vertices = self.source_mesh.vertices[sort_inds]
            inv_sort_inds = np.argsort(sort_inds)

            faces = inv_sort_inds[self.source_mesh.faces]
            sort_inds = np.lexsort((faces[:, 2], faces[:, 1], faces[:, 0]))
            faces = faces[sort_inds]
        else:
            vertices = self.source_mesh.vertices
            faces = self.source_mesh.faces
        if self.vertex_order == "zyx":
            vertices = vertices[:, [2, 1, 0]]
        self.vertices = torch.from_numpy(vertices).to(torch.int64)
        self.faces = torch.from_numpy(faces).to(torch.int64)
        face_vertices = self.vertices[self.faces]  # (num_faces, 3, 3)
        self.face_vertices = face_vertices.reshape(-1, 9) # (num_faces, 3, 3) 
        self.vertex, self.half_edges, self.triangle_faces = create_half_edges(vertices, faces)

    def is_special_token(self, token):
        return torch.eq(token,self.special_tokens).all(dim=-1).any()

    def mask_l_move(self, step):
        while(self.mask_l < len(self.tokenized_token) and self.mask_l < self.mask_r and 
              self.is_special_token(self.tokenized_token[self.mask_l])):
            self.mask_l += 1
        for _ in range(step):
            if self.mask_l < len(self.tokenized_token) and self.mask_l < self.mask_r:
                self.mask_l += 1
            while(self.mask_l < len(self.tokenized_token) and self.mask_l < self.mask_r and 
                  self.is_special_token(self.tokenized_token[self.mask_l])):
                self.mask_l += 1

    def add_start_id(self):
        self.tokenized_token.append(self.start_id)
        self.cur_root[self.token_idx] = self.mask_l
        
        self.token_idx = self.token_idx + 1
        self.mask_r = self.mask_r + 1

    def add_new_id(self):
        self.tokenized_token.append(self.new_id)
        self.mask_l = self.token_idx
        self.cur_root[self.token_idx] = self.mask_l
        
        self.token_idx = self.token_idx + 1
        self.mask_r = self.mask_r + 1

    def add_end_id(self):
        self.token_idx = self.token_idx - 1
        self.mask_r = self.mask_r - 1
        
        self.mask_l_move(step=1)
        
        self.token_idx = self.token_idx + 1
        self.mask_r = self.mask_r + 1


    def add_face(self, he):
        new_face_cord_list = [self.vertices[he.s.index], 
                              self.vertices[he.e.index], self.vertices[he.v.index]]
        new_face_cord = torch.stack(new_face_cord_list).view(-1)
        self.tokenized_token.append(new_face_cord)
        
        self.token_idx = self.token_idx - 1
        self.mask_r = self.mask_r - 1
        self.mask_l_move(step=0)
        self.token_idx = self.token_idx + 1
        self.mask_r = self.mask_r + 1
        
        self.cur_root[self.token_idx] = self.mask_l
        
        self.token_idx = self.token_idx + 1
        self.mask_r = self.mask_r + 1
    
    def face_bfs_traverse(self):
        queue = deque()  # Edge queue
        self.tokenized_faces = []
        self.cur_root = torch.zeros((3*self.faces.shape[0]+10,), dtype=torch.int64)
        self.token_idx, self.mask_l, self.mask_r = 0, 0, 0
        self.add_start_id()
        # Iterate through all edges to ensure non-connected parts are processed
        for he_idx, he in enumerate(self.half_edges):
            if he.t.vis:
                continue
            he.t.vis = True  # Mark the current face as visited
            # Start a new BFS traversal
            queue.append(he)
            self.add_new_id()
            self.add_face(he)
            while queue:
                he = queue.popleft()

                if len(he.p.o) > 0:
                    for each_he_o in he.p.o:
                        if each_he_o.t.vis:
                            continue
                        each_he_o.t.vis = True
                        self.add_face(each_he_o)
                        queue.append(each_he_o)
                
                if len(he.n.o) > 0:
                    for each_he_o in he.n.o:
                        if each_he_o.t.vis:
                            continue
                        each_he_o.t.vis = True
                        self.add_face(each_he_o)
                        queue.append(each_he_o)
                        
                if len(he.o) > 0:
                    for each_he_o in he.o:
                        if each_he_o.t.vis:
                            continue
                        each_he_o.t.vis = True
                        self.add_face(each_he_o)
                        queue.append(each_he_o)

                self.add_end_id()
        total_len = len(self.tokenized_token)
        self.cur_root = self.cur_root[:total_len]
        self.tokenized_token = torch.stack(self.tokenized_token)
        self.next_root = torch.cat([self.cur_root[1:], torch.tensor([total_len])], dim=0)

def create_token_from_mesh_face_bfs_v2_clean(mesh, n_discrete_size, token_map, vertex_order):
    tokenizer = FaceBfsTokenizer(mesh, n_discrete_size, token_map, vertex_order)
    tokenizer.face_bfs_traverse()

    tokenized_face = tokenizer.tokenized_token
    special_tokens = tokenizer.special_tokens

    next_root = tokenizer.next_root

    valid_range = torch.tensor([0, len(tokenized_face)])
    valid_range = torch.tensor([0, len(tokenized_face)])
    
    # Add EOS token
    token_len = len(tokenized_face) + 1
    next_root = torch.cat([next_root, torch.tensor([token_len])], dim=0)

    tokenized_face = torch.cat([tokenized_face, token_map["eos"].unsqueeze(0)], dim=0)
    valid_range[1] += 1
    data_dict = {
        "token": tokenized_face,  # Store token data directly
        "valid_range": valid_range,  # Store valid_range
        "special_tokens": special_tokens,
        "next_root": next_root,
    }
    return data_dict

def discretize(
    t,
    num_discrete,
    continuous_range = (-0.5, 0.5),
):
    lo, hi = continuous_range
    assert hi > lo

    t = (t - lo) / (hi - lo)
    t *= num_discrete
    t -= 0.5

    return t.round().astype(np.int32).clip(min = 0, max = num_discrete - 1)

def undiscretize(
    t,
    num_discrete = 128,
    continuous_range = (-0.5, 0.5),
):
    lo, hi = continuous_range
    assert hi > lo

    t += 0.5
    t /= num_discrete
    return t * (hi - lo) + lo

def undiscretize_tensor(
    t: torch.Tensor,
    num_discrete,
    continuous_range: tuple = (-0.5, 0.5),
) -> torch.Tensor:
    """
    Convert a discretized tensor back to continuous space
    
    Args:
        t: Input tensor, usually a discrete value of Long type
        continuous_range: Range of continuous values (low, high)
        num_discrete: Number of discrete bins
    
    Returns:
        The converted continuous tensor
    """
    lo, hi = continuous_range
    assert hi > lo, "continuous_range must have hi > lo"
    
    # Ensure the input is float type for calculation
    if t.dtype in (torch.long, torch.int32, torch.int64):
        t = t.float()
    
    # Perform the conversion operation
    t = t + 0.5
    t = t / num_discrete
    t = t * (hi - lo) + lo
    
    return t
        
def normalize(mesh):
    bounds = np.array([mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)])
    mesh.vertices = mesh.vertices - (bounds[0] + bounds[1])[None, :] / 2
    mesh.vertices = mesh.vertices / (bounds[1] - bounds[0]).max()
    mesh.vertices = mesh.vertices.clip(-0.5, 0.5)  # Clamp vertex coordinates to [-0.5, 0.5] 

def discrete_and_clean(mesh, n_discrete):
    vertices = discretize(mesh.vertices, num_discrete=n_discrete)
    new_mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces)
    new_mesh.merge_vertices()
    new_mesh.update_faces(new_mesh.nondegenerate_faces())
    new_mesh.update_faces(new_mesh.unique_faces())
    new_mesh.remove_unreferenced_vertices()
    return new_mesh

