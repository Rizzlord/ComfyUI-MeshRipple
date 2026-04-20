# ComfyUI-MeshRipple

A ComfyUI implementation of **MeshRipple**, a powerful autoregressive model for generating high-quality 3D meshes from point clouds. This node allows you to convert point clouds into clean, water-tight triangular meshes directly within ComfyUI.

## Features
- **High-Quality Mesh Generation**: Generates manifold meshes from sparse or dense point clouds.
- **Multiple Model Versions**: Supports both the standard `10k` model and the `NSA` (Nested Sparse Attention) variant.
- **Integrated MICHE Encoder**: Uses the Michelangelo (MICHE) point-cloud encoder for robust feature extraction.
- **Performance Optimized**: Supports KV-caching for fast generation and optional model compilation for math kernels.

## Installation

1. Clone this repository into your `ComfyUI/custom_nodes` directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/Rizzlord/ComfyUI-MeshRipple
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Model Setup

You need to place the following models in your ComfyUI models directory:

### MeshRipple Weights
Place these in `ComfyUI/models/meshripple/`:
- `meshRipple_10k.pth`
- `meshRipple_nsa.pth`

### MICHE Weights
Place these in `ComfyUI/models/miche/`:
- `shapevae-256.ckpt`
- `shapevae-256.yaml` (Ensure `clip_model_version` is set to `"openai/clip-vit-large-patch14"`)

## Nodes

### 1. MeshRipple Model Loader
Loads the MeshRipple transformer and the associated MICHE conditioner.
- **model_name**: Choose between the standard 10k model or the NSA version.
- **compile_model**: If True, compiles the math kernels (takes time on first run but speeds up generation).

### 2. MeshRipple Generator
The main node that generates the mesh.
- **points**: The input point cloud (must include normals, total 6 channels).
- **sample_points**: Number of points to sample from the input (default 16384).
- **max_faces**: Safety limit for the number of faces to generate.
- **top_k / top_p / temperature**: Sampling parameters to control mesh diversity and quality.
- **use_kv_cache**: Enables KV caching for a massive speed boost (mathematically lossless).

### 3. Trimesh to Points (16k)
A utility node to convert a standard `TRIMESH` object into a sampled point cloud compatible with MeshRipple.

## Technical Details
MeshRipple represents meshes as a sequence of triangles. The model predicts the vertices of each face in an autoregressive manner, conditioned on a point cloud features extracted by the Michelangelo encoder.

## Acknowledgments
This node is based on the [MeshRipple](https://github.com/Zheng-Zhiyuan/MeshRipple) research project and incorporates encoders from [Michelangelo](https://github.com/Bytedance/Michelangelo).
