# Code documentation

## Anomaly detection

### l2gv2.anomaly_detection.get_outliers(patch_emb, patch_data, emb, k)

### l2gv2.anomaly_detection.nodes_in_patches(patch_data)

### l2gv2.anomaly_detection.normalized_anomaly(patch_emb, patch_data, emb)

### l2gv2.anomaly_detection.raw_anomaly_score_node_patch(aligned_patch_emb, emb, node)

## Induced subgraph

### l2gv2.induced_subgraph.induced_subgraph(data: torch_geometric.data.Data, nodes, extend_hops=0)

## Manopt optimization

### l2gv2.manopt_optimization.ANPloss_nodes_consecutive_patches(Rotations, scales, translations, patches, nodes, dim, random_choice=True)

R: list of orthogonal matrices for embeddings of patches.

### l2gv2.manopt_optimization.double_intersections_nodes(patches)

### l2gv2.manopt_optimization.optimization(patches, nodes, dim)

## Models

### l2gv2.models.Node2Vec_(data, emb_dim, w_length=20, c_size=10, w_per_node=10, n_negative_samples=1, p=1, q=1, num_epochs=100)

### l2gv2.models.Node2Vec_patch_embeddings(patch_data, emb_dim, w_length=20, c_size=10, w_per_node=10, n_negative_samples=1, p=1, q=1, num_epochs=100)

### l2gv2.models.VGAE_patch_embeddings(patch_data, dim=100, hidden_dim=32, num_epochs=100, decoder=None, device='cpu', lr=0.01)

### *class* l2gv2.models.VGAEconv(\*args: Any, \*\*kwargs: Any)

Bases: `Module`

#### forward(data: torch_geometric.data.Data)

### l2gv2.models.chunk_embedding(chunk_size, patches, dim=2)

### l2gv2.models.speye(n, dtype=torch.float)

identity matrix of dimension n as sparse_coo_tensor.

### l2gv2.models.train(data, model, loss_fun, num_epochs=100, verbose=True, lr=0.01, logger=<function <lambda>>)

<!-- Embeddings -->
<!-- ---------- -->
<!-- .. automodule:: l2gv2.embedding.embeddings -->
<!-- :members: -->
<!-- :undoc-members: -->
<!-- :show-inheritance: -->

## Patch

### *class* l2gv2.patch.lazy.BaseLazyCoordinates

Bases: `ABC`

#### *abstract property* shape

### *class* l2gv2.patch.lazy.LazyCoordinates(x, shift=None, scale=None, rot=None)

Bases: [`BaseLazyCoordinates`](#l2gv2.patch.lazy.BaseLazyCoordinates)

#### save_transform(filename)

#### *property* shape

### *class* l2gv2.patch.lazy.LazyFileCoordinates(filename, \*args, \*\*kwargs)

Bases: [`LazyCoordinates`](#l2gv2.patch.lazy.LazyCoordinates)

#### *property* shape

### *class* l2gv2.patch.lazy.LazyMeanAggregatorCoordinates(patches)

Bases: [`BaseLazyCoordinates`](#l2gv2.patch.lazy.BaseLazyCoordinates)

#### as_array(out=None)

#### get_coordinates(nodes, out=None)

#### *property* shape

<a id="module-l2gv2.patch.patch"></a>

### *class* l2gv2.patch.patch.FilePatch(nodes, filename, shift=None, scale=None, rot=None)

Bases: [`Patch`](#l2gv2.patch.patch.Patch)

### *class* l2gv2.patch.patch.MeanAggregatorPatch(patches)

Bases: [`Patch`](#l2gv2.patch.patch.Patch)

#### get_coordinate(node)

get coordinate for a single node

Args:
: node: Integer node index

#### get_coordinates(nodes)

get coordinates for a list of nodes

Args:
: nodes: Iterable of node indeces

#### *property* patches

### *class* l2gv2.patch.patch.Patch(nodes, coordinates=None)

Bases: `object`

Class for patch embedding

#### coordinates *= None*

patch embedding coordinates

#### get_coordinate(node)

get coordinate for a single node

Args:
: node: Integer node index

#### get_coordinates(nodes)

get coordinates for a list of nodes

Args:
: nodes: Iterable of node indeces

#### index *= None*

mapping of node index to patch coordinate index

#### *property* shape

shape of patch coordinates

(shape[0] is the number of nodes in the patch
and shape[1] is the embedding dimension)

<!-- .. automodule:: l2gv2.patch.patches -->
<!-- :members: -->
<!-- :undoc-members: -->
<!-- :show-inheritance: -->
<!-- .. automodule:: l2gv2.patch.utils -->
<!-- :members: -->
<!-- :undoc-members: -->
<!-- :show-inheritance: -->
