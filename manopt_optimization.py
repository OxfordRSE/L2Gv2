import autograd.numpy as anp
import pymanopt
import pymanopt.manifolds
import pymanopt.optimizers
import random
import local2global as l2g
import local2global.example as ex
import numpy as np

def double_intersections_nodes(patches):
    double_intersections=dict()
    for i in range(len(patches)):
        for j in range(i+1, len(patches)):
            double_intersections[(i,j)]=list(set(patches[i].nodes.tolist()).intersection(set(patches[j].nodes.tolist())))
    return double_intersections
    

def ANPloss_nodes_consecutive_patches(Rotations, scales, translations , patches, nodes, dim, random_choice=True):
    'R: list of orthogonal matrices for embeddings of patches.'
    l=0
    #fij=dict()
    for i in range(len(patches)-1):
        if random_choice:
            for n in random.sample(nodes[i,i+1], dim+1):
                theta1=scales[i]*patches[i].get_coordinate(n)@Rotations[i]+ translations[i]
                theta2= scales[i+1]*patches[i+1].get_coordinate(n)@Rotations[i+1]+translations[i+1]
                l+=anp.linalg.norm(theta1-theta2)**2
        else:
            for n in nodes[i,i+1]:
                theta1=scales[i]*patches[i].get_coordinate(n)@Rotations[i]+ translations[i]
                theta2= scales[i+1]*patches[i+1].get_coordinate(n)@Rotations[i+1]+translations[i+1]
                l+=anp.linalg.norm(theta1-theta2)**2
            

    return l #, fij

def optimization(patches,nodes, dim ):
    n_patches=len(patches)


    anp.random.seed(42)

    Od=[pymanopt.manifolds.SpecialOrthogonalGroup(dim) for i in range(n_patches)]
    Rd=[ pymanopt.manifolds.Euclidean(dim) for i in range(n_patches)]
    R1=[ pymanopt.manifolds.Euclidean(1) for i in range(n_patches)]
    prod=Od + Rd + R1


    manifold = pymanopt.manifolds.product.Product(prod)


    @pymanopt.function.autograd(manifold)
    def cost(*R):
    
        Rs=[r for r in R[:n_patches]]
        ts=[t for t in R[n_patches:2*n_patches]]
        ss=[sr for sr in R[2*n_patches :]]
        return ANPloss_nodes_consecutive_patches(Rs, ss, ts, patches, nodes, dim )

    problem = pymanopt.Problem(manifold, cost)

    optimizer = pymanopt.optimizers.SteepestDescent()
    result = optimizer.run(problem,  reuse_line_searcher=True)

    Rotations=result.point[:n_patches]

    translations=result.point[n_patches:2*n_patches]

    scales=result.point[2*n_patches:]
    emb_problem = l2g.AlignmentProblem(patches)

    embedding = np.empty((emb_problem.n_nodes, emb_problem.dim))
    for node, patch_list in enumerate(emb_problem.patch_index):
        embedding[node] = np.mean([scales[i]*emb_problem.patches[p].get_coordinate(node)@Rotations[i] + translations[i] for i, p in enumerate(patch_list)], axis=0)
    
    
    return result, embedding

