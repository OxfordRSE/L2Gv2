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
    
def total_loss(Rotations, scales, translations, nodes,  patches,dim,  k, rand=False):
    'R: list of orthogonal matrices for embeddings of patches.'
    l=0
    fij=dict()
    
    
    for i, p in enumerate(patches):
        for j, q in enumerate(patches[i+1:]):
            if rand:
                for n in random.sample(nodes[i,j+i+1], min(k*dim+1, len(nodes[i, i+j+1]))) :
                    theta1=scales[i]*p.get_coordinate(n)@Rotations[i]+ translations[i]
                    theta2= scales[j+i+1]*q.get_coordinate(n)@Rotations[j+i+1]+translations[j+i+1]
                    l+=np.linalg.norm(theta1-theta2)**2
                
                    fij[(i, j+1+i, n)]=[theta1, theta2]
                
            else:
                for n in nodes[i,j+i+1]:
                    theta1=scales[i]*Rotations[i]@ p.get_coordinate(n)+ translations[i]
                    theta2= scales[j+i+1]*Rotations[j+i+1]@q.get_coordinate(n)+translations[j+i+1]
                    l+=np.linalg.norm(theta1-theta2)**2
                
                    fij[(i, j+1+i, n)]=[theta1, theta2]

    return 1/len(patches)*l, fij

def loss(Rotations, scales, translations, nodes,  patches, dim,k, consecutive=False, random_choice_in_intersections=False, fij=False):
    'R: list of orthogonal matrices for embeddings of patches.'

    if consecutive:
        l, f = consecutive_loss(Rotations, scales, translations  ,nodes, patches,  dim,k,  rand=random_choice_in_intersections)
        if fij:
            return l, f
        else:
            return l
    else: 
        l, f= total_loss(Rotations, scales, translations, nodes,  patches,dim,k,  rand=random_choice_in_intersections)
        if fij:
            return l, f
        else:
            return l
        
            
def consecutive_loss(Rotations, scales, translations ,nodes,  patches , dim,k, rand=True):
    'R: list of orthogonal matrices for embeddings of patches.'
    l=0
    fij=dict()
  
    
    
    for i in range(len(patches)-1):
        if rand:
            for n in random.sample(nodes[i,i+1], min(k*dim+1, len(nodes[i,i+1]))):
                theta1=scales[i]*patches[i].get_coordinate(n)@Rotations[i]+ translations[i]
                theta2= scales[i+1]*patches[i+1].get_coordinate(n)@Rotations[i+1]+translations[i+1]
                l+=np.linalg.norm(theta1-theta2)**2
                
                fij[(i, 1+i, n)]=[theta1, theta2]
        else:
            for n in nodes[i,i+1]:
                theta1=scales[i]*patches[i].get_coordinate(n)@Rotations[i]+ translations[i]
                theta2= scales[i+1]*patches[i+1].get_coordinate(n)@Rotations[i+1]+translations[i+1]
                l+=np.linalg.norm(theta1-theta2)**2
                
                fij[(i, 1+i, n)]=[theta1, theta2]
            

    return l, fij


    




def ANPloss_nodes_consecutive_patches(Rotations, scales, translations , patches, nodes, dim,k, rand=True):
    'R: list of orthogonal matrices for embeddings of patches.'
    l=0
    #fij=dict()
    for i in range(len(patches)-1):
        if rand:
            
            for n in random.sample(nodes[i,i+1], min(k*dim+1, len(nodes[i,i+1]))):
                theta1=scales[i]*patches[i].get_coordinate(n)@Rotations[i]+ translations[i]
                theta2= scales[i+1]*patches[i+1].get_coordinate(n)@Rotations[i+1]+translations[i+1]
                l+=anp.linalg.norm(theta1-theta2)**2
        else:
            for n in nodes[i,i+1]:
                theta1=scales[i]*patches[i].get_coordinate(n)@Rotations[i]+ translations[i]
                theta2= scales[i+1]*patches[i+1].get_coordinate(n)@Rotations[i+1]+translations[i+1]
                l+=anp.linalg.norm(theta1-theta2)**2

    return l #, fij

def ANPloss_nodes(Rotations, scales, translations , patches, nodes, dim,k, rand=True):
    'R: list of orthogonal matrices for embeddings of patches.'
    l=0
    #fij=dict()
    
    
    for i, p in enumerate(patches):
        for j, q in enumerate(patches[i+1:]):
            if rand:
                for n in random.sample(nodes[i,j+i+1], min(k*dim+1, len(nodes[i, j+1+i]))) :
                    theta1=scales[i]*p.get_coordinate(n)@Rotations[i]+ translations[i]
                    theta2= scales[j+i+1]*q.get_coordinate(n)@Rotations[j+i+1]+translations[j+i+1]
                    l+=anp.linalg.norm(theta1-theta2)**2
                
                    #fij[(i, j+1+i, n)]=[theta1, theta2]
                
            else:
                for n in nodes[i,j+i+1]:
                    theta1=scales[i]*Rotations[i]@ p.get_coordinate(n)+ translations[i]
                    theta2= scales[j+i+1]*Rotations[j+i+1]@q.get_coordinate(n)+translations[j+i+1]
                    l+=anp.linalg.norm(theta1-theta2)**2
                
                    #fij[(i, j+1+i, n)]=[theta1, theta2]

    return 1/len(patches)*l #fij



def optimization(patches, nodes,k,consecutive=True,   random_choice=True ):
    n_patches=len(patches)
    dim= np.shape(patches[0].coordinates)[1]


    anp.random.seed(42)

    Od=[pymanopt.manifolds.SpecialOrthogonalGroup(dim) for i in range(n_patches)]
    Rd=[ pymanopt.manifolds.Euclidean(dim) for i in range(n_patches)]
    R1=[ pymanopt.manifolds.Euclidean(1) for i in range(n_patches)]
    prod=Od + Rd + R1


    manifold = pymanopt.manifolds.product.Product(prod)


    
    if consecutive:
        @pymanopt.function.autograd(manifold)
        def cost(*R):
    
            Rs=[r for r in R[:n_patches]]
            ts=[t for t in R[n_patches:2*n_patches]]
            ss=[sr for sr in R[2*n_patches :]]
            return ANPloss_nodes_consecutive_patches(Rs, ss, ts , patches, nodes, dim, k, rand=random_choice)
    else:
        @pymanopt.function.autograd(manifold)
        def cost(*R):
    
            Rs=[r for r in R[:n_patches]]
            ts=[t for t in R[n_patches:2*n_patches]]
            ss=[sr for sr in R[2*n_patches :]]
            return ANPloss_nodes(Rs, ss, ts , patches, nodes, dim, k, rand=random_choice)
        

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



def loss_dictionary(Rs, ss, ts, nodes, patches, dim, k):
    L=dict()
    for i in range(2):
        for j in range(2):
            L[i,j]=loss(Rs, ss, ts, nodes,  patches,
                            dim, k, consecutive=i, random_choice_in_intersections=j, fij=False)
    return L