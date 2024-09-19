import biobox as bb
import torch
import plotly.graph_objects as go
from PSIPREDauto.functions import single_submit
import os, shutil
import numpy as np
import rmsd
import torch.nn as nn
import pytorch3d.transforms as trans


def get_atomic_form_factor_dict(q, device):
    # Extended atomic form factor parameters for common elements using HF and RHF methods
    params = {
        'H': [0.489918, 0.262003, 0.196767, 0.049879, 20.6593, 7.74039, 49.5519, 2.20159, 0.001305],
        'C': [2.31000, 1.02000, 1.58860, 0.865000, 20.8439, 10.2075, 0.568700, 51.6512, 0.215600],
        'N': [12.2126, 3.13220, 2.01250, 1.16630, 0.005700, 9.89330, 28.9975, 0.582600, 0.255300],
        'O': [3.04850, 2.28680, 1.54630, 0.867000, 13.2771, 5.70110, 0.323900, 32.9089, 0.250800],
        'CA': [2.31000, 1.02000, 1.58860, 0.865000, 20.8439, 10.2075, 0.568700, 51.6512, 0.215600],
        'S': [6.90530, 5.20340, 1.43790, 1.58630, 1.46790, 22.2151, 0.253600, 56.1720, 0.215600]
        # Add more elements as needed
    }

    volLst = {
        'H': 5.15,
        'C': 16.44,
        'N': 2.49,
        'O': 9.13,
        'CA': 16.44,
        'S': 19.5432
        # Add more elements as needed
    }
    
    form_factor_q_dict = {}

    # Ensure q is a PyTorch tensor on the correct device
    q = torch.as_tensor(q, device=device)

    # calculate the water scattering
    # scattering of oxygen
    a1, a2, a3, a4, b1, b2, b3, b4, c = params['O']
    fh = a1 * torch.exp(-b1 * (q / (4 * torch.pi))**2) + \
         a2 * torch.exp(-b2 * (q / (4 * torch.pi))**2) + \
         a3 * torch.exp(-b3 * (q / (4 * torch.pi))**2) + \
         a4 * torch.exp(-b4 * (q / (4 * torch.pi))**2) + c
    
    # scattering of 2 hydrogen
    a1, a2, a3, a4, b1, b2, b3, b4, c = params['H']
    fh += 2.0 * (a1 * torch.exp(-b1 * (q / (4 * torch.pi))**2) + \
                 a2 * torch.exp(-b2 * (q / (4 * torch.pi))**2) + \
                 a3 * torch.exp(-b3 * (q / (4 * torch.pi))**2) + \
                 a4 * torch.exp(-b4 * (q / (4 * torch.pi))**2) + c)
    
    for element in params.keys():
        # add the amino acid scattering
        a1, a2, a3, a4, b1, b2, b3, b4, c = params[element]
        f_q = a1 * torch.exp(-b1 * (q / (4 * torch.pi))**2) + \
              a2 * torch.exp(-b2 * (q / (4 * torch.pi))**2) + \
              a3 * torch.exp(-b3 * (q / (4 * torch.pi))**2) + \
              a4 * torch.exp(-b4 * (q / (4 * torch.pi))**2) + c

        #add an excluded volume factor
        vr = volLst[element]
        fex = vr * torch.exp(-torch.pi * q * q * torch.pow(torch.tensor(vr, device=device), 1.5))

        # hydration factor
        
        form_factor_q_dict[element] = [f_q, fex, fh]

    return {element: [f_q.to(device), fex.to(device), fh.to(device)] for element, (f_q, fex, fh) in form_factor_q_dict.items()}
    

def pdb_to_ca(pdb_fl, chain='A', device='cuda'):
    """ Reads in a PDB file, returns the CA backbone for a given chain 

    Args:
        pdb_fl (str): PDB file path.
        chain (str): Chain ID of interest.

    Returns: 
        (torch.tensor), size ([N,3]).
    """
    mol = bb.Molecule(pdb_fl)
    idx1 = (mol.data['name']=='CA').values
    idx2 = (mol.data['name']=='CA A').values
    CA = torch.from_numpy(mol.coordinates[0][np.logical_or(idx1,idx2)].copy())
    return CA.float()


def get_neighbour_distances(CA):
    """ Computes the distance between residue i and residue i+1.

    Args:
        CA (torch.tensor): CA backbone, size ([N,3])

    Returns: 
        (torch.tensor), size ([N-1]).
    """
    return torch.cdist(CA, CA)[1:,].diagonal()

    
def get_nonneighbour_distances(CA):
    """ Computes the distances between residues i and j, where j>i+1. 
    
    Args:
        CA (torch.tensor): CA backbone, size ([N,3])
        
    Returns: 
        (torch.tensor), size ([(N-2)*(N-1)/2])
    """
    cdist = torch.cdist(CA, CA)
    N = cdist.shape[0]
    mask = torch.triu(torch.ones((N, N), dtype=bool, device=CA.device), diagonal=2)
    return cdist[mask]


def curvature(CA):
    """ Computes the curvature of the subsection (i,i+1,i+2,i+3)
    
    Args:
        CA (torch.tensor): CA backbone, size ([N,3])
        
    Returns: 
        (torch.tensor), size ([N-3])
    """
    v1, v2, v3, v4 = CA[:-3], CA[1:-2], CA[2:-1], CA[3:]
    m1, m2, m3 = (v2+v1)/2, (v3+v2)/2, (v4+v3)/2
    v1, v2 = m1-m3, m2-m3
    cross = torch.cross(v1, v2, dim=-1)
    sin_theta = torch.linalg.norm(cross, dim=-1) / (torch.linalg.norm(v1, dim=-1) * torch.linalg.norm(v2, dim=-1))
    return (2*torch.abs(sin_theta)) / torch.linalg.norm(m1-m2, dim=-1)


def torsion(v):
    """ Computes the torsion of the subsection (i,i+1,i+2,i+3)
    
    Args:
        CA (torch.tensor): CA backbone, size ([N,3])
        
    Returns: 
        (torch.tensor), size ([N-3])
    """
    v1, v2, v3, v4 = v[:-3], v[1:-2], v[2:-1], v[3:]
    e1, e2, e3 = v2-v1, v3-v2, v4-v3
    n1 = torch.nn.functional.normalize(torch.cross(e1, e2, dim=-1), dim=-1)
    n2 = torch.nn.functional.normalize(torch.cross(e2, e3, dim=-1), dim=-1)
    cos_theta = (n1*n2).sum(dim=-1)
    theta = torch.arccos(cos_theta)
    idx = torch.where((torch.cross(n1, n2, dim=-1)*e2).sum(dim=-1) < 0)
    theta[idx] *= -1
    length = (torch.linalg.norm(e1, dim=-1) + torch.linalg.norm(e2, dim=-1) + torch.linalg.norm(e3, dim=-1)) / 3
    return (2/length) * torch.sin(theta/2)

    
def wij_matrix(CA):
    """ Computes the the exact evalutation of the Gauss integral over line segments i - i+1, j - j+1
    
    Args:
        CA (torch.tensor): CA backbone, size ([N,3])
        
    Returns: 
        (torch.tensor), size ([N-1,N-1])
    """
    starts = CA[:-1]
    ends = CA[1:]
    A = starts.reshape(1,-1,3) - starts.reshape(-1,1,3)
    B = ends.reshape(1,-1,3) - ends.reshape(-1,1,3)
    C = ends.reshape(1,-1,3) - starts.reshape(-1,1,3)
    n1 = torch.nn.functional.normalize(torch.cross(A,C,dim=-1),dim=-1)
    n2 = torch.nn.functional.normalize(torch.cross(C,B,dim=-1),dim=-1)
    n3 = torch.nn.functional.normalize(torch.cross(B,-C.transpose(1,0),dim=-1),dim=-1)
    n4 = torch.nn.functional.normalize(torch.cross(-C.transpose(1,0),A,dim=-1),dim=-1)
    arg1 = (n1*n2).sum(dim = -1)
    arg2 = (n2*n3).sum(dim = -1)
    arg3 = (n3*n4).sum(dim = -1)
    arg4= (n4*n1).sum(dim = -1)
    epsilon=1e-8
    arg1 = torch.clamp(arg1,-1+epsilon,1-epsilon)
    arg2 = torch.clamp(arg2,-1+epsilon,1-epsilon)
    arg3 = torch.clamp(arg3,-1+epsilon,1-epsilon)
    arg4 = torch.clamp(arg4,-1+epsilon,1-epsilon)
    omega_star = torch.arcsin(arg1) +torch.arcsin(arg2) + torch.arcsin(arg3) + torch.arcsin(arg4) 
    #omega_star = arg1 +arg2 + arg3 + arg4 
    Cii = ends-starts
    r12 = Cii.reshape(-1,1,3)
    r34 = Cii.reshape(1,-1,3)
    sign_arg = (torch.cross(r34,r12,dim=-1)*A).sum(dim=-1)
    #print(torch.tanh(sign_arg))
    #sign_arg = torch.round(sign_arg,decimals=10)
    #print(torch.sign(sign_arg))
    return (omega_star*torch.tanh(100.0*sign_arg))/(4*torch.pi)
    #return (omega_star)/(4*torch.pi)


def secondary_structure_transform_matrix(x, y, gamma, device):
    """ Computes the matrix for a rigid transorm of e1 to a given subsection.

    Args:
        x (torch.tensor): First point of subsection, size ([1,3])
        y (torch.tensor): Last point of subsection, size ([1,3])
        gamma (torch.tensor): Angle of rotation about the axis y-x, size ([1])
    
    Returns: 
        (torch.tensor), size ([3,3])
    """
    t = y - x
    r = torch.norm(t)
    theta = torch.arccos(t[2] / r)
    phi = torch.sign(t[1]) * torch.arccos(t[0] / torch.norm(t[:2]))
    alpha, beta = phi, -(torch.pi/2 - theta)
    roll = torch.tensor([[1., 0., 0.],
                         [0., torch.cos(gamma), -torch.sin(gamma)],
                         [0., torch.sin(gamma), torch.cos(gamma)]], device=device)
    pitch = torch.tensor([[torch.cos(beta), 0., torch.sin(beta)],
                          [0., 1., 0.],
                          [-torch.sin(beta), 0., torch.cos(beta)]], device=device)
    yaw = torch.tensor([[torch.cos(alpha), -torch.sin(alpha), 0.],
                        [torch.sin(alpha), torch.cos(alpha), 0.],
                        [0., 0., 1.]], device=device)
    scale = torch.tensor([[r, 0, 0], [0, 1, 0], [0, 0, 1]], device=device)
    return yaw @ pitch @ roll @ scale

    
def neighbouring_distances_gaussian(CA,t):
    """ Evaluates the log of a Normal PDF for each neighbouring distance of the CA backbone.

    Args:
        CA (torch.tensor): CA backbone, size ([N,3])
    
    Returns:
        (torch.tensor), size([N-1])
    """
    mean = torch.tensor(3.80523548, device=CA.device)
    std = torch.tensor(0.02116009, device=CA.device) + t
    neighb_norm = torch.distributions.Normal(mean, std)
    return neighb_norm.log_prob(get_neighbour_distances(CA))
    

def nonneighbouring_distances_gaussian(CA,t):
    """ Evaluates the log of a Logistic CDF for each nonneighbouring distance of the CA backbone.

    Args:
        CA (torch.tensor): CA bacbkone, size([N,3])

    Returns: 
        (torch.tensor), size([(N-2)*(N-1)/2])
    """
    mean = torch.tensor(3.97512675, device=CA.device)
    std = torch.tensor(0.12845008, device=CA.device) + t
    base_distribution = torch.distributions.Uniform(0, 1)
    transforms = [torch.distributions.SigmoidTransform().inv, torch.distributions.AffineTransform(loc=mean, scale=std)]
    logistic = torch.distributions.TransformedDistribution(base_distribution, transforms)
    return torch.log(logistic.cdf(get_nonneighbour_distances(CA)))


def nonneighbouring_distances_gaussian_full(distances,t):
    """ Evaluates the log of a Logistic CDF for each nonneighbouring distance of the CA backbone.

    Args:
        CA (torch.tensor): CA bacbkone, size([N,3])

    Returns: 
        (torch.tensor), size([(N-2)*(N-1)/2])
    """
    mean = torch.tensor(1.5, device=distances.device)
    std = torch.tensor(0.12845008, device=distances.device) + t
    base_distribution = torch.distributions.Uniform(0, 1)
    transforms = [torch.distributions.SigmoidTransform().inv, torch.distributions.AffineTransform(loc=mean, scale=std)]
    logistic = torch.distributions.TransformedDistribution(base_distribution, transforms)
    return torch.log(logistic.cdf(distances))


def curvature_torsion_gmm(CA, t):
    """ Evaluates the log of a GaussianMixture CDF for each (curvature, torsion) pair along the CA backbone.

    Args:
        CA (torch.tensor): CA backbone, size([N,3])

    Returns: 
        (torch.tensor), size ([N-3])
    """
    means = torch.tensor([
       [ 0.06058326,  0.52086702],
       [ 0.20354845, -0.43681879],
       [ 0.32653185,  0.10680209],
       [ 0.28794512,  0.36394794],
       [ 0.44709165,  0.23228048],
       [ 0.25774574, -0.2966153 ],
       [ 0.23052334,  0.43603841],
       [ 0.05777108, -0.5207016 ],
       [ 0.31934864,  0.24332755],
       [ 0.30096342, -0.0863423 ],
       [ 0.28108115, -0.41811307],
       [ 0.16042419, -0.4701952 ],
       [ 0.41742309,  0.03588332],
       [ 0.41473458, -0.20601845],
       [ 0.11983773,  0.06702979],
       [ 0.22570592, -0.37314479],
       [ 0.14809181,  0.49104272],
       [ 0.11111029, -0.50235152],
       [ 0.41761349,  0.28318495],
       [ 0.42450639,  0.12904249]
    ], device=CA.device)
    
    # gmm_means = torch.tensor(means)
    
    covs = torch.tensor([
        [[ 7.04155236e-04, -8.83605643e-05],
        [-8.83605643e-05,  2.65604927e-05]],

       [[ 6.39507804e-04,  3.01550713e-05],
        [ 3.01550713e-05,  6.57440951e-04]],

       [[ 1.92306166e-03,  1.81313366e-04],
        [ 1.81313366e-04,  4.48259407e-03]],

       [[ 1.79773879e-03, -6.63194399e-04],
        [-6.63194399e-04,  1.35961282e-03]],

       [[ 3.01273989e-04, -1.34858369e-04],
        [-1.34858369e-04,  1.10979077e-03]],

       [[ 1.43575965e-03, -4.77701253e-06],
        [-4.77701253e-06,  2.92939588e-03]],

       [[ 1.29654408e-03, -6.39874344e-04],
        [-6.39874344e-04,  6.31371399e-04]],

       [[ 5.65304078e-04,  6.61918535e-05],
        [ 6.61918535e-05,  2.41693652e-05]],

       [[ 2.50539292e-03, -7.05901197e-04],
        [-7.05901197e-04,  2.90720293e-03]],

       [[ 2.04970255e-03,  1.45984567e-03],
        [ 1.45984567e-03,  7.20361221e-03]],

       [[ 6.34135540e-04,  2.19431653e-04],
        [ 2.19431653e-04,  3.29865283e-04]],

       [[ 6.35348915e-04,  8.16890067e-05],
        [ 8.16890067e-05,  4.56805530e-04]],

       [[ 4.01761001e-04, -3.92457384e-05],
        [-3.92457384e-05,  3.45932949e-03]],

       [[ 2.91904612e-03,  1.40546236e-03],
        [ 1.40546236e-03,  9.38315748e-03]],

       [[ 3.67445039e-03,  2.25397074e-03],
        [ 2.25397074e-03,  1.33360570e-01]],

       [[ 1.10496781e-03,  4.65014772e-04],
        [ 4.65014772e-04,  1.08117163e-03]],

       [[ 1.51318164e-03, -6.37439661e-04],
        [-6.37439661e-04,  3.39644612e-04]],

       [[ 6.46987825e-04,  1.39159547e-04],
        [ 1.39159547e-04,  1.68725519e-04]],

       [[ 1.12793078e-03, -3.13566732e-04],
        [-3.13566732e-04,  1.21481311e-03]],

       [[ 1.21827541e-03,  2.38071894e-04],
        [ 2.38071894e-04,  2.14505033e-03]]
    ], device=CA.device) + t
    
    # gmm_covs = torch.tensor(covs)+t
    mix = torch.distributions.Categorical(torch.tensor([0.05416937, 0.0781135 , 0.03287735, 0.05483729, 0.07992936,
       0.03406397, 0.06485402, 0.06947408, 0.04092266, 0.02313928,
       0.02514197, 0.07938146, 0.03246065, 0.01171809, 0.02077582,
       0.06535025, 0.06796419, 0.07890037, 0.028788  , 0.05713831], device=CA.device))
    
    comp = torch.distributions.MultivariateNormal(means, covs)
    gmm = torch.distributions.MixtureSameFamily(mix, comp)
    kap = curvature(CA)
    tau = torsion(CA)
    kt = torch.stack((kap,tau),dim=-1)
    
    return gmm.log_prob(kt)
    

# def acn_penalty(CA,t):
#     """ Evaluates the log of a Normal PDF for the ACN of the CA backbone

#     Args:
#         CA (torch.tensor), size ([N,3])

#     Returns:
#         (tensor.float)
#     """  
#     acn = torch.abs(wij_matrix(CA)[torch.nonzero(wij_matrix(CA), as_tuple=True)]).sum()
#     #acn = wij_matrix(CA).sum()
#     x = CA.shape[0]
#     mean = torch.tensor((x/7.78238813)**1.36137177 - 2.37741229, device=CA.device)
#     std = torch.tensor(0.02869433*x+0.52430145, device=CA.device) + t
#     acn_norm = torch.distributions.Normal(mean,std)
#     return acn_norm.log_prob(acn)


def build_SKMT_backbone(params, nonlinker_reps, device):
    linker_params, gammas = params
    CA = []
    for i in range(len(linker_params)-1):
        if len(linker_params[i]) > 2:
            CA.append(linker_params[i][0].unsqueeze(0))
            CA.append(linker_params[i][int(len(linker_params[i])/2)].unsqueeze(0))
        else:
            CA.append(linker_params[i][0].unsqueeze(0))
        mat = secondary_structure_transform_matrix(linker_params[i][-1], linker_params[i+1][0], gammas[i], device).T
        nonlink = linker_params[i][-1] + nonlinker_reps[i] @ mat
        CA.append(nonlink[0].unsqueeze(0))
    CA.append(linker_params[-1][0].unsqueeze(0))
    if not torch.equal(CA[-1], linker_params[-1][-1]):
        CA.append(linker_params[-1][-1].unsqueeze(0))
    return torch.cat(CA, dim=0)

    
def acn_penalty(CA,t):
    """ Evaluates the log of a Normal PDF for the ACN of the CA backbone

    Args:
        CA (torch.tensor), size ([N,3])

    Returns:
        (tensor.float)
    """
    acn = torch.abs(wij_matrix(CA)[torch.nonzero(wij_matrix(CA), as_tuple=True)]).sum()
    x = CA.shape[0]
    mean = torch.tensor((x/4)**1.4 - 6, device=CA.device)
    std = torch.tensor(0.12444077*x + 3.20058622, device=CA.device) + t
    base_distribution = torch.distributions.Uniform(0, 1)
    transforms = [torch.distributions.SigmoidTransform().inv, torch.distributions.AffineTransform(loc=mean, scale=std)]
    logistic = torch.distributions.TransformedDistribution(base_distribution, transforms)
    return torch.log(logistic.cdf(acn))


def load_FP(fp_fl):
    """ Reads in a secondary structure fingerprint, with a simplified 3 letter code.
        H = Helix, S = Strand, - = Linker

    Args:
        fp_fl (str): fingeprint file location
    
    Returns:
        FP (str): secondary structure fingerprint 
    """
    FP = []
    with open(fp_fl) as file:
        for line in file:
            FP.extend(list(line.replace(" ", "")))
    FP = ''.join(FP)
    if FP[0]!='-':
        FP[0] = '-'
    if FP[-1]!='-':
        FP[-1] = '-'
    return FP


def get_linker_indices(FP):
    """ Finds the indices of linker residues. 

    Args:
        FP (str): secondary structure fingerprint

    Returns:
        (list)
    """
    return [i for i, c in enumerate(FP) if c=='-']

def find_intervals(nums):
    """ Computes the intervals corresponding to a list of integers containing gaps
        e.g. find_intervals([0,1,2,5,7,8,9]) = [[0,2],[5,5],[7,9]] 
    
    Args:
        nums (list): list of integers

    Returns:
        intervals (list of lists): list of (first integer, last integer) tuples
    """
    intervals = []
    start = None
    
    for i in range(len(nums)):
        if i == 0 or nums[i] != nums[i-1] + 1:
            if start is not None:
                intervals.append([start, nums[i-1]])
            start = nums[i]
    
    if start is not None:
        intervals.append([start, nums[-1]])
    
    return intervals


def get_linker_intervals(FP):
    """ Finds the start and end index of linker subsections.

    Args:
        FP (str): secondary structure fingerprint

    Returns:
        (list of lists): list of (start index, end index) tuples
    """
    linkers = get_linker_indices(FP)
    return find_intervals(linkers)

def get_nonlinker_intervals(FP):
    """ For each nonlinker subsection, finds the index of endpoint of the previous linker, and the index of the start point of the subsequent linker.

    Args
        FP (str): secondary structure fingerprint

    Returns:
        (list of lists): list of (start index, end index) tuples
    """
    lindex = get_linker_intervals(FP)
    return [[lindex[i][1],lindex[i+1][0]] for i in range(len(lindex)-1)]

def single_representative(nonlinker_subsec, x, y, device):
    """ Maps a given nonlinker subsection to a canonical representative lying between [0,1] on the x axis.

    Args:
        nonlinker_subsec (torch.tensor): Nonlinker subsection of CA backbone, size ([m,3])

    Returns:
        (torch.tensor), size([m,3])
    """
    return (nonlinker_subsec-x) @ torch.linalg.inv(secondary_structure_transform_matrix(x, y, torch.tensor(0., dtype=torch.double), device=device)).T

def nonlinker_representatives(CA,FP,device):
    """
    Given a CA backbone and secondary structure fingeprint, returns a list of the canonical representative
    of each nonlinker subsection.
    """
    nonlinker_intervals = get_nonlinker_intervals(FP)
    reps = []
    for i in nonlinker_intervals:
        reps.append(single_representative(CA[i[0]+1:i[1]], CA[i[0]], CA[i[1]], device))
    return reps


def get_linker_params(CA,FP):
    """Returns a list of tensors, each tensor is the coordinates of linker subsection."""
    return [CA[i[0]:i[1]+1] for i in get_linker_intervals(FP)]


def get_params(CA, FP, device):
    """Returns a tuple of the linker subsections, and the gammas (initialised to 0)
    The gammas represent the rotation for the nonlinker representative."""
    linker_params = get_linker_params(CA, FP)
    gammas = [torch.tensor(0., dtype=torch.double, device=device) for i in range(len(linker_params)-1)]
    return [linker_params, gammas]


def build_backbone(params, nonlinker_reps, device):
    """Given the linker parameters and gammas, and the canonical nonlinker representatives
    builds a CA backbone tensor."""
    linker_params, gammas = params
    CA = []
    for i in range(len(linker_params)-1):
        CA.append(linker_params[i])
        mat = secondary_structure_transform_matrix(linker_params[i][-1], linker_params[i+1][0], gammas[i], device).T
        nonlink = linker_params[i][-1].to(device) + nonlinker_reps[i].to(device) @ mat
        CA.append(nonlink)
    CA.append(linker_params[-1])
    return torch.cat(CA, dim=0).to(device)
    

def plotMol(coords):
    fig = go.Figure(data=[go.Scatter3d(
        x=coords[:,0].cpu(), y=coords[:,1].cpu(), z=coords[:,2].cpu(),
        mode='lines+markers',
        line=dict(color='black', width=20),
        marker=dict(size=1, color='black')
    )])
    fig.update_layout(width=1000, height=1000, showlegend=False,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data'
        )
    )
    fig.show()


def pdb_to_fasta(pdb_file_loc,chain):
### Opens a PDB file using biobox, retrieves the primary sequence,
### writes it to a fasta file.
    aa3to1={
   'ALA':'A', 'VAL':'V', 'PHE':'F', 'PRO':'P', 'MET':'M',
   'ILE':'I', 'LEU':'L', 'ASP':'D', 'GLU':'E', 'LYS':'K',
   'ARG':'R', 'SER':'S', 'THR':'T', 'TYR':'Y', 'HIS':'H',
   'CYS':'C', 'ASN':'N', 'GLN':'Q', 'TRP':'W', 'GLY':'G',
   'MSE':'M', 'HID': 'H', 'HIP': 'H'
    }
    M = bb.Molecule(pdb_file_loc)
    CA,idx  = M.atomselect(chain,'*','CA',get_index=True)
    aa = [aa3to1[i[0]] for i in M.get_data(idx,columns=['resname'])]
    with open(pdb_file_loc[:-4]+'.fasta','w+') as fout:
          fout.write(''.join(aa))


def get_secstruc_psipred(pdb_file_loc,chain):
### Using PSIPredauto, runs a PSIPred SecStruc prediction
### for a given PDB file.
    if not os.path.isfile(pdb_file_loc[:-4]+'.fasta'):
        pdb_to_fasta(pdb_file_loc,chain)
    if not os.path.isdir(pdb_file_loc[:-4]+'.fasta output/'):
        single_submit(pdb_file_loc[:-4]+'.fasta', "foo@bar.com", '')


def convert(s):
### Converts a list of letters into a word
    new = ""
    for x in s:
        new+= x
    return new


def simple_ss_clean(fp):
### Simple SecStruc clean up check for singleton SSEs.
    for i in range(len(fp)-1):
        if fp[i-1]==fp[i+1] and fp[i-1]!=fp[i]:
            fp[i]=fp[i-1]
    return convert(fp)


def get_ss_fp_psipred(fasta_file_loc):
### From the outpuit of the PSIPredrun run, converts to the
### simple 3 letter code SecStruc FP.
    dssp_to_simp = {"I" : "H",
                 "S" : "-",
                 "H" : "H",
                 "E" : "S",
                 "G" : "H",
                 "B" : "S",
                 "T" : "-",
                 "C" : "-"
                 }
    lines = []
    with open(fasta_file_loc+' output/'+os.path.splitext(os.path.basename(fasta_file_loc))[0]+'.ss','r') as fin:
        for line in fin:
            lines.append(line.split())
    ss = [dssp_to_simp[i[2]] for i in lines]
    return simple_ss_clean(ss)

def write_fingerprint_file(pdb_file_loc,chain):
### For a given PDB file, sets up a call to PSIPred to predict
### the SecStruc, then writes it to a file in the same dir as the PDB file.
    get_secstruc_psipred(pdb_file_loc,chain)
    ss = get_ss_fp_psipred(pdb_file_loc[:-4]+'.fasta')
    with open(os.path.dirname(pdb_file_loc)+'/fingerPrint1.dat','w+') as fout:
        fout.write(ss)
    os.remove(pdb_file_loc[:-4]+'.fasta')
    shutil.rmtree(pdb_file_loc[:-4]+'.fasta output')


def overlayMolSubsecs(start_coords,current_coords,subsec_start=0,subsect_end=-1):
    ### Plots a CA backbone tensor.
    start_coords = start_coords.numpy()
    current_coords = current_coords.numpy()
    #aligned = rmsd.kabsch_fit(current_coords,start_coords)
    diff_coords = np.array([np.linalg.norm(start_coords[i]-current_coords[i]) for i in range(len(start_coords))])
    subsec_start = max(0,diff_coords.argmax()-50)
    subsect_end = min(diff_coords.argmax()+50,len(diff_coords))
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=start_coords[:,0], 
        y=start_coords[:,1], 
        z=start_coords[:,2],
        name='Start',
        visible='legendonly',
        marker=dict(
            size=1,
            color='black',
        ),
        line=dict(
            color='black',
            width=10

        )
    ))
    fig.add_trace(go.Scatter3d(
        x=current_coords[:,0], 
        y=current_coords[:,1], 
        z=current_coords[:,2],
        name='Current',
        visible='legendonly',
        marker=dict(
            size=1,
            color='blue',
        ),
        line=dict(
            color='blue',
            width=10
        )
    ))
    fig.add_trace(go.Scatter3d(
        x=current_coords[:,0], 
        y=current_coords[:,1], 
        z=current_coords[:,2],
        name='Change',
        marker=dict(
            size=1,
            color=diff_coords,
            colorscale='jet',
            colorbar=dict(thickness=20),
        ),
        line=dict(
            color=diff_coords,
            colorscale='jet',
            colorbar=dict(thickness=20),
            width=10
        )
    ))

    fig.update_layout(width=1000,height=1000)
    fig.update_layout(
        showlegend=True,
        legend=dict(x=0),
        scene=dict(
            xaxis_title='',
            yaxis_title='',
            zaxis_title='',
            aspectratio = dict( x=1, y=1, z=1 ),
            aspectmode = 'manual',
            xaxis = dict(
                gridcolor="white",
                showbackground=False,
                zerolinecolor="white",
                nticks=0,
                showticklabels=False),
            yaxis = dict(
                gridcolor="white",
                showbackground=False,
                zerolinecolor="white",
                nticks=0,
                showticklabels=False),
            zaxis = dict(
                gridcolor="white",
                showbackground=False,
                zerolinecolor="white",
                nticks=0,
                showticklabels=False),),
    )
    fig.show()


def formFactorFold(formFactorSet, c1, c2, device):
    c12 = torch.tensor((1.0, -c1[0], c2[0]), device=device)
    c12lst = c12.tile(len(formFactorSet), 1)
    return torch.einsum("bi,bij->bj", c12lst, formFactorSet.float())


def fastDebye(coords,q_vals,formFactorSetIn,c1,c2):
    formFactorSet = formFactorFold(formFactorSetIn,c1,c2)
    I_q = torch.zeros(len(q_vals))
    # calculate all distatnces
    distSet =torch.cdist(coords,coords)
    distSet = distSet.view(1,*distSet.shape)
    q_vals = q_vals.view(-1,1,1)
    sincArg = q_vals*distSet/torch.pi
    sincArgs = torch.sinc(sincArg)
    formFactorSwitched = torch.swapaxes(formFactorSet,0,1)
    formFactorList = torch.einsum("bi,bj->bij",formFactorSwitched,formFactorSwitched)
    return torch.einsum("bij,bkl->b",formFactorList,sincArgs)


def getSideChianDirecs(cacoords):
    arcs = torch.roll(cacoords, 1, 0) - cacoords 
    direcs = arcs - torch.roll(arcs, 1, 0)
    direcs = torch.nn.functional.normalize(direcs, dim=-1)
    direcs = torch.roll(direcs, -1, 0)
    direcDerivs = torch.roll(direcs, -1, 0) - torch.roll(direcs, -2, 0)
    mask = torch.zeros(len(cacoords), device=cacoords.device)
    mask[0] = mask[-1] = 1.0
    direcsOut = direcs + torch.einsum("bi,b->bi", direcDerivs, mask)
    return torch.nn.functional.normalize(direcsOut, dim=-1)
    

def stripResidueList(residueList):
    return [res for res in residueList] 


def getCalphas(structure):
    residues = stripResidueList(list(structure.get_residues()))
    atoms = np.array([atom for res in residues for atom in res]) 
    cacoordinates = []
    for atom in atoms:
        if atom.get_name() == "CA":
            cacoordinates.append(atom.coord)
    return np.array(cacoordinates)

def readSideChainCoords(structure):
    residues = stripResidueList(list(structure.get_residues()))
    sideChainLens = np.array([len([atom for atom in res])for res in residues])
    atomsPreShift = [np.array([atom.coord for atom in res]) for res in residues]
    atoms =np.concatenate([atomsPreShift[i]-atomsPreShift[i][1] for i in range(len(atomsPreShift))])
    sideChainPos = np.cumsum(sideChainLens)
    direcPDB =np.array([amino[4]-amino[1] if (len(amino) >= 5)  else np.array([0.0,0.0,1.0]) for amino in atomsPreShift])
    return torch.from_numpy(np.insert(sideChainPos,0,0)),torch.from_numpy(sideChainLens),torch.from_numpy(atoms),torch.from_numpy(direcPDB).float()
    

def alignSideChain(CA,aminoCoords,aminoLengthList,direcsStruct,direcsPDB):

    CA = CA.float()
    aminoCoords = aminoCoords.float()
    direcsStruct = direcsStruct.float()
    direcsPDB = direcsPDB.float()
    
    # get the cross product of the model and pdb side chains, this is the direction we rotate the sidehian to fit    
    rotationDirections =torch.nn.functional.normalize(torch.cross(direcsPDB,direcsStruct,dim=1))
    # calculate the vector angles to rotate through
    inner_product = (direcsPDB*direcsStruct).sum(dim=1)
    pdb_norm = direcsPDB.pow(2).sum(dim=1).pow(0.5)
    struct_norm =  direcsStruct.pow(2).sum(dim=1).pow(0.5)
    cos = inner_product/(2.0*pdb_norm*struct_norm)
    angles = torch.acos(cos)
    # set up the rotation
    axisAngle = torch.einsum("bi,b->bi",rotationDirections,angles)
    transformMat = trans.axis_angle_to_matrix(axisAngle)
    # apply the matrix transformation to each set of side chains, need to extend the rotationmatrix list to fit the amino acid list
    transformMat = torch.repeat_interleave(transformMat,aminoLengthList, dim=0)
    # apply the matrix transformation to each amino acid first rotate
    rotatedAminos =torch.einsum("bij,bj->bi",transformMat,aminoCoords)
    # now the translation first extend out the ca coordinate list
    translationMat = torch.repeat_interleave(CA,aminoLengthList, dim=0)
    translatedAminos = rotatedAminos+translationMat
    return translatedAminos

def alignVecs(direcs,aminoCoords,capos):
    # check if its glycine if so do nothing, no sidechain!
    if len(aminoCoords) >= 5:
        # pos 2 (index 1) is ca and pos 5 (index 4) Cb
        shiftedCoords = aminoCoords -aminoCoords[1]
        direcPDB = aminoCoords[4]-aminoCoords[1]
        cp = normalise(np.cross(direcPDB,direc))
        ang = vectorAngle(direc,direcPDB)
        r = R.from_rotvec(ang*cp)
        rmat =r.as_matrix()
        rotCoords =np.array([rmat@cd for cd in shiftedCoords])
        return  rotCoords + capos
    else:
        return  aminoCoords + capos



def fastDebyeUnderlying(CA, aminoPDBCoords, lenList, sideChainDirecsOg, q_vals, device):
    #generate the predicted normals
    normals = getSideChianDirecs(CA)
    coords = alignSideChain(CA, aminoPDBCoords, lenList, normals, sideChainDirecsOg)
    distSet = torch.cdist(coords, coords)
    distSet = distSet.view(1, *distSet.shape)
    q_vals = q_vals.view(-1, 1, 1)
    sincArg = q_vals * distSet / torch.pi
    sincArgs = torch.sinc(sincArg)
    return sincArgs.to(device), torch.cdist(coords, coords).to(device)



def debyeVaryingC(sincArgs, c1c2Pair, q_vals, formFactorSetIn, experimentalData, device):
    formFactorSet = formFactorFold(formFactorSetIn, [c1c2Pair[0]], [c1c2Pair[1]], device)
    formFactorSwitched = torch.swapaxes(formFactorSet, 0, 1)
    formFactorList = torch.einsum("bi,bj->bij", formFactorSwitched, formFactorSwitched)
    modelScatter = torch.einsum("bij,bkl->b", formFactorList, sincArgs)
    modelScatterLog = torch.log(modelScatter)
    mask = q_vals <= 0.1
    experimentalDataLog = torch.log(experimentalData)
    noMean = torch.sum(mask * torch.ones(len(mask), device=device))
    shift = torch.sum((mask * (experimentalDataLog - modelScatterLog))) / noMean
    modelScatterLog = modelScatterLog + shift
    modelScatterNew = torch.exp(modelScatterLog)
    dif = (modelScatterLog - experimentalDataLog)
    qGaussians = qGaussian(q_vals, 0.0, device)
    scatPen = qGaussians.log_prob(dif).sum()
    return scatPen


def findInitialScat(CA, aminoPDBCoords, lenList, sideChainDirecsOg, q_vals, formFactorSetIn, experimentalScatter, device):
    sincArgs, molDists = fastDebyeUnderlying(CA, aminoPDBCoords, lenList, sideChainDirecsOg, q_vals, device)
    c1 = torch.linspace(0.95, 1.15, steps=5, device=device)
    c2 = torch.linspace(-2.0, 4.0, steps=20, device=device)
    c1c2List = torch.tensor([[i, j] for i in c1 for j in c2], device=device)
    return c1c2List[torch.argmax(torch.tensor([debyeVaryingC(sincArgs, i, q_vals, formFactorSetIn, experimentalScatter, device) for i in c1c2List]))]
    

def fastDebyeCA(CA, aminoPDBCoords, lenList, sideChainDirecsOg, q_vals, formFactorSetIn, c1, c2, device):
    formFactorSet = formFactorFold(formFactorSetIn, c1, c2, device)
    normals = getSideChianDirecs(CA)
    coords = alignSideChain(CA, aminoPDBCoords, lenList, normals, sideChainDirecsOg)
    I_q = torch.zeros(len(q_vals), device=device)
    distSet = torch.cdist(coords, coords)
    distSet = distSet.view(1, *distSet.shape)
    q_vals = q_vals.view(-1, 1, 1)
    sincArg = q_vals * distSet / torch.pi
    sincArgs = torch.sinc(sincArg)
    formFactorSwitched = torch.swapaxes(formFactorSet, 0, 1)
    formFactorList = torch.einsum("bi,bj->bij", formFactorSwitched, formFactorSwitched)
    return torch.einsum("bij,bkl->b", formFactorList, sincArgs), torch.cdist(coords, coords)


def qsdWeight(qv):
     return 0.0001 + (2*0.1)/(torch.pi*torch.arctan(torch.tensor(40*(0.2))))*(torch.arctan(40.0*(qv - 0.2)) - torch.arctan(torch.tensor(40*(-0.2))))


def qGaussian(q, t, device):
    mean = torch.zeros(q.shape, device=device)
    std = qsdWeight(q) + 0.001 * t
    return torch.distributions.Normal(mean, std)


def c1c2Func(c1, c2):
    exponent1 = torch.pow((c1[0] - 1.035) / (1.16 - 0.94) / 2.0, 10.0)
    exponent2 = torch.pow((c2[0] - 1.0) / (4.1 + 1.9) / 2.0, 10.0)
    return torch.exp(5000.0 * (exponent1 + exponent2))


def scatterPenalty(CA, aminoPDBCoords, lenList, sideChainDirecsOg, q_vals, formFactorSet, experimentalData, c1, c2, t):

    device = CA.device
    modelScatter, distSet = fastDebyeCA(CA, aminoPDBCoords, lenList, sideChainDirecsOg, q_vals, formFactorSet, c1, c2, device)
    modelScatterLog = torch.log(modelScatter)
    mask = q_vals <= 0.1
    experimentalDataLog = torch.log(experimentalData)
    noMean = torch.sum(mask * torch.ones(len(mask), device=device))
    shift = torch.sum((mask * (experimentalDataLog - modelScatterLog))) / noMean
    modelScatterLog = modelScatterLog + shift
    modelScatterNew = torch.exp(modelScatterLog)
    dif = c1c2Func(c1, c2) * (modelScatterLog - experimentalDataLog)
    qGaussians = qGaussian(q_vals, t, device)
    scatPen = qGaussians.log_prob(dif).sum()
    N = distSet.shape[0]
    mask = torch.triu(torch.ones((N, N), dtype=bool, device=device), diagonal=2)
    nonneighbs = distSet[mask]
    return scatPen, nonneighbouring_distances_gaussian_full(nonneighbs, t).sum()


def prepScatterData(file, CA, qmin, qmax, device):
    scatDat = np.loadtxt(file)
    intensities = (scatDat.T)[1]
    distSet = torch.flatten(torch.cdist(CA,CA))
    maxDist = float(1.2*torch.max(distSet))
    ns = int(2.0*maxDist*(qmax-qmin)/np.pi)
    binLims = np.linspace(qmin,qmax,ns)
    digitized = np.digitize(scatDat, binLims)
    smoothedDataLens = [len(scatDat[ digitized == i]) for i in range(1, len(binLims))]
    binPos = np.insert(np.cumsum(smoothedDataLens),0,0)
    return torch.tensor([scatDat[ digitized == i].mean() for i in range(1, len(binLims))], device=device), \
           torch.tensor([np.median(intensities[binPos[i]:(binPos[i+1]+1)]) for i in range(len(binPos)-1)], device=device)
    
    
    

