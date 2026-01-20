from ase import Atoms, build, io, geometry
import numpy as np 
import random, re
from scipy.stats import norm, multivariate_normal
from math import *
import json

# Internal import statements
from Misc import GeneralUtils as GU

# Atomic template stuff is going to proceed this line #
rSiO = 1.62
rOH = 1.0
rAlO = 1.85
theta_min, theta_max = 115.0, 175.0
thetaW = 104.5
#######################################################

def WeighAtoms(LayerDict, frame, baseweight):
    """
    Function to associated weights with the atoms depending on their position and identity
    Ins  | -> LayerDict: From GetSheets function, formatted dictionary of clay layers with atomic positions and elements. 
         | -> frame: frame is loaded purely to get the cell information.
         | -> baseweight:  Normalisation factor for each applied weight i.e. weight/baseweight = actualweight
    Outs | -> WeightsMatrix: Array of atom indices and weights to aid in their sampling as removed atoms.

    Weights are applied using the following logic:
    1. Any substitutions in a tetrahedral layer (Mg/Al for Si) are VASTLY more likely to be leeched out, as they are in direct contact with the surfactant region, they get a higher weighting
    2. As it's assumed that the edges of the mineral will be water facing, a weighting is applied along the x axis so that the further from the center of the clay, the more likely the leeching is

    This function will therefore operate by iterating over all the layers of the clay in turn as ordered within the dictionary, it will first act only on the Al/Si sheets and weigh these based on their z and xy values. 
    These weights will be drawn from normal distributions: The Z profile distribution will be bimodal with maxima at both average Si z positions of that layer. Sigma is carefully chosen to ensure sufficient weighting to 
    the central Al layer atoms. The xy profile is also a normal distribution but 3 dimensional to favour y edges and x edges appropriately.
    """
    # Create ouput matrix shape 
    Weights = np.zeros([len(frame), 2])
    Weights[:,0] = [x.index for x in frame]
    
    n = 10000

    # XY probability is given without care of layer composition, and can be done out prior to entering any loops.
    xy_sigma = 100 # Large value means palpable deviations in probability based on xy distance from center
    xy_cov = np.array([[xy_sigma,0],[0,xy_sigma]])
    xy_middle = np.array([(np.max(frame.positions[:,0])-np.min(frame.positions[:,0]))/2, (np.max(frame.positions[:,1])-np.min(frame.positions[:,1]))/2])
    xy_normal = multivariate_normal(xy_middle, xy_cov)
    x = np.linspace(np.min(frame.positions[:,0])-25,np.max(frame.positions[:,0]+25), n)
    y = np.linspace(np.min(frame.positions[:,1])-25,np.max(frame.positions[:,1]+25), n)
    X,Y = np.meshgrid(x, y)
    XY_pos = np.empty(X.shape + (2,))
    XY_pos[:, :, 0] = X 
    XY_pos[:, :, 1] = Y
    XY_prob = xy_normal.pdf(XY_pos)
    XY_prob = np.max(XY_prob) - XY_prob
    XY_pdf = np.zeros([XY_prob.shape[0]+1, XY_prob.shape[1]+1])
    XY_pdf[1:,1:] = XY_prob
    XY_pdf[1:,0], XY_pdf[0,1:] = x,y

    # Iterate over the z profile using the layers in the LayerDict - only care for the Si and Al layers.
    for l in LayerDict.keys():
        layer = LayerDict[l]
        surface_vals = []
        for s in layer:
            sheet = layer[s]
            if 'Si' in s:
                # Distribution is created based off of the z profiles of the layers so access top and bottom z values
                surface_vals.append([np.mean(sheet[:,-1]), np.min(sheet[:,-1]), np.max(sheet[:,-1])])
            elif 'Al' in s:
                surface_vals.append([np.mean(sheet[:,-1]), np.min(sheet[:,-1]), np.max(sheet[:,-1])])

        z1, z2 = np.random.normal(surface_vals[0][0], 1.75, n), np.random.normal(surface_vals[2][2], 1.75, n)
        z = np.concatenate([z1,z2])
        z_eval_points = np.linspace(np.min(z)-1, np.max(z)+1, n)
        bimodal_Z_pdf = norm.pdf(z_eval_points, loc=surface_vals[0][0], scale=1.75) + norm.pdf(z_eval_points, loc=surface_vals[2][2], scale=1.75)
        Z_pdf = np.concatenate([z_eval_points.reshape(z_eval_points.shape[0],1), bimodal_Z_pdf.reshape(bimodal_Z_pdf.shape[0],1)],1)

        for s in layer:
            sheet = layer[s]
            for atom in sheet:
                idx = atom[1]
                if atom[0] not in [1.00, 8.00, 14.00]:
                    weight_XY = XY_pdf[((np.abs(XY_pdf[1:,0] - atom[2])).argmin()), ((np.abs(XY_pdf[0,1:] - atom[3])).argmin())]
                    weight_Z = Z_pdf[((np.abs(Z_pdf[:,0] - atom[4])).argmin()),1]
                else: 
                    weight_XY, weight_Z = 0, 0
                atom_weight = (weight_XY + weight_Z)*baseweight
                if atom_weight > 1.0:
                    atom_weight = 1.0
                Weights[int(idx),1] = atom_weight
                
    return Weights