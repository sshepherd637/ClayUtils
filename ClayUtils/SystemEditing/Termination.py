#!/Users/samshepherd/Work/Codes/PyEnvs/pyenv/bin/python3

from ase import Atoms, build, io, geometry
import numpy as np 
import random, re
from scipy.stats import norm, multivariate_normal
from math import *
import json

# Internal import statements
from Utils import GeneralUtils as GU

# Atomic template stuff is going to proceed this line #
rSiO = 1.62
rOH = 1.0
rAlO = 1.85
theta_min, theta_max = 115.0, 175.0
thetaW = 104.5
#######################################################

def TerminateSystem(CL, frame, axis=1, vac=50, ATs=None, ACs=None):
    """
    Function to terminate a system provided with an axis, adds vacuum too throughout the function and then populate the terminated surface based on the nature of the elements found there.
    ASE won't allow for edits to the add_vacuum method, so will operate by altering the cell instead. This function has been editted to take as inputs details on individual clay layers as
    this allows for a much tighter control of the termination of the system without worrying too much about cell vectors messing with the distances to the vacuum surface. 
    """
    # Generate cell systems, holding lists, and other arrays prior to worrying about any of the clay layers which we can then place the ClayLayer atoms into, and work as required from there. 
    SystemDetails = {'Types': [], 'Charges': [], 'Bonds': [], 'Angles': []}
    vacSystem = frame.copy()
    cell = vacSystem.cell.cellpar()
    cell[axis] += vac

    tempSystem = Atoms()
    z_vals = []
    edge_lengths = []
    LayerCounter = 0

    for k, v in CL.items():
        if v:
            z_list = []
            indices = []
            LayerCounter += 1
            LayerSys = Atoms()
            for sk in CL[k].keys():
                tempSys = CL[k][sk]
                idx = tempSys[:,1]
                if ATs != None:
                    atomtypes = [ATs[int(x)] for x in idx]
                    SystemDetails['Types'].extend(atomtypes)
                if ACs != None:
                    atomcharges = [ACs[int(x)] for x in idx]
                    SystemDetails['Charges'].extend(atomcharges)
                pos = tempSys[:,2:]
                pos[:,axis] += vac/2
                Ats = Atoms([int(x) for x in tempSys[:,0]], positions=pos)
                LayerSys += Ats
                z_list.extend(CL[k][sk][:,4])
            z_vals.append(np.mean(np.array(z_list)))    
            eF = np.max(LayerSys.positions[:,axis])
            eN = np.min(LayerSys.positions[:,axis])
            print(f'Layer {LayerCounter}: Near Edge {eN}, Far Edge {eF}')
            edges = (eN, eF)
            edge_lengths.append(edges)
            LayerSys.set_pbc(True)
            tempSystem += LayerSys
    
    tempSystem.set_cell(cell)
    tempSystem.set_pbc(True)
    tempSystem.center(axis=axis)

    # Get Interlayer z values to differentiate layers more accurately 
    IntVals = []
    for ii in range(len(z_vals)-1):
        lay1, lay2 = z_vals[ii], z_vals[ii+1]
        IntVals.append(np.mean([lay1,lay2]))

    new_atoms = Atoms()
    print(len(tempSystem))
    for ii, atom in enumerate(tempSystem):
        if ii % 2500 == 0:
            print(f'Terminator has surpassed atom...{ii}')
        # get Layer of the atom based on the Z profile unless system is mono-layer
        if IntVals == []:
            atLayer = 0
        else:
            if atom.position[2] < IntVals[0]:
                atLayer = 0
            elif atom.position[2] > IntVals[0] and atom.position[2] < IntVals[1]:
                atLayer = 1
            elif atom.position[2] > IntVals[1] and atom.position[2] < IntVals[2]:
                atLayer = 2
            elif atom.position[2] > IntVals[2]:
                atLayer = 3

        edgeN, edgeF = edge_lengths[(atLayer)][0], edge_lengths[(atLayer)][1]
        isFarEdge = False
        vector = -1
        if atom.position[axis] > tempSystem.cell.cellpar()[axis]/2:
            isFarEdge = True 
            vector = 1

        distances = tempSystem.get_distances(atom.index, indices=None, mic=True)
        varA, varB = 1.0, 1.5
        if atom.symbol in ['Si', 'Al', 'Mg'] and ((atom.position[axis] >= edgeN - varA and atom.position[axis] <= edgeN + varA) or (atom.position[axis] <= edgeF + varB and atom.position[axis] >= edgeF - varB)):
            if atom.symbol == 'Si':
                O_planeVecs = tempSystem.get_distances(atom.index, indices=[np.argsort(distances[:])[1:4]], mic=True, vector=True)
                O_plane = np.array([(O_planeVecs[1]-O_planeVecs[0]), (O_planeVecs[2]-O_planeVecs[0])])
                O_norm = np.cross(O_plane[0], O_plane[1])*-1 # Normal is in the direction of the planes
                O_pos = (O_norm / np.linalg.norm(O_norm))*rSiO + atom.position
                O_atom = Atoms('O', positions=(np.array([O_pos[0], O_pos[1], O_pos[2]]).reshape([1,3])))
                H = GU.genHPosition(O_atom[0], direction=vector)
                H_atom = Atoms(H[0], positions=H[1])
                OH = O_atom + H_atom
                new_atoms += OH
                SystemDetails['Types'].extend([2,1])
                SystemDetails['Charges'].extend([-0.95, 0.4250])
                SystemDetails['Bonds'].append([((len(tempSystem)-1) + (len(new_atoms)-1)), ((len(tempSystem)-1) + (len(new_atoms))), 'hydroxyl'])
                SystemDetails['Angles'].append([atom.symbol, atom.index, ((len(tempSystem)-1) + (len(new_atoms)-1)), ((len(tempSystem)-1) + (len(new_atoms)))])
            elif atom.symbol == 'Al' or 'Mg':
                if distances[distances[:] < 2.0].shape[0] > 4:
                    passTest = False
                    rdm_counter = 0
                    while passTest == False:
                        H2O = GU.genH2OPositions(atom,direction=vector)
                        tmp = Atoms(atom.symbol, positions=np.array(atom.position).reshape(1,3))
                        tmp += H2O
                        tmp_pos = tmp.get_all_distances()[0,:]
                        if len(tmp_pos[tmp_pos < rAlO]) == 1:
                            passTest = True
                            new_atoms += H2O
                            SystemDetails['Types'].extend([6,7,7])
                            SystemDetails['Charges'].extend([-0.8476, 0.4238, 0.4238])
                            SystemDetails['Bonds'].append([((len(tempSystem)-1) + (len(new_atoms)-2)), ((len(tempSystem)-1) + (len(new_atoms)-1)), 'water'])
                            SystemDetails['Bonds'].append([((len(tempSystem)-1) + (len(new_atoms)-2)), ((len(tempSystem)-1) + (len(new_atoms))), 'water'])
                            SystemDetails['Angles'].append([((len(tempSystem)-1) + (len(new_atoms)-1)), ((len(tempSystem)-1) + (len(new_atoms)-2)), ((len(tempSystem)-1) + (len(new_atoms)))])                      
                        else:
                            rdm_counter += 1
                elif distances[distances[:] < 2.0].shape[0] <= 4:
                    O_planeVecs = tempSystem.get_distances(atom.index, indices=[np.argsort(distances[:])[1:4]], mic=True, vector=True)
                    O_plane = np.array([(O_planeVecs[1]-O_planeVecs[0]), (O_planeVecs[2]-O_planeVecs[0])])
                    O_norm = np.cross(O_plane[0], O_plane[1])*-1 # Normal is in the direction of the planes
                    O_pos = (O_norm / np.linalg.norm(O_norm))*rSiO + atom.position
                    O_atom = Atoms('O', positions=(np.array([O_pos[0], O_pos[1], O_pos[2]]).reshape([1,3])))
                    H = GU.genHPosition(O_atom[0], direction=vector)
                    H_atom = Atoms(H[0], positions=H[1])
                    OH = O_atom + H_atom 
                    new_atoms += OH
                    SystemDetails['Types'].extend([2,1])
                    SystemDetails['Charges'].extend([-0.95, 0.4250])
                    SystemDetails['Bonds'].append([((len(tempSystem)-1) + (len(new_atoms)-1)), ((len(tempSystem)-1) + (len(new_atoms))), 'hydroxyl'])
                    SystemDetails['Angles'].append([atom.symbol, int(atom.index), ((len(tempSystem)-1) + (len(new_atoms)-1)), ((len(tempSystem)-1) + (len(new_atoms)))])
        elif atom.symbol == 'O' and ((distances[:] < 2).sum() == 2) and ((atom.position[axis] >= edgeN - varA and atom.position[axis] <= edgeN + varA) or (atom.position[axis] <= edgeF + varB and atom.position[axis] >= edgeF - varB)):
            H = GU.genHPosition(atom, direction=vector)
            H_atom = Atoms(H[0], positions=H[1])
            new_atoms += H_atom
            nearest_m = np.argwhere(distances[:] < 2)
            if atom.index == nearest_m[0][0]:
                nearM = nearest_m[1][0]
            else:
                nearM = nearest_m[0][0]
            SystemDetails['Types'].extend([1])
            SystemDetails['Charges'].extend([0.4250])
            SystemDetails['Bonds'].append([atom.index, ((len(tempSystem)-1) + (len(new_atoms))), 'hydroxyl'])
            SystemDetails['Angles'].append([tempSystem[nearM].symbol, int(nearM), int(atom.index), ((len(tempSystem)-1) + (len(new_atoms)))]) # <- problem with the typing happening here still

    tempSystem += new_atoms
    return tempSystem, SystemDetails

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

pdb_viewer = {'occupancy': 1,
              'bfactor': 0,
              'residuenames': 'MOL',
              'atomtypes': 'STR',
              'residuenumbers': 1}

if __name__ == "__main__":
    init = io.read('TMA-Mont.pdb')
    ClayLayers, sys = GU.GetSheets(init, supercell=[1,1,1])
    #Weights = WeighAtoms(ClayLayers, sys, 5)
    vac = TerminateSystem(ClayLayers, sys, vac=100)
    for Arr in ['occupancy', 'bfactor', 'residuenames', 'atomtypes', 'residuenumbers']:
        try:
            for ii, atom in enumerate(vac):
                vac.arrays[Arr][ii] = pdb_viewer[Arr]
                if Arr == 'atomtypes':
                    vac.arrays[Arr][ii] = atom.symbol
        except:
            print(f"No array with name {Arr} found")
    positions = vac.positions
    positions[:,2] += 1.5 # slight offset to avoid atoms phasing into the next cell
    vac.set_positions(positions)
    io.write('TerminatedSurface.pdb', vac)