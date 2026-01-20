#!/Users/samshepherd/Work/Codes/PyEnvs/pyenv/bin/python3

from ase import Atoms, build, io, geometry
import numpy as np 
import random, re
from scipy.stats import norm, multivariate_normal
from math import *
import json

# Atomic template stuff is going to proceed this line #
rSiO = 1.62
rOH = 1.0
rAlO = 1.85
theta_min, theta_max = 115.0, 175.0
thetaW = 104.5
#######################################################

def MIC(vec,cell,icell):
    """
    Apply minimum image convention to a vector to account for
    periodic boundary conditions
    """

    ivc = np.dot(icell,vec)
    rvc = np.round(ivc)
    cvc = np.dot(cell,rvc)
    ovc = vec - cvc
    return ovc

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def genHPosition(atomO, direction, atomH=None):
    """
    Generates a random hydrogen position based on a number of constraints given a position
    """
    if atomH == None:
        r, a, Ct = rOH, random.uniform(0,360), random.uniform(theta_min, theta_max)
        a, t = radians(a), radians(180.0 - Ct)
        pos = np.array([(r*sin(a)*sin(t)),direction*(r*cos(t)),(r*cos(a)*sin(t))]).reshape([1,3])
        for ii, axis_pos in enumerate(atomO.position):
            pos[0,ii] += axis_pos
        nAtom = ['H', pos]
    else:
        """
        If we already have a H atom placed close to the O atom (maybe through the creation of a water molecule) we place the second atom at any vector which is the correct angle from the other atoms and then rotate it around that
        bond axis randomly. Repetition may be required to ensure the added atom is far enough away from some surface/atom.
        """
        Hv = (np.array(atomH.position) - atomO.position)
        Hv /= np.linalg.norm(Hv)
        if np.allclose(Hv, [1,0,0]):
            r = np.array([0,1,0])
        else:
            r = np.array([1,0,0])
        Hu = np.cross(Hv, r)
        w = np.cos(radians(thetaW))*Hv + np.sin(radians(thetaW))*Hu
        rotation_angle = radians(np.random.randint(0,360))
        Rm = rotation_matrix(Hv, rotation_angle)
        Rv = np.dot(Rm,w)
        RV = np.array([Rv[ii]+atomO.position[ii] for ii in range(np.array(atomO.position).shape[0])]).reshape(1,3)
        nAtom = ['H', RV]
    return nAtom

def genH2OPositions(atomAl, direction):
    """
    Generates a water molecule orientation above a bonded atom.
    1) Generates oxygen atom directly above the aluminium atom, this is entirely logical due to the charge of the atoms and the spacing available to the water molecules.
    2) Randomly place the first hydrogen atom anywhere within the corresponding polar available coordinates.
    3) Place the final hydrogen atom at the appropriate angle and distance based on random assignment 
    """
    pos_O = np.array([x for x in atomAl.position]).reshape(1,3)
    pos_O[0,1] -= (rAlO + np.random.uniform(-0.35, 0.35))
    O = Atoms('O', positions=pos_O)
    H2O_H1 = genHPosition(O[0], direction)
    H1 = Atoms('H', positions=H2O_H1[1])
    H2O_H2 = genHPosition(O[0],direction, atomH=H1[0])
    H2 = Atoms('H', positions=H2O_H2[1])
    return O + H1 + H2

def GetIntraOHAngles(frame, limAtoms=None):
    """
    Iterates over the atoms in the frame to gather all the necessary intralayer M-O-H angles (This will also gather additional weird groups from terminated surfaces)
    """
    frame.set_pbc(True)
    H_angles = []
    for ii, atom in enumerate(frame):
        if limAtoms is not None:
            if ii < limAtoms:
                pass
            if ii >= limAtoms:
                return H_angles
                break

        Al_idx = []
        O_idx = []
        if atom.symbol == 'H':
            distances = frame.get_distances(atom.index, indices=None, mic=True)
            Al_subs = np.argwhere(distances < 2.75)[:,0]
            O_subs = np.argwhere(distances < 2.0)[:,0]
            for jj in Al_subs:
                atomjj = frame[jj]
                if atomjj.symbol == 'Al' or atomjj.symbol == 'Mg':
                    Al_idx.append(atomjj.index)
            for jj in O_subs:
                atomjj = frame[jj]
                if atomjj.symbol == 'O':
                    O_idx.append(atomjj.index)
            OH_1 = [frame[Al_idx[0]].symbol, int(Al_idx[0]), int(O_idx[0]), atom.index]
            OH_2 = [frame[Al_idx[1]].symbol, int(Al_idx[1]), int(O_idx[0]), atom.index]
            H_angles.extend([OH_1, OH_2])

    return H_angles

def GetSheets(frame, supercell=None):
    elementDict = {'H': 1, 'O': 8, 'Mg': 12, 'Al': 13, 'Si': 14}
    """
    Atoms which are closer to the edges of the mineral and atoms which have been substituted at the apical surfaces of individual layers are more likely to get removed
    so these will have a special weight applied to them. Otherwise a weighting gradient will be applied based on the distance the atom is away from the xy centre of the 
    cell.
    """
    # Need to get how many layers of the mineral there are and cannot rely on a) cell size / unit cell or b) stoichiometry
    # This method operates by counting 'common atoms' at slices through the cell. It is janky, but can determine the number of 
    # layers based on the number of 'Al' led sheets.
    if supercell is not None:
        print(f'The use of a supercell may have an effect on the termination of the surface adjacent to the added vacuum!\nPlease check the output system created by this code...')
        expansion = []
        for ii, expand in enumerate(supercell):
            aExpand = [0,0,0]
            aExpand[ii] = expand
            expansion.append(aExpand) 
        frame = build.make_supercell(frame, expansion, wrap=True)
    
    atomic_sheets_count = {'H': 0, 'O': 0, 'Al': 0, 'Si': 0, 'Interlayer': 0}
    atomic_sheets_order_str = []
    atomic_sheets_order_idx = []
    atomic_sheet_details = {}
    for ii in range(0, int(np.round(frame.cell[2][2]))*2):
        idx = ii
        ii /= 2
        atoms_slice = []
        counts = {'H': 0, 'O': 0, 'Mg': 0, 'Al': 0, 'Si': 0}
        bin_width = [ii-0.25, ii+0.25]
        for jj, atom in enumerate(frame):
            if atom.position[2] > bin_width[0] and atom.position[2] <= bin_width[1]:
                counts[atom.symbol] += 1
                atoms_slice.append([atom.symbol, atom.index, atom.position])
        if any(counts.values()):
            atomic_sheets_count[max(counts, key=counts.get)] += 1
            atomic_sheets_order_str.append(max(counts, key=counts.get))
            atomic_sheets_order_idx.append(idx)
            atomic_sheet_details[idx] = atoms_slice
        else:
            atomic_sheets_count['Interlayer'] += 1
            atomic_sheets_order_str.append('Interlayer')
    #    atomic_sheet_details[max(counts, key=counts.get)].append(atoms_slice)    
    # This fine a grid creates false interlayer instances within the layers themselves and multiple instances between layers, this can be corrected quite easily after iteration
    sO = ' '.join(atomic_sheets_order_str)
    s1 = sO.replace('Interlayer Interlayer Interlayer Interlayer', ']--[')
    s1 = [x for x in s1.split(' ') if x != 'Interlayer']
    Layers = ''.join(s1).split(']--[')[:-1]
    output_sheet_dict = {}
    for ii, Layer in enumerate(Layers):
        if ii == 0:
            isheet = 0
        nSheets = len(re.sub( r"([A-Z])", r" \1", Layer).split())
        strSheets = re.sub( r"([A-Z])", r" \1", Layer).split()
        outDictSheetstr = [x for x in range(len(strSheets))]
        idxSheets = atomic_sheets_order_idx[isheet:isheet+nSheets]
        output_sheet_dict[ii] = {}
        # With slice idx gathered, we can now work with relative atomic sheets rather than indirect atom indexes        
        for jj, sheet in enumerate(idxSheets):
            atomNums = []
            for atom in atomic_sheet_details[sheet]:
                atomNums.append(float(elementDict[atom[0]])) 
            atomArray = np.array(atomNums).reshape(len(atomNums), 1)
            idxArray = np.array([atom[1] for atom in atomic_sheet_details[sheet]]).reshape(len(atomNums), 1)
            posArray = np.array([atom[2] for atom in atomic_sheet_details[sheet]])
            detArray = np.concatenate((atomArray, idxArray, posArray), axis=1)
            output_sheet_dict[ii][f'{strSheets[jj]}-{outDictSheetstr[jj]}'] = detArray
        isheet += len(re.sub( r"([A-Z])", r" \1", Layer).split())
    # Output dictionary is formatted so that each layer is a value in the top most dictionary, with each atom slice a sub-dictionary. These need to be numbered to ensure that each value is maintained.
                
    return output_sheet_dict, frame

def generate_box(frame: Atoms, supercell_m: list, substitute: bool, InterD: float):   
    """
    Function to generate supercells of the clay mineral, substitute the pyrophyllite structure and assign types/charges to the atoms
    """
    # reduce the frame to basic cell contents (i.e. strip away all interlayer distance)
    sys = frame.copy()

    Int = InterD # Interlayer separation based on the added molecule 
    positions = frame.positions
    positions[:,2] -= np.min(positions[:,2])
    cell_params = frame.cell.cellpar()
    cell_params[2] = np.max(positions[:,2]) + Int

    frame.set_positions(positions)
    frame.set_cell(cell_params)
    frame.wrap(eps=1e-5)

    if supercell_m is not None:
        sys = build.make_supercell(frame, P=[[supercell_m[0], 0, 0],[0, supercell_m[1], 0],[0, 0, supercell_m[2]]], wrap=True)
        #num_replacements = supercell_m[0] * supercell_m[1] * supercell_m[2] # This is very highly charged and needs to be reduced to allow for much more stable simulations
        num_replacements = ((supercell_m[0] * supercell_m[1] * supercell_m[2])/16.0)*random.randint(4,8)
        si_replacements = np.ceil(num_replacements/3)
        al_replacements = np.ceil((num_replacements/3)*2)
        print(num_replacements)
    else:
        sys = frame
        num_replacements = 1
        
    cell = sys.get_cell().T
    icell = np.linalg.inv(cell)

    if substitute is True:
        # get lists of both al and si atoms.
        al_list = [i for i in range(len(sys)) if sys[i].symbol=='Al']
        si_list = [i for i in range(len(sys)) if sys[i].symbol=='Si']
        
        # shuffle list
        random.shuffle(al_list)
        random.shuffle(si_list)
        
        # select replacements for substitutions
        si_repl = si_list[:int(si_replacements)]
        al_repl = al_list[:int(al_replacements)]
        
        for ii in si_repl:
            sys[ii].symbol = 'Al'
        for ii in al_repl:
            sys[ii].symbol = 'Mg'

        sys.info['substitutions'] = f"Si{' '.join(np.array(si_repl).astype(str))} Al {' '.join(np.array(al_repl).astype(str))}"
        with open('Substitutions_info.txt', 'w+') as handle:
            for sub in (si_repl + al_repl):
                handle.writelines(f'{sub}\n')
        handle.close()
        io.write('substitute.pdb', sys, format='proteindatabank')

    # Generate a list of hydroxyl groups and oxygens 
    all_oh, oh_oxygens, h2o = [], [], []
    for atomii in sys:
        if atomii.symbol == 'O':
            n_H = []
            o_h = False 
            for atomjj in sys:
                if atomjj.symbol == 'H':
                    dr = np.linalg.norm(MIC(atomii.position-atomjj.position, cell, icell))
                    if dr < 1.2:
                        o_h = True 
                        n_H.append(atomjj.index)
                        h = atomjj 
            if o_h == True and len(n_H) < 2:
                all_oh.append([atomii.index,h.index])
                oh_oxygens.append(atomii.index)
            elif o_h == True and len(n_H) == 2:
                h2o.append([atomii.index, n_H[0], n_H[1]])
                

    n_mnt = len(sys)
    charge_dict = {1:0.425, 2:-0.95, 3:-1.05, 4:1.575, 5:2.1, 6:-0.8476, 7:0.4238, 8:1.36, 10:1.575, 11:-1.1808, 12:-1.1688, 13:-1.2996, 14:-1.0808, 15:(-1.2996-0.1298)}
    types = [] 
    charges = []

    ### Hydrogen block ##
    for atomii in sys:
        atomType = -1
        if atomii.symbol == 'H':
            if atomii.index in h2o:
                atomType = 6
            else:
                atomType = 1 
    ## Silicon block ##
        elif atomii.symbol == 'Si':
            atomType = 5 
    ## Magnesium block ##
        elif atomii.symbol == 'Mg':
            atomType = 8 
    ## Aluminium block ##
        elif atomii.symbol == 'Al':
            if substitute == True and atomii.index in si_repl:
                    atomType = 10 
            else:
                atomType = 4
    ## Oxygen block ##
        elif atomii.symbol == 'O':
            if atomii.index in h2o[:]: # Terminal surface water oxygen
                atomType = 7
            elif atomii.index in oh_oxygens: # hydroxyl oxygen
                mg_defect = False 
                for atomjj in sys:
                    if atomjj.symbol == 'Mg':
                        if np.linalg.norm(MIC(atomii.position-atomjj.position,cell,icell)) < 2:
                            if mg_defect == False:
                                mg_defect = True 
                    if mg_defect == True:
                        atomType = 14 
                    else:
                        atomType = 2
            else: # it must then be a bridging oxygen
                nnTSub, nnOSub = 0, 0
                if substitute == True:
                    for jj in si_repl:
                        if np.linalg.norm(MIC(atomii.position-sys[jj].position,cell,icell)) < 2:
                            nnTSub += 1
                    for jj in al_repl:
                        if np.linalg.norm(MIC(atomii.position-sys[jj].position,cell,icell)) < 2:
                            nnOSub += 1
                if nnTSub + nnOSub == 0: # No subs, just bridging
                    atomType = 3 

                if nnTSub == 1 and nnOSub == 0:
                    atomType = 12 # Tetrahedral substitution 
                if nnTSub == 0 and nnOSub == 1:
                    atomType = 11 # Octahedral substitution 
                if (nnTSub > 0 or nnOSub > 0) and (nnTSub + nnOSub == 2): # editted slightly as some substitutions can be adjacent to one another
                    atomType = 13
                if (nnTSub > 0 or nnOSub > 0) and (nnTSub + nnOSub > 2): # higher coordinated atoms are possible in this arrangement
                    atomType = 15
        if atomType == -1:
            print(f'Atom {atomii.index} is causing problems')
        types.append(atomType)

    total_charge = 0
    for ii in range(len(sys)):
        charges.append(charge_dict[types[ii]])
        total_charge += charge_dict[types[ii]]


    print((-3*(supercell_m[0]*supercell_m[1]*supercell_m[2]) - np.round(total_charge,4))/(-3*(supercell_m[0]*supercell_m[1]*supercell_m[2])), -3*(supercell_m[0]*supercell_m[1]*supercell_m[2]),  np.round(total_charge, 5))
    return sys, types, charges

def finishChargesandTypes(system, details):
    """
    VERY SPECIFIC function that allows for the charge to be rounded to a suitable number and print that number so that specifics can be considered for the addition of the Quartenary Amines
    Also finishes the hydroxyl bond types section with the added atoms from the termination of the surface.
    """
    charge = 0
    for val in details['Charges']:
        charge += np.round(val, 4)
    
    left_over = np.round(charge, 4)
    additional_charge = np.round((left_over % 4), 4)
    print(-additional_charge)
    
    # charge is best spread over atoms near substitutions, so will depend on how many of these atoms we can use to spread the charge over
    count11, count12, count13, count14 = 0, 0, 0, 0
    for sub in details['Types']:
        if sub == 11:
            count11 += 1
        elif sub == 12:
            count12 += 1
        elif sub == 13:
            count13 += 1
        elif sub == 14:
            count14 += 1
    
    suitableNum = 80
    print(count11, count12, count13, count14)
    cPa = -additional_charge/suitableNum
    
    
    suitableAtoms = []
    for atom in system:
        atype = details['Types'][int(atom.index)]
        if atype == 14:
            suitableAtoms.append(str(atom.index))
        if atom.symbol == 'H' and atom.index < 10240:
            dists = system.get_distances(atom.index, indices=None, mic=True)
            nearest_o = np.argwhere(dists[:] < 1.1)
            if nearest_o[1][0] != atom.index:
                details['Bonds'].append([nearest_o[1][0], atom.index, 'hydroxyl'])
            elif nearest_o[1][0] == atom.index:
                details['Bonds'].append([nearest_o[0][0], atom.index, 'hydroxyl'])
    
    tmp_copy = details['Charges']
    chosenAtoms = random.sample(suitableAtoms, suitableNum)
    for atom in system:
        if str(atom.index) in chosenAtoms:
            details['Charges'][int(atom.index)] += cPa
    print(np.round(np.sum(details['Charges']), 4))

def printerGeneral(details):
    """
    Simple function that writes a more detailed list of all the necessary types, charges, bonds and angles
    """
    with open('SystemDetails.txt', 'w+') as handle:
        handle.writelines(f'Types\n\n')
        for ii, Type in enumerate(details['Types']):
            handle.writelines(f'{ii+1} {Type}\n')

        handle.writelines(f'Charges\n\n')
        for ii, Charge in enumerate(details['Charges']):
            handle.writelines(f'{ii+1} {Charge}\n')

        handle.writelines(f'Bonds\n\n')
        for ii, Bond in enumerate(details['Bonds']):
            handle.writelines(f'{ii+1} {Bond[0]} {Bond[1]} {Bond[2]}\n')

        handle.writelines(f'\nAngles\n\n')
        for ii, Angle in enumerate(details['Angles']):
            if len(Angle) > 3:
                handle.writelines(f'{ii+1} {Angle[0]} {Angle[1]} {Angle[2]} {Angle[3]}\n')
            else:
                handle.writelines(f'{ii+1} H2O {Angle[0]} {Angle[1]} {Angle[2]}\n')
    return

def printerLammps(system, details, intraOHdetails):
    """
    Lammps printer function
    """
    with open('Clay.data', 'w+') as handle:
        handle.writelines(
    f"""LAMMPS Description

            {len(system)} atoms
            {len(details['Bonds'])} bonds
            {len(details['Angles'])+len(intraOHdetails)} angles
            0 dihedrals
            0 impropers

            15 atom types
            2 bond types
            6 angle types
            0 dihedral types
            0 improper types

        0.0 {system.cell[0][0]} xlo xhi
        0.0 {system.cell[1][1]} ylo yhi 
        0.0 {system.cell[2][2]} zlo zhi 
        {system.cell[1][0]} {system.cell[2][0]} {system.cell[2][1]} xy xz yz

    Masses 

    1  1.008  # Hydroxyl Hydrogen
    2  15.999 # Hydroxyl Oxygen
    3  15.999 # Bridging Oxygen 
    4  26.982 # Octahedral Aluminium
    5  28.085 # Tetrahedral Silicon
    6  15.999 # Water Oxygen
    7  1.008  # Water Hydrogen
    8  24.305 # Magnesium Defect
    9  22.990 # Sodium Ion -> Not used
    10 26.982 # Tetrahedral Aluminium 
    11 15.999 # Bridging Oxygen Os 
    12 15.999 # Bridging Oxygen Ts
    13 15.999 # Bridging Oxygen Ds
    14 15.999 # Hydroxyl Oxygen s
    15 15.999 # Bridging Oxygen Ms (Higher Substitutions)

    Atoms # full

""")
        for ii, atom in enumerate(system):
            # this is idx, molNum (1), type, charge, then pos x y z
            handle.writelines(f'{ii+1} 1 {details['Types'][ii]} {details['Charges'][ii]} {system[ii].position[0]} {system[ii].position[1]} {system[ii].position[2]}\n')

        handle.writelines("""\nBonds\n\n""")

        for ii, bond in enumerate(details['Bonds']):
            # this is bondNumber, bondType, atom1 atom2
            if details['Bonds'][ii][2] == 'water':
                btype = 2
            elif details['Bonds'][ii][2] == 'hydroxyl':
                btype = 1
            handle.writelines(f'{ii+1} {btype} {details['Bonds'][ii][0]+1} {details['Bonds'][ii][1]+1}\n')

        handle.writelines("""\nAngles\n\n""")

        angles = 0
        for jj, Iangle in enumerate(intraOHdetails):
            if Iangle[0] == 'Al':
                aType = 3
            if Iangle[0] == 'Mg':
                aType = 5
            angleLine = f'{jj+1} {aType} {Iangle[1]+1} {Iangle[2]+1} {Iangle[3]+1}\n'
            handle.writelines(angleLine)
            angles += 1

        for ii, angle in enumerate(details['Angles']):
            # this is angleNumber, angleType, atom2, atom1, atom3
            if len(details['Angles'][ii]) == 3:
                aType = 1
                handle.writelines(f'{angles+ii+1} {aType} {details['Angles'][ii][0]+1} {details['Angles'][ii][1]+1} {details['Angles'][ii][2]+1}\n')
            elif len(details['Angles'][ii]) == 4:
                if details['Angles'][ii][0] == 'Al':
                    aType = 2
                    handle.writelines(f'{angles+ii+1} {aType} {details['Angles'][ii][1]+1} {details['Angles'][ii][2]+1} {details['Angles'][ii][3]+1}\n')
                elif details['Angles'][ii][0] == 'Mg':
                    aType = 4
                    handle.writelines(f'{angles+ii+1} {aType} {details['Angles'][ii][1]+1} {details['Angles'][ii][2]+1} {details['Angles'][ii][3]+1}\n')
                elif details['Angles'][ii][0] == 'Si':
                    aType = 6
                    handle.writelines(f'{angles+ii+1} {aType} {details['Angles'][ii][1]+1} {details['Angles'][ii][2]+1} {details['Angles'][ii][3]+1}\n')
    return