import numpy as np
import pymatgen as pmg
from pymatgen.core import Structure,Element,Lattice
import json
import sys 
import os
# from matminer.featurizers.site import SiteElementalProperty
from matminer.utils.data import MagpieData
# def read_cif(path):
#     struct=Structure.from_file(path)
#     return struct

def get_lattice_constants(structure):
    
    params = structure.lattice.abc
    angs = structure.lattice.angles
    
    consts = np.asarray((params,angs))
    
    return consts

def get_coordinates(structure):
    coords=[]
    sites = structure.sites
    
    for i in sites:
        coords.append(i.coords.tolist())
        
    coords=np.asarray(coords)
    diff=30-coords.shape[0]
    
    coords=np.pad(coords,[(0,diff),(0,0)])
    return coords


def get_frac_coordinates(structure):
    frac_coords=[]
    sites = structure.sites
    
    for i in sites:
        frac_coords.append(i.frac_coords.tolist())
    
    frac_coords=np.asarray(frac_coords)
    diff=30-frac_coords.shape[0]
    
    frac_coords=np.pad(frac_coords,[(0,diff),(0,0)])
        
    return frac_coords



# def element_matrix(structure,max_atomic_num,properties):
#     element_features = len(properties)
#     atomic_matrix= np.zeros((element_features+1,max_atomic_num))
#     elements=list(set(structure.species))
#     for i in elements:
#         atomic_num=i.Z
#         atomic_matrix[0,atomic_num]+=1
#         atomic_matrix[1:,atomic_num]=EF.featurize(i)
#     return atomic_matrix

# def element_matrix(structure,max_atomic_num,properties):
#     element_features = len(properties)
#     atomic_matrix= np.zeros((element_features+1,max_atomic_num))
#     sites=len(structure)
#     elements=list(set(structure.species))
#     for i in range(sites):
#         atomic_num=structure[i].specie.Z
#         atomic_matrix[0,atomic_num]+=1
#     for i in elements:
#         atomicnum=i.Z
#         atomic_matrix[1:,atomicnum]=EF.featurize(i)
#     return atomic_matrix

def element_matrix(cif_str,properties):

    structure=Structure.from_str(cif_str,fmt='cif')
    element_features = len(properties)
    atomic_matrix= np.zeros((element_features+1,5))
    sites=len(structure)
    elements=list(set(structure.species))
    j=0
    for i in elements:
        atomic_matrix[1:,j]=ElementFeaturizer(i,properties)
        j=j+1
    for i in range(sites):
        atomic_num=structure[i].specie.Z
        index = np.where(atomic_matrix[1]==atomic_num)
        atomic_matrix[0,index]+=1
    return atomic_matrix,condensed_props

def real_matrix(cif_str,properties, max_atomic_num=100):
    
    structure=Structure.from_str(cif_str,fmt='cif')
    cell_matrix=get_lattice_constants(structure)
    coords=get_coordinates(structure)
    basis_matrix=get_frac_coordinates(structure)
    atomicmatrix=element_matrix(structure,properties)
    matrix=np.vstack((cell_matrix,coords, basis_matrix))
    diff = atomicmatrix.shape[1]-matrix.shape[1]
    matrix=np.pad(matrix,((0,0),(0,diff)))
    realmatrix = np.vstack((atomicmatrix,matrix))
    return realmatrix



# class ElementFeaturizer():
#     def __init__(self, data_source=None, properties=('Number',)):
#         self.data_source = MagpieData()
#         self.properties = properties
#     def featurize(self,element):
#         props = []
#         for j in range(len(self.properties)):
#             props.append(self.data_source.get_elemental_property(element,self.properties[j]))
#         props=np.asarray(props)
#         return props
    
def ElementFeaturizer(element, properties, data_source=MagpieData()):
    props = []
    for j in properties:
            props.append(data_source.get_elemental_property(element,j))
    props=np.asarray(props)
    return props
    
def num_sites(cif_str):
    structure=Structure.from_str(cif_str,fmt='cif')
    coords=[]
    sites = structure.sites
    return len(sites)

def num_elements(cif_str):
    structure=Structure.from_str(cif_str,fmt='cif')
    return len(list(set(structure.species)))

    
Properties=[
 'AtomicRadius',
 'AtomicVolume',
 'AtomicWeight',
 'BoilingT',
 'BulkModulus',
 'Column',
 'CovalentRadius',
 'Density',
 'DipolePolarizability',
 'ElectronAffinity',
 'Electronegativity',
 'FirstIonizationEnergy',
 'FusionEnthalpy',
 'GSbandgap',
 'GSenergy_pa',
 'GSestBCClatcnt',
 'GSestFCClatcnt',
 'GSmagmom',
 'GSvolume_pa',
 'HeatCapacityMass',
 'HeatCapacityMolar',
 'HeatFusion',
 'HeatVaporization',
 'HHIp',
 'HHIr',
 'ICSDVolume',
 'IonizationEnergies',
 'IsAlkali',
 'IsDBlock',
 'IsFBlock',
 'IsMetal',
 'IsMetalloid',
 'IsNonmetal',
 'LogThermalConductivity',
 'MeltingT',
 'MendeleevNumber',
 'MiracleRadius',
 'MolarVolume',
 'NdUnfilled',
 'NdValence',
 'NfUnfilled',
 'NfValence',
 'NpUnfilled',
 'NpValence',
 'NsUnfilled',
 'NsValence',
 'Number',
 'NUnfilled',
 'NValence',
 'n_ws^third',
 'phi',
 'Polarizability',
 'Row',
 'SecondIonizationEnergy',
 'ShearModulus',
 'SpaceGroupNumber',
 'ThermalConductivity',
 'VdWRadius',
 'ZungerPP-r_d',
 'ZungerPP-r_p',
 'ZungerPP-r_pi',
 'ZungerPP-r_s',
 'ZungerPP-r_sigma']

# Properties=['Number',
#             'Column',
#  'AtomicRadius',
#  'AtomicVolume',
#  'AtomicWeight',
#  'BoilingT',
 
#  'CovalentRadius',
#  'Density',
#  'DipolePolarizability',
#  'Electronegativity',
#  'FirstIonizationEnergy',
#  'FusionEnthalpy',
#  ]

