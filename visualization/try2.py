import numpy as np
import pandas as pd
import scipy.io
from scipy.io import loadmat

'''
Read in voxels and minus the column # by 1 to get the correct index
hand picked the first 500
'''

variable = pd.DataFrame(np.loadtxt("highest_corr_voxels_20190410-214719.txt"))
variable.reset_index(drop=True, inplace=True)
variable = variable - 1
voxels = variable[0:500]
voxels = voxels.applymap(int)

#print(len(voxels))
# 500
#print(voxels[0][0])
# 155

'''
Read in the common coordinates
Have the option of keeping index num or not in the commented line
'''

df = pd.read_csv("common_voxel_col_to_coord.csv", header=0, parse_dates=False)
result = pd.DataFrame(df)
#temp = result.loc[result['voxel_col'] == voxels[0][0]]

'''
A forloop to match voxels to df 
'''

vox_df = pd.DataFrame(columns=['x','y','z'])

for i in voxels[0]:
    temp = result.loc[result['voxel_col'] == i]
    vox_df = vox_df.append(temp)
vox_df = vox_df.applymap(int)
#print(vox_df)

vox_df.to_csv("/Users/andreawan/onedrive/Spring_19/DS_340W/visualization/voxel.csv")

'''
#mat_contents = np.loadtxt('demo.m')
mat = scipy.io.loadmat('demo.m', struct_as_record=True)
print(mat)
'''

