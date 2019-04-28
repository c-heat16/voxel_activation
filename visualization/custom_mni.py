import nibabel as nib
import subprocess as sp
import pandas as pd
import numpy as np

#your_data_matrix should be 60 x 73 x 46 corresponding to the MNI space data matrix coordinates
df = np.array(pd.read_csv("try3.csv"))
nifti = nib.Nifti1Image(df,affine = np.eye(4))
print(nifti)
fname = 'save path'
nib.nifti1.save(nifti,fname)
command = ['fslcpgeom','/usr/local/fsl/MNI_brain.nii.gz',fname,'-d']
copygeom = sp.Popen(command, shell=True)
sp.Popen.wait(copygeom)