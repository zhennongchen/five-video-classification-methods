import os.path
import settings
import function_list as ff
import numpy as np
cg = settings.Experiment() 

main_folder = os.path.join(cg.oct_main_dir,'UCF101')

a = np.load(os.path.join(main_folder,'checkpoints/approach1/inception_val_top_5_acc.npy'),allow_pickle=True)
print(a.shape)
print(a[144])
print(np.max(a),np.where(a == np.max(a)))