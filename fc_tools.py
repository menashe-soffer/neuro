import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from rdm_tools_new import pierson, mysavefig

def generate_global_fc_map(data, first_bin=0, last_bin=-1):
    
    # it is assumed that the domensions are: {epoch, contact, time_bin}
    
    num_epochs, num_contacts, num_bins = data.shape
    map = np.zeros((num_contacts, num_contacts))
    for i1 in range(num_contacts):
        print(i1)
        x1 = data[:, i1, first_bin:last_bin].flatten()
        for i2 in range(num_contacts):
            x2 = data[:, i2, first_bin:last_bin].flatten()
            map[i1, i2] = pierson(x1, x2, mode='p')
    
    return map

def show_fc_map(map):
    
   fig, ax = plt.subplots(1, 1)
   sns.heatmap(map, ax=ax)
   
   #mysavefig(subfolder=1, name="functional_connectivity", fig=fig)
   
   return fig 