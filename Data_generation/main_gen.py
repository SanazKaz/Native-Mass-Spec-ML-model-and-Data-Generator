from Mass_Spec_Simulator import NativeMassSpecSimulator
import numpy as np
import torch 
import pickle
import psutil
import time

if __name__ == '__main__':
    print(f'Running simulator at {time.time()}')
    simulator = NativeMassSpecSimulator(
        monomer_masses=[25644, 15000],  
        resolution=1000,
        chargewidth=10,
        maxcharge=50, 
        noise_level=0.000,
        Q=0.1,
        F=1,
        AO=1,
        VA=1
    )

    n_proteins = 6
    num_spectra = 10000
    print(f'Generating {num_spectra} spectra for {n_proteins} proteins')
    
    
    spectra_dataset = torch.zeros((num_spectra, 2001))
    interaction_matrices_dataset = torch.zeros((num_spectra, n_proteins, n_proteins)) 


    for i in range(num_spectra):
        binned_mz_range, binned_normalized_spectrum, interaction_matrix = simulator.generate_single_spectrum(n_proteins)
        spectra_dataset[i] = torch.tensor(binned_normalized_spectrum)
        interaction_matrices_dataset[i] = torch.tensor(interaction_matrix)

with open('spectra.pkl', 'wb') as f, open('matrices.pkl', 'wb') as g:
    pickle.dump(spectra_dataset, f)
    pickle.dump(interaction_matrices_dataset, g)
        

print(f'Finished simulator at {time.time()}')