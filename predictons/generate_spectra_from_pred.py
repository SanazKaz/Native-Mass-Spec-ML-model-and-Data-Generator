import numpy as np
import torch
import time
from Mass_Spec_ML_Project.Final_Structure.Data_generation.Mass_Spec_Simulator import NativeMassSpecSimulator

def creating_single_matrices(matrices):
    """ counts how many matrices have positive values 
    should correlate with the number of non zero elements in the matrix
    and therefore the spectra"""

    new_matrices_list = []
    original_matx_num = 0
    count = 0
    for matrix_index, matrix in enumerate(matrices):
            rows, cols = matrix.shape
            original_matx_num += 1 # keep track of the original matrix ids

            for i in range(rows):
                for j in range(cols):
                    if matrix[i, j] > 0:
                        count += 1
                        new_matrix = np.zeros((rows, cols))
                        new_matrix[i, j] =  matrix[i, j]
                        new_matrices_list.append((matrix_index, new_matrix))
    print(f"positive count is:", count)
    print(f"original matrix count is:", original_matx_num)
    return new_matrices_list



def run_mass_spec_simulator(y_pred_reshaped, num_spectra=30, monomer_masses=[25644, 15000], resolution=1000, chargewidth=10, maxcharge=30, noise_level=0.000, Q=0.1, F=1, AO=1, VA=1):
    print(f'Running simulator at {time.time()}')
    simulator = NativeMassSpecSimulator(
        monomer_masses=monomer_masses,
        resolution=resolution,
        chargewidth=chargewidth,
        maxcharge=maxcharge,
        noise_level=noise_level,
        Q=Q,
        F=F,
        AO=AO,
        VA=VA
    )

    print(f'Generating {num_spectra} spectra')

    single_matrices_list = creating_single_matrices(y_pred_reshaped[0:num_spectra])
    singular_array = [matrix for _, matrix in single_matrices_list]
    singular_array = np.array(singular_array)
    print(singular_array[0].shape)
    print(singular_array[0])

    single_spectra_dataset = torch.zeros((num_spectra, 2001))
    single_interaction_matrices_dataset = torch.zeros((num_spectra, 6, 6))

    for i in range(num_spectra):
        binned_mz_range, binned_normalized_spectrum, Interaction_matrices = simulator.generate_spectrum_from_pred(singular_array[i])
        single_spectra_dataset[i] = torch.tensor(binned_normalized_spectrum)
        single_interaction_matrices_dataset[i] = torch.tensor(Interaction_matrices)

    print(f'Finished simulator at {time.time()}')
    
    return single_spectra_dataset, single_interaction_matrices_dataset

# # Example usage:
# if __name__ == '__main__':
#     # Assuming y_pred_reshaped is defined somewhere
#     y_pred_reshaped = np.random.randn(100, 6, 6)  # Example data, replace with actual data
#     single_spectra_dataset, single_interaction_matrices_dataset = run_mass_spec_simulator(y_pred_reshaped, num_spectra=30)

if __name__ == "__main__":
    y_pred_reshaped = torch.randn(100, 6, 6)  # Example data, replace with actual data
    single_spectra_dataset, single_interaction_matrices_dataset = run_mass_spec_simulator(y_pred_reshaped, num_spectra=30)
    print(f"single_spectra_dataset shape: {single_spectra_dataset.shape}")
    print(f"single_interaction_matrices_dataset shape: {single_interaction_matrices_dataset.shape}")