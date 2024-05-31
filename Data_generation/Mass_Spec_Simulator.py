import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


class NativeMassSpecSimulator:
    def __init__(self, monomer_masses, resolution, chargewidth, maxcharge, noise_level, Q, F, AO, VA):
        self.monomer_masses = monomer_masses
        self.resolution = resolution
        self.chargewidth = chargewidth
        self.maxcharge = maxcharge
        self.noise_level = noise_level
        self.Q = Q
        self.F = F
        self.AO = AO
        self.VA = VA
        #self.run_counter = 0  # Initialize run counter

    def simulate_complex_spectrum(self, complex_mass):
        """ Simulate a mass spectrum for a single complex"""

        mz_range = np.arange(1, 20001)
        spectrum = np.zeros_like(mz_range, dtype=float)

        MS = complex_mass
        MA = self.Q * MS**0.76
        ME = MS + MA
        ZA = 0.0467 * ME**0.533 + self.F
        TT = np.exp(-(np.arange(self.maxcharge) - 50)**2 / self.chargewidth)
        sumT = np.sum(TT)
        WC = np.zeros(self.maxcharge)
        DE = np.zeros_like(WC)

        for charge in range(self.maxcharge):
            WC[charge] = np.exp(-(charge + 1 - ZA)**2 / self.chargewidth) / sumT
            DE[charge] = (1 - np.exp(-1620 * (9.1 * (charge + 1) / ME)**1.75)) * self.AO * self.VA if ME > 0 else 0

        WD = WC * DE

        for charge in range(1, self.maxcharge + 1):
            mz = ME / charge
            while mz >= 20000 and charge < self.maxcharge: # if the mz is too high then increase the charge - temporary fix as it is not ideal and charge should increase with mass
                charge += 13 # this is a temporary fix to increase the charge if the mz is too high - 13 is the minimum req for 250k/20k to be within range
                mz = ME / charge
                if mz >= 20000:
                    print(f"skipping this one")
                    break

            if mz <= 20000:
                lower_limit = max(1, int(mz - self.resolution / 10))
                upper_limit = min(20000, int(mz + self.resolution / 10))
                for axis in range(lower_limit, upper_limit):
                    spread = np.exp(-((axis - mz)**2) / (2 * (self.resolution / 100)**2))
                    spectrum[axis] += WD[charge - 1] * spread


        return spectrum
    
    def simulate_mass_spectrum(self, interaction_matrix):
        """ Simulate a mass spectrum with all complexes based on the interaction matrix"""
        mz_range = np.arange(1, 20001)
        combined_spectrum = np.zeros_like(mz_range, dtype=float)
        #peak_labels = []

        for i, j in np.argwhere(interaction_matrix > 0):
            stoich_A = i  # Stoichiometry of Protein A
            stoich_B = j  # Stoichiometry of Protein B
            complex_mass = stoich_A * self.monomer_masses[0] + stoich_B * self.monomer_masses[1]
            
            
            spectrum = self.simulate_complex_spectrum(complex_mass)

            scaled_spectrum = spectrum * interaction_matrix[i, j] 



            combined_spectrum += scaled_spectrum
     
        normalized_spectrum = combined_spectrum 

        return mz_range, normalized_spectrum #, peak_labels
    

    def create_interaction_matrix(self, n_proteins):
            """Create a random square interaction matrix with x by x, probability is low for 1 """
            interaction_matrix = np.random.choice([0, 1], size=(n_proteins, n_proteins), p=[4./5, 1./5]) # for binary

            return interaction_matrix


    def generate_single_spectrum(self, n_proteins):
        """Generate a single mass spectrum for n_proteins n_proteins
        main def that is called to generate the spectra for the dataset"""


        interaction_matrix = self.create_interaction_matrix(n_proteins)
        mz_range, normalized_spectrum = self.simulate_mass_spectrum(interaction_matrix)
        
        binned_normalised_spectrum = np.zeros(2001)
        bin_counts = np.zeros(2001)

        for mz, intensity in zip(mz_range, normalized_spectrum):
            bin_idx = int(mz // 10)
            if bin_idx < 2001:
                binned_normalised_spectrum[bin_idx] += intensity ## this is added the intensity to the bin index ? kind of dum
                bin_counts[bin_idx] += 1
        
        mask = bin_counts > 0
        binned_normalised_spectrum[mask] /= bin_counts[mask]

        
        noise = np.random.normal(0, self.noise_level, size=binned_normalised_spectrum.size)
        binned_normalised_spectrum += noise
        binned_mz_range = np.arange(0, 20010, 10)
        
        return binned_mz_range, binned_normalised_spectrum, interaction_matrix




    def generate_spectrum_from_pred(self, matrix):
        interaction_matrix = matrix
        mz_range, normalized_spectrum = self.simulate_mass_spectrum(interaction_matrix)
        
        binned_normalised_spectrum = np.zeros(2001)
        bin_counts = np.zeros(2001)

        for mz, intensity in zip(mz_range, normalized_spectrum):
            bin_idx = int(mz // 10)
            if bin_idx < 2001:
                binned_normalised_spectrum[bin_idx] += intensity ## this is added the intensity to the bin index ? kind of dum
                bin_counts[bin_idx] += 1
        
        mask = bin_counts > 0
        binned_normalised_spectrum[mask] /= bin_counts[mask]

        noise = np.random.normal(0, self.noise_level, size=binned_normalised_spectrum.size)
        binned_normalised_spectrum += noise

        binned_mz_range = np.arange(0, 20010, 10)
        return binned_mz_range, binned_normalised_spectrum, interaction_matrix        
    