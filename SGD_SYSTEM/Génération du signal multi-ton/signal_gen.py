"""
Ir divinkas - 2024-06-17
CTO at SYNAPTIC Lab
This script generates a multi-tone signal based on specified 
parameters such as sampling frequency, duration, and frequencies of the tones. 
The generated signal is then saved as a .npy 
file for further use in signal processing tasks.
Symiling the signal generation process, we utilize NumPy for efficient
array operations and signal synthesis. The script is designed to be flexible, allowing users to easily modify the

"""
import numpy as np
import matplotlib.pyplot as plt

# zone_one parameters
fs = 1000  # Sampling frequency in Hz
ff =  [50, 150, 300]  # Frequencies of the tones in Hz
duration = 1  # Duration of the signal in seconds