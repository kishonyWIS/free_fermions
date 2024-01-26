import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt


def calculate_fourier_transform(sequence, dt, axis=-1, shift=True):
    # Perform Fourier transform along the specified axis
    fft_result = np.fft.fft(sequence, axis=axis)

    # Calculate the corresponding frequencies
    frequencies = np.fft.fftfreq(sequence.shape[axis], d=dt)

    # If shifting is requested, shift the zero frequency component to the center
    if shift:
        fft_result = np.fft.fftshift(fft_result, axes=axis)
        frequencies = np.fft.fftshift(frequencies)

    # Calculate angular frequencies
    angular_frequencies = 2 * np.pi * frequencies

    normalization_factor = np.sqrt(sequence.shape[axis] * 2 * np.pi)
    fft_result = fft_result / normalization_factor

    return angular_frequencies, fft_result


def plot_fourier_transform(sequence, dt, axis=-1, shift=True):
    # Calculate Fourier transform
    angular_frequencies, fft_result = calculate_fourier_transform(sequence, dt, axis, shift)

    # Plot the original and transformed sequences
    plt.subplot(2, 1, 1)

    if axis is not None:
        time_values = np.arange(0, sequence.shape[axis] * dt, dt)
        plt.plot(time_values, sequence, marker='o')
        plt.title(f'Original Sequence along axis {axis}')
    else:
        plt.plot(np.arange(0, len(sequence) * dt, dt), sequence, marker='o')
        plt.title('Original Sequence')

    plt.subplot(2, 1, 2)
    plt.plot(angular_frequencies, np.abs(fft_result), marker='o')
    plt.title('Fourier Transform')

    plt.show()



# Example usage:
if __name__ == '__main__':
    sample_sequence = np.cos(np.arange(1000)*np.pi*2/200)
    dt = 1/200
    plot_fourier_transform(sample_sequence, dt)
