import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the metadata file
metadata_path = 'meta_data.npy'
metadata = np.load(metadata_path)

# Initialize lists to store sequence lengths and audio file lengths
sequence_lengths = []
audio_file_lengths = []

# Parameters for feature extraction
window_size = 1024  # Change this to your actual window size
hop_size = 512      # Change this to your actual hop size

# Iterate over each file's metadata and calculate sequence length and audio file length
for file_metadata in metadata:
    number_of_samples = file_metadata[1]
    sample_rate = file_metadata[0]
    sequence_length = (number_of_samples - window_size) // hop_size + 1
    audio_file_length = number_of_samples / sample_rate
    sequence_lengths.append(sequence_length)
    audio_file_lengths.append(audio_file_length)

# Convert the lists to NumPy arrays for easier calculations
sequence_lengths = np.array(sequence_lengths)
audio_file_lengths = np.array(audio_file_lengths)

# Calculate max, min, avg, and median for sequence lengths
max_sequence_length = np.max(sequence_lengths)
min_sequence_length = np.min(sequence_lengths)
avg_sequence_length = np.mean(sequence_lengths)
median_sequence_length = np.median(sequence_lengths)

# Calculate max, min, and total number of audio files
max_audio_length = np.max(audio_file_lengths)
min_audio_length = np.min(audio_file_lengths)
total_audio_files = len(audio_file_lengths)

print(f'Max sequence length: {max_sequence_length}')
print(f'Min sequence length: {min_sequence_length}')
print(f'Average sequence length: {avg_sequence_length}')
print(f'Median sequence length: {median_sequence_length}')
print(f'Max audio file length: {max_audio_length} seconds')
print(f'Min audio file length: {min_audio_length} seconds')
print(f'Total number of audio files: {total_audio_files}')

# Plot the distribution of audio file lengths
plt.figure(figsize=(12, 6))
sns.histplot(audio_file_lengths, bins=15, kde=True, color='skyblue')
plt.title('Distribution of Audio File Lengths')
plt.xlabel('Audio File Length (seconds)')
plt.ylabel('Count')
plt.savefig('audio_file_lengths_distribution.png')

# Plot the distribution of sequence lengths
plt.figure(figsize=(12, 6))
sns.histplot(sequence_lengths, bins=15, kde=True, color='salmon')
plt.title('Distribution of Sequence Lengths')
plt.xlabel('Sequence Length')
plt.ylabel('Count')
plt.savefig('sequence_lengths_distribution.png')

