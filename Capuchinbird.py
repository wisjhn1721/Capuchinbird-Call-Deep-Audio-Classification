# Name: John Wise
# Date: 9/2/22
# In this project we will create a build a Deep Audio Classification model with Tensorflow and Python. This project was
# built around the HP Unlocked challenge. The Challenge is to build a Machine Learning model and code to count the
# number of Capuchinbird calls within a given clip. The Data is split into Training and Testing Data. For Training Data
# we have provided enough clips to get a decent model but you can also find, parse, augment and use additional audio
# clips to improve your model performance.
#
# Data: https://www.kaggle.com/datasets/kenjee/z-by-hp-unlocked-challenge-3-signal-processing?resource=download
#
# Tutorial: https://www.youtube.com/watch?v=ZLIPkmmDJAc

import os
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from itertools import groupby
import csv


# define paths to files
CAPUCHIN_FILE = os.path.join('data', 'Parsed_Capuchinbird_Clips', 'XC3776-3.wav')
NOT_CAPUCHIN_FILE = os.path.join('data', 'Parsed_Not_Capuchinbird_Clips', 'afternoon-birds-song-in-forest-0.wav')


# Loads a wav file, converts Stereo->Mono, and then converts it into 16Hz sample rate using interpolation
def load_wav_16k_mono(filename):
    # Load encoded wav file to a byte-encoded string
    file_contents = tf.io.read_file(filename)
    # Decode wav (tensors by channels)
    # This produces the mono data from the stereo input data. Probably calculated with (Left[i] + Right[i])/ 2
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Removes trailing axis
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Goes from 44100Hz to 16000hz. Because of Nyquist, the max frequency we can represent in our signal is 8k hz.
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

# Lets plot the wave to get an understanding of what our data looks like.
wave = load_wav_16k_mono(CAPUCHIN_FILE)
nwave = load_wav_16k_mono(NOT_CAPUCHIN_FILE)

plt.plot(wave)
plt.plot(nwave)
plt.show()

# Create TensorFlow Datasets
# This will return a list of strings for all files matching the pattern in the string
pos = tf.data.Dataset.list_files('data/Parsed_Capuchinbird_Clips/*.wav')
neg = tf.data.Dataset.list_files('data/Parsed_Not_Capuchinbird_Clips/*.wav')

# Right now we just have the file but we need to give each file a label in the dataset we just created.
# tf.ones(len(pos)) provides a tensor of ones with a length equal to the number of files in the positive directory
# tf.zeros(len(pos)) provides a tensor of zeros
# positives will be a list of tuples like (file path, label) ex: (data/Parsed_Capuchinbird_Clips/XC3776-0.wav, 1.0)
positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))
negatives = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg)))))
# Now we want to join all of these together and shuffle so that we are dealing with one variable to use for training.
data = positives.concatenate(negatives)
data = data.shuffle(10000)
print(data.as_numpy_iterator().next())


# Exploratory Data Analysis
# We want to make sure we are capturing all or most of the capuchinbird call so lets find the average size.
# Calculate the wave cycle length
lengths = []
for file in os.listdir(os.path.join('data', 'Parsed_Capuchinbird_Clips')):
    tensor_wave = load_wav_16k_mono(os.path.join('data', 'Parsed_Capuchinbird_Clips', file))
    lengths.append(len(tensor_wave))

# Calculate Mean, Min, and Max
# mean/sample_rate will give us the average bird call length (3.38 seconds in this case)
Mean = tf.math.reduce_mean(lengths)
Min = tf.math.reduce_min(lengths)
Max = tf.math.reduce_max(lengths)
print(f"Average Length: {Mean/16000} seconds")
print(f"Minimum Length: { Min/16000} seconds")
print(f"Maximum Length: { Max/16000} seconds")

# Now lets build a preprocessing function to convert our data to a spectrogram
# This function will be part of our data processing pipeline so we will want to bring in the label and return it.
def preprocess(file_path, label):
    wav = load_wav_16k_mono(file_path)  # down sample and flatten our input audio file
    wav = wav[:48000]  # Shorten it to a length equal to Mean so that all files are normalized to the same length
    # not all files are equal to or larger than the mean so for those files smaller we will create a zero padding
    zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
    # we override our wav variable to the zero padding concatenated with the wav
    wav = tf.concat([zero_padding, wav], 0)

    # to create our spectrogram we will use the short time fourier transform
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    # Shape is 1491x257 before this line. Running the lin below will add our channels dimension for our CNN model.
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label


# Now lets test our our preprocess function and visualize the output.
filepath, label = positives.shuffle(buffer_size=10000).as_numpy_iterator().next()
spectrogram, label = preprocess(filepath, label)
plt.figure(figsize=(40, 10))
plt.imshow(tf.transpose(spectrogram)[0])
plt.show()


# Create Training and Testing Partitions
# Create a Tensorflow Data Pipeline
data = data.map(preprocess)
data = data.cache()
data = data.shuffle(buffer_size=10000)
data = data.batch(16)  # train on 16 samples at a time
data = data.prefetch(8)  # Reduce CPU bottle necking by prefetching

# Split into Training and Testing Partitions
# 70% for training and 30% for testing
num_training_samples = round(len(data) * 0.7)
num_testing_samples = round(len(data) * 0.3)
train = data.take(num_training_samples)
test = data.skip(num_training_samples).take(num_testing_samples)


# Lets Build Deep Learning Model!
# Build Sequential Model, Compile and View Summary
model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(1491, 257, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# More info on Binary Cross Entropy: https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a
model.compile('Adam', loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

model.summary()

# Fit Model, View Loss and KPI Plots
hist = model.fit(train, epochs=10, validation_data=test)

plt.title('Loss')
plt.plot(hist.history['loss'], 'r')
plt.plot(hist.history['val_loss'], 'b')
plt.show()

plt.title('Precision')
plt.plot(hist.history['precision'], 'r')
plt.plot(hist.history['val_precision'], 'b')
plt.show()

plt.title('Recall')
plt.plot(hist.history['recall'], 'r')
plt.plot(hist.history['val_recall'], 'b')
plt.show()

# Now lets make a prediction on a single clip
X_test, y_test = test.as_numpy_iterator().next()
yprime = model.predict(X_test)

# Convert these logitsd (confidence metrics) to classes
yprime = [1 if prediction > 0.95 else 0 for prediction in yprime]


# Now we will use our trained model to detect capuchinbird calls within the forest clips. We will do so by sliding (similar
# to convolving) over the forest samples and predicting whether the clip that we are looking at contains a capuchinbird
# call or not.
# Our forest clips are mp3s so we will need to create a function which will reduce them to mono and downsample to 16k hz

def load_mp3_16k_mono(filename):
    # Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio.
    res = tfio.audio.AudioIOTensor(filename)
    # Convert to tensor and combine channels
    tensor = res.to_tensor()
    tensor = tf.math.reduce_sum(tensor, axis=1) / 2
    # Extract sample rate and cast
    sample_rate = res.rate
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Resample to 16 kHz
    wav = tfio.audio.resample(tensor, rate_in=sample_rate, rate_out=16000)
    return wav


# We will use this function to convert our audio slices into spectrograms
def preprocess_mp3(sample, index):
    sample = sample[0]
    zero_padding = tf.zeros([48000] - tf.shape(sample), dtype=tf.float32)
    wav = tf.concat([zero_padding, sample],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram


# Before we predict across all files, lets see how our model does on one file.
mp3 = os.path.join('data', 'Forest Recordings', 'recording_00.mp3')
wav = load_mp3_16k_mono(mp3)

# lets convert this file into slices that are 3 seconds long (48,000/16,000 = 3)
audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=48000, sequence_stride=48000, batch_size=1)
audio_slices = audio_slices.map(preprocess_mp3)
audio_slices = audio_slices.batch(64)

yhat = model.predict(audio_slices)
# We onlyt want to keep those clips that have a confidence of +95%
yhat = [1 if prediction > 0.95 else 0 for prediction in yhat]

# In the case where a single call goes over multiple clips, we want to count that as 1 rather than 2 or how ever many clips
# consecutively detected the call.
yhat = [key for key, group in groupby(yhat)]
calls = tf.math.reduce_sum(yhat).numpy()
print("Number of calls: ", calls)


# Now that we see this works for one file, lets loop through all the files and make predictions
results = {}
for file in os.listdir(os.path.join('data', 'Forest Recordings')):
    FILEPATH = os.path.join('data', 'Forest Recordings', file)

    wav = load_mp3_16k_mono(FILEPATH)
    audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=48000, sequence_stride=48000,
                                                                batch_size=1)
    # Convert these audio slices into spectrograms
    audio_slices = audio_slices.map(preprocess_mp3)
    # Batch the spectrograms in groups of 64
    audio_slices = audio_slices.batch(64)

    yhat = model.predict(audio_slices)

    results[file] = yhat

# For every audio file we have an array of predictions over each slice
print(results)


# Now lets convert our predictions into classes ( 0s or 1s)
class_preds = {}
for file, logits in results.items():
    class_preds[file] = [1 if prediction > 0.95 else 0 for prediction in logits]


# Now lets create a sum over each file's predictions in order to find the sum of calls in each recording
postprocessed = {}
for file, scores in class_preds.items():
    postprocessed[file] = tf.math.reduce_sum([key for key, group in groupby(scores)]).numpy()
print("Processed Predictions\nCall Density: ", postprocessed)


# Now we will export these predictions to a csv to better analyze their call density content
with open('results.csv', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['recording', 'capuchin_calls'])
    for key, value in postprocessed.items():
        writer.writerow([key, value])
