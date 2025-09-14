import librosa
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image


def audio_to_melspectrogram(audio_path, n_mels=256, n_fft=4096, hop_length=None):
    y, sr = librosa.load(audio_path, sr=None)
    y = librosa.util.normalize(y)

    if hop_length is None:
        hop_length = n_fft // 4

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec_db, sr


def save_spectrogram(mel_spec, save_path):
    plt.figure(figsize=(5,5))
    librosa.display.specshow(mel_spec, x_axis='time', y_axis="mel")
    plt.axis('off') # remove axis
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    

mel_spec, sr = audio_to_melspectrogram('demo.wav')
save_spectrogram(mel_spec, 'output_demp.png')


model = load_model('gunshot_model.h5')

img = image.load_img("output_demp.png", target_size=(256, 256))
img_array = image.img_to_array(img)  # convert to array
img_array = img_array / 255.0  # normalize if needed (match training)
img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

res = model.predict(img_array)
predicted_class = np.argmax(res)
print("Predicted class:", predicted_class)
