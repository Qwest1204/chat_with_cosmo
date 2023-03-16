import torch
import sounddevice as sd
import time
import colorful as cf
import random

language = 'en'
model_id = 'v3_en'
sample_rate = 48000
speaker = 'en_1'
put_accent = True 
put_yoo = True 
device = torch.device('cpu')
text = "Hi, i'm a voice master"


model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                          model='silero_tts',
                          language=language,
                          speaker=model_id)
model.to(device)

audio = model.apply_tts(text=text,
                        speaker=speaker,
                        sample_rate=sample_rate,
                        put_accent=put_accent,
                        put_yo=put_yoo)


def neo_print(value, base_delay=0.03):
    for char in value:
        print(char, end='', flush=True)
        time.sleep(base_delay + random.randint(1, 100) * base_delay / 100)
    print()
if __name__ == '__main__':
    neo_print(cf.blue("Cosmo: " + text))
    sd.play(audio, sample_rate)
    time.sleep(len(audio) / sample_rate + 1.1)
    sd.stop()



