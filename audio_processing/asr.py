from transformers import AutoModel
import torch
import torchaudio

"""
load the indic-conformer-600m-multilingual model and perform ASR
input: 
  audio_file (str) :path to the audio file, 
  lang_id (str) :language ID for ASR. Currently supported languages and their IDs are:
      Assamese: "as"      Bengali: "bn"       Bodo: "brx"
      Dogri: "doi"        Gujarati: "gu"      Hindi: "hi"
      Kannada: "kn"       Konkani: "kok"      Kashmiri: "ks"
      Maithili: "mai"     Malayalam: "ml"     Manipuri: "mni"
      Marathi: "mr"       Nepali: "ne"        Odia: "or"
      Punjabi: "pa"       Sanskrit: "sa"      Santali: "sat"
      Sindhi: "sd"        Tamil: "ta"         Telugu: "te"
      Urdu: "ur"
output: transcription of the audio file in the specified language
"""
def asr_transcribe (audio_file, lang_id):
    # Load the model
    model = AutoModel.from_pretrained("ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True)

    # Load an audio file
    wav, sr = torchaudio.load(audio_file)
    wav = torch.mean(wav, dim=0, keepdim=True)

    target_sample_rate = 16000  # Expected sample rate
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
        wav = resampler(wav)

    # Perform ASR with RNNT decoding
    transcription_rnnt = model(wav, lang_id, "rnnt")
    print("RNNT Transcription:", transcription_rnnt)
    return transcription_rnnt
