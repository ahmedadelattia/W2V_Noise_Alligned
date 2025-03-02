from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")
from whisper_normalizer.english import EnglishTextNormalizer

import jiwer
import pandas as pd
import torch
from tqdm import tqdm
import sys
# Set this flag to True to use KenLM decoding via pyctcdecode.


normalizer = EnglishTextNormalizer()
model_name = sys.argv[1]
#optional kenlm_model_path 
kenlm_model_path = sys.argv[2] if len(sys.argv) > 2 else None
print(kenlm_model_path)
use_kenlm = kenlm_model_path is not None
if use_kenlm:
    # Use KenLM inference with pyctcdecode.
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    import librosa
    import numpy as np
    from pyctcdecode import build_ctcdecoder

    # Load the model and processor
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    
    # Build labels list sorted by index from the tokenizer vocabulary.
    vocab_dict = processor.tokenizer.get_vocab()
    sorted_vocab = sorted(vocab_dict.items(), key=lambda item: item[1])
    labels = [token for token, idx in sorted_vocab]
    
    # Set your KenLM model file path and tuning parameters.
    kenlm_alpha = 1  # Language model weight (tunable)
    kenlm_beta = 0.1  # Word insertion bonus (tunable)

    # Build the CTC decoder with KenLM.
    decoder = build_ctcdecoder(
        labels,
        kenlm_model_path=kenlm_model_path,
        alpha=kenlm_alpha,
        beta=kenlm_beta,
    )
    
    def transcribe_audio(audio_path):
        """Transcribe audio using wav2vec2 with KenLM decoding."""
        # Load audio and resample to 16kHz.
        speech, sr = librosa.load(audio_path, sr=16000)
        # Prepare the input values.
        input_values = processor(speech, return_tensors="pt", sampling_rate=16000).input_values
        # Run inference.
        with torch.no_grad():
            logits = model(input_values).logits
        logits = logits[0].cpu().numpy()
        # Decode using the KenLM decoder.
        transcription = decoder.decode(logits)
        return normalizer(transcription)
else:
    # Use the default Hugging Face pipeline.
    wav2vec2_pipeline = pipeline(
    "automatic-speech-recognition", 
    model=model_name, 
    device=0 if torch.cuda.is_available() else -1
)

    def transcribe_audio(audio_path):
        """Transcribe audio using the Wav2Vec2 pipeline."""
        result = wav2vec2_pipeline(audio_path)
        return normalizer(result["text"].strip())
# Load test manifest (Fairseq-style).
def read_manifest(manifest_name, split):
    manifest = {}
    with open(f"../finetune_w2v_fairseq/manifest/{manifest_name}/{split}.tsv", "r") as f:
        file_paths = f.readlines()
    with open(f"../finetune_w2v_fairseq/manifest/{manifest_name}/{split}.wrd", "r") as f:
        wrd = f.readlines()
    # The .ltr file is read but not used here; adjust if needed.
    with open(f"../finetune_w2v_fairseq/manifest/{manifest_name}/{split}.ltr", "r") as f:
        ltr = f.readlines()
    root = file_paths[0].split("\t")[0].strip()
    for i in range(1, len(file_paths)):
        path = root + "/" + file_paths[i].split("\t")[0].strip()
        manifest[path] = normalizer(wrd[i-1].strip())
    return root, manifest

manifest_name = "Librispeech"

# Process multiple splits.
root, manifest = read_manifest(manifest_name, f"valid")
audio_paths = list(manifest.keys())
ground_truths = list(manifest.values())

# Verify matching counts.
assert len(audio_paths) == len(ground_truths), "Mismatch between audio files and ground truth transcripts."

# Transcribe each audio file.
hypotheses = []
for path in tqdm(audio_paths, desc=f"Transcribing split valid"):
    hypotheses.append(transcribe_audio(path))

# Compute Word Error Rate (WER).
wer = jiwer.wer(ground_truths, hypotheses)
print(f"Word Error Rate (WER) split: {wer}")
