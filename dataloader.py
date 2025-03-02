import os
import json
import torch
from torch.utils.data import Dataset
import torchaudio
from whisper_normalizer.english import EnglishTextNormalizer
class SimClassDataset(Dataset):
    """
    A PyTorch Dataset for SimClass data.
    
    Clean audio files are stored in:
        /home/ahmed/Research_Data_2/Data/SimClass/Audio/{split}
    
    Noisy audio files are stored in:
        /home/ahmed/Research_Data_2/Data/SimClass/Noisy/Audio/{split}
    (Each noisy file may have a suffix like _snr5, _snr10, etc. If no _snr is present, the file is actually clean.)
    
    Transcripts are stored in:
        /home/ahmed/Research_Data_2/Data/SimClass/Transcripts/{split}.json
    and are expected to have entries like:
    
        {
            "audio_path": "Audio/development/filename.wav",
            "text": "transcript text...",
            "duration": 24.538
        }
    
    For each noisy file, this dataset returns:
      - input_values_noisy: The waveform tensor for the noisy audio.
      - input_values_clean: The waveform tensor for the corresponding clean audio (if available; otherwise None).
      - attention_mask_noisy: A mask of ones with the same length as the noisy waveform.
      - transcript: The transcript text.
      - sample_rate: The sample rate (from torchaudio.load).
    """
    def __init__(
        self,
        split: str,
        base_dir: str = "/home/ahmed/Research_Data_2/Data/SimClass",
        transform=None,
    ):
        """
        Args:
            split (str): Data split (e.g., "development", "train", "test").
            base_dir (str): The base directory for SimClass data.
            transform: Optional transform to apply to the audio waveform.
        """
        self.split = split
        self.transform = transform

        # Define directories.
        self.clean_dir = os.path.join(base_dir, "Audio", split)
        self.noisy_dir = os.path.join(base_dir, "Noisy_no_silence", "Audio", split)
        transcript_file = os.path.join(base_dir, "Transcripts", f"{split}.json")
        self.normalize = EnglishTextNormalizer()
        # Load transcripts and create a mapping from base file name to transcript text.
        with open(transcript_file, "r") as f:
            transcripts = json.load(f)
        self.transcripts = {}
        for entry in transcripts:
            # Extract the file name from the transcript's audio_path.
            base_name = os.path.basename(entry["audio_path"])
            self.transcripts[base_name] = entry["text"]

        # List all .wav files in the noisy directory.
        self.noisy_files = [
            fname for fname in os.listdir(self.noisy_dir) if fname.endswith(".wav")
        ]
        self.noisy_files.sort()  # sort for deterministic order

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        # Get the noisy file name and full path.
        noisy_fname = self.noisy_files[idx]
        noisy_path = os.path.join(self.noisy_dir, noisy_fname)

        # Load the noisy audio.
        noisy_waveform, sample_rate = torchaudio.load(noisy_path)
        # If the waveform has shape (1, T), squeeze it to (T,)
        if noisy_waveform.ndim == 2 and noisy_waveform.shape[0] == 1:
            noisy_waveform = noisy_waveform.squeeze(0)
        # Create an attention mask (all ones).
        attention_mask_noisy = torch.ones(noisy_waveform.shape[0], dtype=torch.long)

        # Determine the base file name to look up the clean file and transcript.
        # If the noisy file has an '_snr' suffix, remove it.
        if "_snr" in noisy_fname:
            base_name = noisy_fname.split("_snr")[0] + ".wav"
        else:
            base_name = noisy_fname.split("_clean")[0] + ".wav"

        # Construct the clean file path.
        clean_path = os.path.join(self.clean_dir, base_name)
        if os.path.exists(clean_path):
            clean_waveform, sr_clean = torchaudio.load(clean_path)
            if clean_waveform.ndim == 2 and clean_waveform.shape[0] == 1:
                clean_waveform = clean_waveform.squeeze(0)
            #assert that the sample rates match and the lengths are the same
            assert sr_clean == sample_rate, f"Sample rates do not match: {sr_clean} vs {sample_rate}"
            assert clean_waveform.shape[0] == noisy_waveform.shape[0], f"Lengths do not match: {clean_waveform.shape[0]} vs {noisy_waveform.shape[0]}"
        else:
            clean_waveform = None

        # Look up the transcript using the base file name.
        transcript = self.transcripts.get(base_name, "")
        transcript = self.normalize(transcript)
        # Optionally apply a transform.
        if self.transform:
            noisy_waveform = self.transform(noisy_waveform)
            if clean_waveform is not None:
                clean_waveform = self.transform(clean_waveform)

        sample = {
            "input_values_noisy": noisy_waveform,
            "attention_mask_noisy": attention_mask_noisy,
            "input_values_clean": clean_waveform,
            "transcript": transcript,
            "sample_rate": sample_rate,
        }
        return sample