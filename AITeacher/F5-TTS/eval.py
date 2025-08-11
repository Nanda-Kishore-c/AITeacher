import argparse
import torch
import torchaudio
from pathlib import Path

# --- Import F5-TTS model and utilities ---
from model import F5TTS
from utils import load_vocab, text_to_sequence
from hifigan import HiFiGAN

def load_model(model_path, vocab_path, device):
    # Load vocabulary
    vocab = load_vocab(vocab_path)
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    # Instantiate the model
    model = F5TTS(vocab_size=len(vocab))
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    return model, vocab

def load_vocoder(vocoder_path, device):
    vocoder = HiFiGAN()
    vocoder.load_state_dict(torch.load(vocoder_path, map_location=device)['generator'])
    vocoder.to(device)
    vocoder.eval()
    return vocoder

def synthesize(model, vocoder, vocab, text, device):
    # Tokenize text
    tokens = text_to_sequence(text, vocab)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    # Generate mel spectrogram
    with torch.no_grad():
        mel = model.infer(tokens)
    # Vocoder: mel to waveform
    with torch.no_grad():
        wav = vocoder.infer(mel).cpu()
    return wav.squeeze(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to F5-TTS model checkpoint')
    parser.add_argument('--vocab', type=str, required=True, help='Path to vocabulary file')
    parser.add_argument('--vocoder', type=str, required=True, help='Path to vocoder checkpoint')
    parser.add_argument('--text', type=str, required=True, help='Text to synthesize')
    parser.add_argument('--output', type=str, required=True, help='Output WAV file path')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Load model and vocoder
    model, vocab = load_model(args.model, args.vocab, args.device)
    vocoder = load_vocoder(args.vocoder, args.device)

    # Synthesize
    waveform = synthesize(model, vocoder, vocab, args.text, args.device)

    # Save output
    torchaudio.save(args.output, waveform.unsqueeze(0), 22050)
    print(f"Synthesized speech saved to {args.output}")

if __name__ == '__main__':
    main()
