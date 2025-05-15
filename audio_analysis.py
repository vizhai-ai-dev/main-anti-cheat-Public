import ffmpeg
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hugging Face token
HF_TOKEN = "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"

class AudioAnalyzer:
    def __init__(self, hf_token: str = None):
        # Initialize Whisper model
        try:
            self.processor = AutoProcessor.from_pretrained(
                "openai/whisper-base",
                token=HF_TOKEN
            )
            self.whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                "openai/whisper-base",
                token=HF_TOKEN
            )
            # Move model to CPU and set to evaluation mode
            self.whisper_model = self.whisper_model.cpu()
            self.whisper_model.eval()
        except Exception as e:
            logger.error(f"Error loading Whisper model: {str(e)}")
            raise RuntimeError("Failed to initialize Whisper model. Please ensure transformers is installed correctly.")
        
        # Initialize Silero VAD
        self.vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=True,
            trust_repo=True
        )
        self.get_speech_timestamps = utils[0]
        self.save_audio = utils[2]
        self.read_audio = utils[3]
        
        # Move model to CPU and set to evaluation mode
        self.vad_model = self.vad_model.cpu()
        self.vad_model.eval()
        
        # Analysis parameters
        self.WHISPER_THRESHOLD = 0.3  # Threshold for whisper detection
        self.MIN_SPEAKER_DURATION = 2.0  # Minimum duration for speaker detection
        self.SILENCE_THRESHOLD = 0.1  # Threshold for silence detection
        
    def _extract_audio(self, video_path: str) -> str:
        """Extract audio from video using ffmpeg."""
        audio_path = str(Path(video_path).with_suffix('.wav'))
        try:
            # First check if the video has an audio stream
            probe = ffmpeg.probe(video_path)
            has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])
            
            if not has_audio:
                logger.warning("No audio stream found in the video. Creating silent audio.")
                # Create a silent audio file with the same duration as the video
                duration = float(probe['format']['duration'])
                stream = ffmpeg.input('anullsrc', f='lavfi', t=duration)
                stream = ffmpeg.output(stream, audio_path, acodec='pcm_s16le', ac=1, ar='16k')
                ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
            else:
                # Extract audio from video
                stream = ffmpeg.input(video_path)
                stream = ffmpeg.output(stream, audio_path, acodec='pcm_s16le', ac=1, ar='16k')
                ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
            
            # Verify the file was created
            if not os.path.exists(audio_path):
                logger.error(f"Failed to create audio file at {audio_path}")
                raise FileNotFoundError(f"Audio extraction failed: {audio_path}")
                
            logger.info(f"Audio extracted successfully to {audio_path}")
            return audio_path
        except ffmpeg.Error as e:
            error_message = e.stderr.decode() if hasattr(e, 'stderr') else str(e)
            logger.error(f"Error extracting audio: {error_message}")
            raise
    
    def _detect_whispers(self, audio_path: str) -> List[Dict[str, float]]:
        """Detect whisper-like segments using Whisper."""
        try:
            # Load and preprocess audio
            import torchaudio
            waveform, sample_rate = torchaudio.load(audio_path)
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Process audio with Whisper
            input_features = self.processor(waveform.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt").input_features
            predicted_ids = self.whisper_model.generate(input_features)
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
            
            # For now, return empty list as we need to implement proper whisper detection
            # This is a placeholder that will be implemented based on the model's output
            return []
        except Exception as e:
            logger.error(f"Error in whisper detection: {str(e)}")
            return []
    
    def _analyze_speakers(self, audio_path: str) -> Tuple[int, List[Dict]]:
        """Analyze speakers using Whisper."""
        try:
            # Load and preprocess audio
            import torchaudio
            waveform, sample_rate = torchaudio.load(audio_path)
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Process audio with Whisper
            input_features = self.processor(waveform.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt").input_features
            predicted_ids = self.whisper_model.generate(input_features)
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
            
            # For now, return default values as we need to implement proper speaker analysis
            # This is a placeholder that will be implemented based on the model's output
            return 1, []  # Assume single speaker for now
        except Exception as e:
            logger.error(f"Error in speaker analysis: {str(e)}")
            return 1, []  # Return default values on error
    
    def _detect_voice_activity(self, audio_path: str) -> List[Dict[str, float]]:
        """Detect voice activity using Silero VAD."""
        try:
            logger.info(f"Starting voice activity detection on {audio_path}")
            
            # Verify file exists
            if not os.path.exists(audio_path):
                logger.error(f"Audio file not found: {audio_path}")
                return []
                
            # Get file info
            file_size = os.path.getsize(audio_path)
            logger.info(f"Audio file size: {file_size} bytes")
            
            if file_size == 0:
                logger.error("Audio file is empty")
                return []
            
            # Read audio file using torchaudio instead of the utility function
            try:
                import torchaudio
                waveform, sample_rate = torchaudio.load(audio_path)
                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                # Resample if needed
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                    waveform = resampler(waveform)
                # Ensure it's a 1D tensor
                wav = waveform.squeeze()
                logger.info(f"Audio loaded successfully: shape={wav.shape}, type={type(wav)}")
            except Exception as audio_err:
                logger.error(f"Error loading audio with torchaudio: {str(audio_err)}")
                # Fallback to original method
                wav = self.read_audio(audio_path, sampling_rate=16000)
                if isinstance(wav, str):
                    logger.error(f"Audio data is a string, not a tensor: {wav[:100]}")
                    return []
                
                if not isinstance(wav, torch.Tensor):
                    logger.info(f"Converting numpy array to tensor, shape: {wav.shape}")
                    wav = torch.tensor(wav, dtype=torch.float32)
            
            # Ensure the model is in evaluation mode
            self.vad_model.eval()
            
            # Set parameters for voice detection
            sampling_rate = 16000
            window_size_samples = 512  # 32ms at 16kHz
            
            # Log tensor information before processing
            logger.info(f"Tensor shape: {wav.shape}, dtype: {wav.dtype}, device: {wav.device}")
            logger.info(f"Sample values - min: {wav.min().item()}, max: {wav.max().item()}, mean: {wav.mean().item()}")
            
            # Get speech timestamps
            speech_timestamps = self.get_speech_timestamps(
                wav, 
                self.vad_model,
                threshold=0.5,
                sampling_rate=sampling_rate,
                min_speech_duration_ms=100,
                min_silence_duration_ms=100,
                window_size_samples=window_size_samples
            )
            
            logger.info(f"Detected {len(speech_timestamps)} speech segments")
            
            # Convert timestamps to seconds
            result = [{"start": ts['start'] / sampling_rate, "end": ts['end'] / sampling_rate} 
                     for ts in speech_timestamps]
            
            return result
        except Exception as e:
            logger.error(f"Error in voice activity detection: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # Return empty list if there's an error
            return []
    
    def analyze_audio(self, video_path: str) -> Dict:
        """Main function to analyze audio from a video."""
        try:
            # Extract audio
            audio_path = self._extract_audio(video_path)
            
            # Get video duration
            probe = ffmpeg.probe(video_path)
            duration = float(probe['format']['duration'])
            
            # Run analysis
            whisper_segments = self._detect_whispers(audio_path)
            num_speakers, speaker_segments = self._analyze_speakers(audio_path)
            voice_segments = self._detect_voice_activity(audio_path)
            
            # Clean up temporary audio file
            os.remove(audio_path)
            
            # Calculate suspicion score (0-100)
            score = min(100, (
                (len(whisper_segments) * 10) +  # Points for whisper segments
                (max(0, num_speakers - 1) * 15) +  # Points for multiple speakers
                (len(voice_segments) * 0.5)  # Points for voice activity
            ))
            
            return {
                "total_duration": str(timedelta(seconds=int(duration))),
                "num_speakers": num_speakers,
                "whispers_detected": len(whisper_segments) > 0,
                "whisper_segments": whisper_segments,
                "speaker_segments": speaker_segments,
                "voice_activity": voice_segments,
                "score": round(score, 1)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing audio: {str(e)}")
            raise
    
    def __del__(self):
        """Cleanup when the object is destroyed."""
        try:
            if hasattr(self, 'whisper_model'):
                del self.whisper_model
            if hasattr(self, 'vad_model'):
                del self.vad_model
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    async def analyze(self, video_path: str) -> Dict:
        """
        Analyze a video for audio issues.
        This is an async wrapper around analyze_audio.
        """
        try:
            result = self.analyze_audio(video_path)
            return {
                "score": result["score"],
                "multiple_speakers": result["num_speakers"] > 1,
                "keyboard_typing_count": 0,  # This would need to be implemented
                "silence_percentage": 0,  # This would need to be calculated from voice_activity
                "background_noise_level": "Low",  # This would need to be implemented
                "speaking_timeline": [
                    {"start": f"00:00:{int(seg['start']):02d}", "end": f"00:00:{int(seg['end']):02d}", "speaker": "primary"}
                    for seg in result["voice_activity"]
                ]
            }
        except Exception as e:
            logger.error(f"Error in audio analysis: {str(e)}")
            raise

# FastAPI implementation
app = FastAPI()

class VideoRequest(BaseModel):
    video_path: str

@app.post("/audio_analysis")
async def audio_analysis_endpoint(request: VideoRequest):
    try:
        analyzer = AudioAnalyzer()
        results = await analyzer.analyze(request.video_path)
        return results
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)  # Using port 8003 to avoid conflicts 