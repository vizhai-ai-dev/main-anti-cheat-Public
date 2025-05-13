import ffmpeg
import whisper
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

class AudioAnalyzer:
    def __init__(self, hf_token: str = None):
        # Initialize Whisper model
        self.whisper_model = whisper.load_model("base")
        
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
        result = self.whisper_model.transcribe(audio_path)
        whisper_segments = []
        
        for segment in result['segments']:
            if segment['avg_logprob'] < self.WHISPER_THRESHOLD:
                whisper_segments.append({
                    "start": segment['start'],
                    "end": segment['end']
                })
        
        return whisper_segments
    
    def _analyze_speakers(self, audio_path: str) -> Tuple[int, List[Dict]]:
        """Analyze speakers using Whisper."""
        result = self.whisper_model.transcribe(audio_path)
        
        # Count unique speakers and their segments
        speakers = set()
        speaker_segments = []
        
        for segment in result['segments']:
            # Use the segment's confidence as a proxy for speaker identification
            # Lower confidence might indicate a different speaker
            speaker_id = f"SPEAKER_{len(speakers)}" if segment['avg_logprob'] < 0.5 else "SPEAKER_0"
            speakers.add(speaker_id)
            
            if segment['end'] - segment['start'] >= self.MIN_SPEAKER_DURATION:
                speaker_segments.append({
                    "speaker": speaker_id,
                    "start": segment['start'],
                    "end": segment['end']
                })
        
        return len(speakers), speaker_segments
    
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

# FastAPI implementation
app = FastAPI()

class VideoRequest(BaseModel):
    video_path: str

@app.post("/audio_analysis")
async def audio_analysis_endpoint(request: VideoRequest):
    try:
        analyzer = AudioAnalyzer()
        results = analyzer.analyze_audio(request.video_path)
        return results
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)  # Using port 8003 to avoid conflicts 