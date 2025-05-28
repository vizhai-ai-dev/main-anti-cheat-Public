#!/usr/bin/env python3
# audio_analysis.py - Audio analysis module for proctoring applications

import librosa
import numpy as np
import json
from typing import Dict, Any, Tuple, List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioAnalyzer:
    """Audio analysis class for detecting anomalies in audio streams."""
    
    def __init__(self, sample_rate: int = 22050):
        """Initialize the audio analyzer.
        
        Args:
            sample_rate: Sample rate for audio processing (default: 22050 Hz)
        """
        self.sample_rate = sample_rate
        self.silence_threshold = 0.01  # Amplitude threshold for silence detection
        self.silence_duration_threshold = 3.0  # Seconds of silence to be considered anomalous
        self.noise_threshold = 0.05  # Amplitude threshold for background noise detection
    
    def extract_mfcc_features(self, audio_data: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        """Extract MFCC features from audio data.
        
        Args:
            audio_data: Audio time series
            n_mfcc: Number of MFCCs to extract
            
        Returns:
            MFCC features as numpy array
        """
        try:
            mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=n_mfcc)
            return mfccs
        except Exception as e:
            logger.error(f"Failed to extract MFCC features: {str(e)}")
            return np.array([])
    
    def detect_multiple_speakers(self, audio_data: np.ndarray) -> Tuple[bool, float]:
        """Detect presence of multiple speakers in audio.
        
        This function uses spectral contrast and energy variance to detect
        potential presence of multiple speakers.
        
        Args:
            audio_data: Audio time series
            
        Returns:
            Tuple containing (is_multiple_speakers, confidence_score)
        """
        try:
            # Extract spectral contrast
            contrast = librosa.feature.spectral_contrast(y=audio_data, sr=self.sample_rate)
            
            # Calculate energy
            energy = librosa.feature.rms(y=audio_data)[0]
            
            # Calculate variance in energy
            energy_variance = np.var(energy)
            
            # Heuristic: High spectral contrast variance and energy variance often indicates multiple speakers
            contrast_variance = np.var(contrast)
            
            # Combine features for a confidence score
            confidence_score = min(1.0, (contrast_variance * 10 + energy_variance * 50))
            
            # Threshold for multiple speakers detection
            threshold = 0.6
            is_multiple = confidence_score > threshold
            
            return is_multiple, confidence_score
        except Exception as e:
            logger.error(f"Failed to detect multiple speakers: {str(e)}")
            return False, 0.0
    
    def detect_background_noise(self, audio_data: np.ndarray) -> Tuple[bool, float]:
        """Detect and measure background noise levels.
        
        Args:
            audio_data: Audio time series
            
        Returns:
            Tuple containing (has_significant_noise, noise_level)
        """
        try:
            # Calculate spectral flatness - high values indicate noise
            flatness = np.mean(librosa.feature.spectral_flatness(y=audio_data))
            
            # Calculate signal-to-noise ratio estimation
            signal_mean = np.mean(np.abs(audio_data))
            
            # Sort amplitude values and take lower 20% as noise estimate
            sorted_amplitudes = np.sort(np.abs(audio_data))
            noise_estimate = np.mean(sorted_amplitudes[:int(len(sorted_amplitudes) * 0.2)])
            
            # Avoid division by zero
            if noise_estimate < 1e-10:
                noise_estimate = 1e-10
                
            snr = 20 * np.log10(signal_mean / noise_estimate)
            
            # Normalize to 0-1 range for noise level
            noise_level = min(1.0, max(0.0, (1.0 - (snr / 60))))
            
            # Determine if noise is significant
            has_significant_noise = noise_level > self.noise_threshold
            
            return has_significant_noise, noise_level
        except Exception as e:
            logger.error(f"Failed to detect background noise: {str(e)}")
            return False, 0.0
    
    def detect_silence(self, audio_data: np.ndarray) -> Tuple[bool, List[Tuple[float, float]]]:
        """Detect prolonged silence periods in audio.
        
        Args:
            audio_data: Audio time series
            
        Returns:
            Tuple containing (has_prolonged_silence, list_of_silence_periods)
            where each period is (start_time, end_time) in seconds
        """
        try:
            # Calculate amplitude envelope
            amplitude_env = np.abs(audio_data)
            
            # Find silent regions (below threshold)
            is_silence = amplitude_env < self.silence_threshold
            
            # Convert to silence segments with start and end times
            silence_periods = []
            in_silence = False
            start_idx = 0
            
            for i, silent in enumerate(is_silence):
                if silent and not in_silence:
                    # Start of silence
                    in_silence = True
                    start_idx = i
                elif not silent and in_silence:
                    # End of silence
                    in_silence = False
                    # Convert indices to seconds
                    start_time = start_idx / self.sample_rate
                    end_time = i / self.sample_rate
                    duration = end_time - start_time
                    
                    # Only include if longer than threshold
                    if duration >= self.silence_duration_threshold:
                        silence_periods.append((start_time, end_time))
            
            # Check if we ended in a silence
            if in_silence:
                start_time = start_idx / self.sample_rate
                end_time = len(audio_data) / self.sample_rate
                duration = end_time - start_time
                
                if duration >= self.silence_duration_threshold:
                    silence_periods.append((start_time, end_time))
            
            has_prolonged_silence = len(silence_periods) > 0
            
            return has_prolonged_silence, silence_periods
        except Exception as e:
            logger.error(f"Failed to detect silence: {str(e)}")
            return False, []
    
    def analyze_audio(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Perform comprehensive analysis of audio data.
        
        Args:
            audio_data: Audio time series
            
        Returns:
            Dictionary with analysis results
        """
        # Make sure audio is mono
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_data = librosa.to_mono(audio_data)
        
        # Extract MFCC features
        mfcc_features = self.extract_mfcc_features(audio_data)
        
        # Detect multiple speakers
        multiple_speakers, speaker_confidence = self.detect_multiple_speakers(audio_data)
        
        # Analyze background noise
        has_noise, noise_level = self.detect_background_noise(audio_data)
        
        # Detect prolonged silence
        has_silence, silence_periods = self.detect_silence(audio_data)
        
        # Compile results
        results = {
            "anomalies_detected": multiple_speakers or has_noise or has_silence,
            "multiple_speakers": {
                "detected": multiple_speakers,
                "confidence": float(speaker_confidence)
            },
            "background_noise": {
                "significant_noise": has_noise,
                "noise_level": float(noise_level)
            },
            "prolonged_silence": {
                "detected": has_silence,
                "periods": [{"start": float(s), "end": float(e)} for s, e in silence_periods]
            },
            "mfcc_features": {
                "shape": list(mfcc_features.shape),
                "mean": float(np.mean(mfcc_features)),
                "std": float(np.std(mfcc_features))
            },
            "audio_duration": len(audio_data) / self.sample_rate
        }
        
        return results

    def analyze_audio_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze audio from a file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Load audio file
            audio_data, _ = librosa.load(file_path, sr=self.sample_rate, mono=True)
            return self.analyze_audio(audio_data)
        except Exception as e:
            logger.error(f"Failed to analyze audio file: {str(e)}")
            return {"error": str(e), "anomalies_detected": False}

# Flask-compatible function to process audio and return JSON results
def process_audio_data(audio_data: np.ndarray, sample_rate: int = 22050) -> Dict[str, Any]:
    """Process audio data and return analysis results.
    
    This function is designed to be called from a Flask route handler.
    
    Args:
        audio_data: Audio time series
        sample_rate: Sample rate of the audio data
        
    Returns:
        Dictionary with analysis results that can be returned as JSON
    """
    analyzer = AudioAnalyzer(sample_rate=sample_rate)
    results = analyzer.analyze_audio(audio_data)
    return results 