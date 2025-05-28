import os
import json
import numpy as np
import librosa
import librosa.display
from pydub import AudioSegment
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class AudioAnalyzer:
    def __init__(self):
        # Pre-trained model parameters (simplified for demo purposes)
        self.feature_scaler = StandardScaler()
        # In a real implementation, load actual pre-trained model weights
        self.model = self._initialize_model()
        
    def _initialize_model(self):
        """Initialize a simple pre-trained model for audio classification."""
        # This is a placeholder - in production, you would load actual weights
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        return model
    
    def _extract_features(self, audio_file_path):
        """Extract audio features using librosa."""
        try:
            # Convert to wav using pydub if not already in wav format
            if not audio_file_path.lower().endswith('.wav'):
                audio = AudioSegment.from_file(audio_file_path)
                wav_path = os.path.splitext(audio_file_path)[0] + '.wav'
                audio.export(wav_path, format='wav')
                audio_file_path = wav_path
            
            # Load the audio file with librosa
            y, sr = librosa.load(audio_file_path, sr=22050)
            
            # Extract features
            # 1. Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            
            # 2. Rhythm features
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            # 3. MFCC features (good for speech/music discrimination)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_means = np.mean(mfccs, axis=1)
            mfcc_vars = np.var(mfccs, axis=1)
            
            # 4. Zero crossing rate (high for noise, lower for tonal sounds)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            
            # 5. RMS energy (loudness)
            rms = librosa.feature.rms(y=y)[0]
            
            # Combine features
            features = np.concatenate([
                [np.mean(spectral_centroids), np.std(spectral_centroids)],
                [np.mean(spectral_rolloff), np.std(spectral_rolloff)],
                [tempo],
                mfcc_means, mfcc_vars,
                [np.mean(zero_crossing_rate), np.std(zero_crossing_rate)],
                [np.mean(rms), np.std(rms)]
            ])
            
            # Clean up temporary file if created
            if audio_file_path != os.path.splitext(audio_file_path)[0] + '.wav':
                if os.path.exists(wav_path):
                    os.remove(wav_path)
                    
            return features
            
        except Exception as e:
            raise Exception(f"Feature extraction failed: {str(e)}")
    
    def _classify_audio(self, features):
        """Classify audio based on extracted features."""
        # In a real implementation, you would use the actual trained model
        # Here we simulate classification based on feature patterns
        
        # Normalize features
        features_scaled = self.feature_scaler.fit_transform(features.reshape(1, -1))
        
        # In a production system, we'd use a real trained model:
        # prediction = self.model.predict(features_scaled)
        
        # For demonstration purposes, use simple heuristics
        mfcc_mean = np.mean(features[14:27])  # MFCC means
        zero_crossing_mean = features[-4]      # Zero crossing rate mean
        rms_mean = features[-2]                # RMS energy mean
        
        # Simple rule-based classification for demo
        if zero_crossing_mean > 0.15:  # High zero-crossing rate often indicates noise
            category = 'noise'
        elif mfcc_mean < -20:  # Lower MFCC values might indicate music
            category = 'music'
        elif rms_mean > 0.1 and zero_crossing_mean > 0.08:  # Multiple voices often have varied energy
            category = 'multiple voices'
        else:
            category = 'normal'  # Default case
            
        confidence = 0.75  # Placeholder confidence score
        
        return category, confidence

def analyze_audio_segment(audio_file_path):
    """
    Analyze an audio segment and classify it.
    
    Args:
        audio_file_path (str): Path to the audio file
        
    Returns:
        dict: JSON-compatible dictionary with classification results
    """
    if not os.path.exists(audio_file_path):
        return {
            "status": "error",
            "message": f"File not found: {audio_file_path}"
        }
    
    try:
        analyzer = AudioAnalyzer()
        features = analyzer._extract_features(audio_file_path)
        category, confidence = analyzer._classify_audio(features)
        
        return {
            "status": "success",
            "classification": {
                "category": category,
                "confidence": confidence
            },
            "file_path": audio_file_path
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Analysis failed: {str(e)}"
        }

# For testing the module directly
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = analyze_audio_segment(sys.argv[1])
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python audio_analyzer.py [audio_file_path]") 