from typing import Dict

class AudioAnalyzer:
    async def analyze(self, video_path: str) -> Dict:
        """
        Analyze a video for audio issues.
        This is an async wrapper around detect_audio_issues.
        """
        try:
            result = self.detect_audio_issues(video_path)
            
            # Calculate confidence based on multiple factors
            audio_quality_confidence = 0.95  # Base confidence from audio processing
            speaker_detection_confidence = 1.0 - (result["multiple_speakers"] * 0.2)  # Fewer speakers = higher confidence
            noise_confidence = 1.0 - (result["background_noise_level"] / 100)  # Lower noise = higher confidence
            
            # Weighted average of confidence factors
            average_confidence = (
                audio_quality_confidence * 0.4 +  # Audio quality
                speaker_detection_confidence * 0.4 +  # Speaker detection accuracy
                noise_confidence * 0.2  # Noise level impact
            )
            
            return {
                "score": result["score"],
                "multiple_speakers": result["multiple_speakers"],
                "keyboard_typing_count": result["keyboard_typing_count"],
                "silence_percentage": result["silence_percentage"],
                "background_noise_level": result["background_noise_level"],
                "average_confidence": round(average_confidence, 3),
                "confidence_metrics": {
                    "audio_quality_confidence": round(audio_quality_confidence, 3),
                    "speaker_detection_confidence": round(speaker_detection_confidence, 3),
                    "noise_confidence": round(noise_confidence, 3)
                }
            }
        except Exception as e:
            logger.error(f"Error in audio analysis: {str(e)}")
            raise 