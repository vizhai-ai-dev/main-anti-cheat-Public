import asyncio
import logging
import os
import json
from datetime import datetime
from run_all import DirectModuleRunner
from cheat_score import CheatScoreCalculator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluationMetrics:
    def __init__(self):
        self.metrics = {
            "timestamp": datetime.now().isoformat(),
            "overall_metrics": {},
            "module_metrics": {},
            "risk_assessment": {},
            "performance_metrics": {}
        }
    
    def calculate_overall_metrics(self, final_results):
        """Calculate overall evaluation metrics"""
        self.metrics["overall_metrics"] = {
            "final_score": final_results["final_score"],
            "risk_level": final_results["risk"],
            "total_reasons": len(final_results["reasons"]),
            "critical_reasons": sum(1 for r in final_results["reasons"] if r["severity"] == "critical"),
            "warning_reasons": sum(1 for r in final_results["reasons"] if r["severity"] == "warning"),
            "average_module_score": sum(final_results["module_scores"].values()) / len(final_results["module_scores"])
        }
    
    def calculate_module_metrics(self, module_results):
        """Calculate metrics for each module"""
        for module_name, results in module_results.items():
            if "error" in results:
                continue
                
            module_metrics = {
                "score": results.get("score", 0),
                "confidence": results.get("average_confidence", 0),
                "anomalies_detected": 0,
                "performance_indicators": {}
            }
            
            # Module-specific metrics
            if module_name == "gaze":
                module_metrics["anomalies_detected"] = results.get("off_screen_count", 0)
                module_metrics["performance_indicators"] = {
                    "off_screen_percentage": results.get("off_screen_time_percentage", 0),
                    "tracking_confidence": results.get("average_confidence", 0)
                }
            
            elif module_name == "lip_sync":
                module_metrics["anomalies_detected"] = 1 if results.get("major_lip_desync_detected", False) else 0
                module_metrics["performance_indicators"] = {
                    "sync_score": results.get("lip_sync_score", 0),
                    "desync_severity": "high" if results.get("major_lip_desync_detected", False) else "low"
                }
            
            elif module_name == "multi_person":
                module_metrics["anomalies_detected"] = results.get("max_people_detected", 0) - 1
                module_metrics["performance_indicators"] = {
                    "multiple_people_time": results.get("time_with_multiple_people", 0),
                    "different_faces": results.get("different_faces_detected", 0)
                }
            
            elif module_name == "audio":
                module_metrics["anomalies_detected"] = sum([
                    1 if results.get("multiple_speakers", False) else 0,
                    results.get("keyboard_typing_count", 0) > 0,
                    results.get("silence_percentage", 0) > 50
                ])
                module_metrics["performance_indicators"] = {
                    "speaker_count": 2 if results.get("multiple_speakers", False) else 1,
                    "typing_activity": results.get("keyboard_typing_count", 0),
                    "silence_percentage": results.get("silence_percentage", 0),
                    "noise_level": results.get("background_noise_level", "unknown")
                }
            
            self.metrics["module_metrics"][module_name] = module_metrics
    
    def calculate_risk_assessment(self, final_results):
        """Calculate detailed risk assessment metrics"""
        risk_scores = {
            "Very High": 4,
            "High": 3,
            "Medium": 2,
            "Low": 1
        }
        
        self.metrics["risk_assessment"] = {
            "risk_level": final_results["risk"],
            "risk_score": risk_scores.get(final_results["risk"], 0),
            "risk_factors": [
                {
                    "factor": reason["text"],
                    "severity": reason["severity"],
                    "module": reason.get("module", "unknown")
                }
                for reason in final_results["reasons"]
            ],
            "risk_distribution": {
                "critical": sum(1 for r in final_results["reasons"] if r["severity"] == "critical"),
                "warning": sum(1 for r in final_results["reasons"] if r["severity"] == "warning")
            }
        }
    
    def calculate_performance_metrics(self, module_results):
        """Calculate performance metrics for the analysis"""
        self.metrics["performance_metrics"] = {
            "modules_analyzed": len(module_results),
            "successful_modules": sum(1 for m in module_results.values() if "error" not in m),
            "failed_modules": sum(1 for m in module_results.values() if "error" in m),
            "module_success_rate": sum(1 for m in module_results.values() if "error" not in m) / len(module_results) * 100,
            "average_confidence": sum(
                m.get("average_confidence", 0) for m in module_results.values() if "error" not in m
            ) / max(1, sum(1 for m in module_results.values() if "error" not in m))
        }
    
    def save_metrics(self, output_path="evaluation_metrics.json"):
        """Save metrics to a JSON file"""
        with open(output_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        logger.info(f"Evaluation metrics saved to {output_path}")
    
    def print_metrics(self):
        """Print formatted metrics"""
        logger.info("\n=== Evaluation Metrics ===")
        
        # Overall Metrics
        logger.info("\nOverall Metrics:")
        for key, value in self.metrics["overall_metrics"].items():
            logger.info(f"{key.replace('_', ' ').title()}: {value}")
        
        # Module Metrics
        logger.info("\nModule Metrics:")
        for module, metrics in self.metrics["module_metrics"].items():
            logger.info(f"\n{module.replace('_', ' ').title()}:")
            logger.info(f"  Score: {metrics['score']}")
            logger.info(f"  Confidence: {metrics['confidence']}")
            logger.info(f"  Anomalies Detected: {metrics['anomalies_detected']}")
            logger.info("  Performance Indicators:")
            for indicator, value in metrics['performance_indicators'].items():
                logger.info(f"    {indicator.replace('_', ' ').title()}: {value}")
        
        # Risk Assessment
        logger.info("\nRisk Assessment:")
        for key, value in self.metrics["risk_assessment"].items():
            if key != "risk_factors":
                logger.info(f"{key.replace('_', ' ').title()}: {value}")
        
        # Performance Metrics
        logger.info("\nPerformance Metrics:")
        for key, value in self.metrics["performance_metrics"].items():
            logger.info(f"{key.replace('_', ' ').title()}: {value}")

async def test_video(video_path: str):
    try:
        # Normalize the video path to handle spaces
        video_path = os.path.abspath(video_path)
        logger.info(f"Starting analysis of video: {video_path}")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        runner = DirectModuleRunner()
        calculator = CheatScoreCalculator()
        
        # Run all modules
        try:
            module_results = await runner.run_all_modules(video_path)
        except Exception as e:
            logger.error(f"Error during module analysis: {str(e)}")
            module_results = {
                "gaze": {"score": 0, "error": str(e)},
                "lip_sync": {"score": 0, "error": str(e)},
                "multi_person": {"score": 0, "error": str(e)},
                "audio": {"score": 0, "error": str(e)}
            }
        
        # Calculate final score
        try:
            final_results = await calculator.compute_score(module_results)
        except Exception as e:
            logger.error(f"Error calculating final score: {str(e)}")
            final_results = {
                "final_score": 0,
                "risk": "Error",
                "reasons": [{"text": f"Error during analysis: {str(e)}", "severity": "critical"}],
                "module_scores": {module: 0 for module in module_results.keys()}
            }
        
        # Calculate evaluation metrics
        metrics = EvaluationMetrics()
        metrics.calculate_overall_metrics(final_results)
        metrics.calculate_module_metrics(module_results)
        metrics.calculate_risk_assessment(final_results)
        metrics.calculate_performance_metrics(module_results)
        
        # Print detailed results
        logger.info("\n=== Analysis Results ===")
        logger.info(f"Final Score: {final_results['final_score']}")
        logger.info(f"Risk Level: {final_results['risk']}")
        
        # Print reasons
        if final_results['reasons']:
            logger.info("\nReasons:")
            for reason in final_results['reasons']:
                logger.info(f"- {reason['text']} (Severity: {reason['severity']})")
        
        # Print module scores
        logger.info("\nModule Scores:")
        for module, score in final_results['module_scores'].items():
            logger.info(f"{module.capitalize()}: {score}")
        
        # Print detailed module results
        logger.info("\nDetailed Module Results:")
        
        # Gaze results
        if 'gaze' in module_results:
            gaze = module_results['gaze']
            logger.info("\nGaze Analysis:")
            if 'error' in gaze:
                logger.error(f"Error in gaze analysis: {gaze['error']}")
            else:
                logger.info(f"Off-screen Count: {gaze.get('off_screen_count', 'N/A')}")
                logger.info(f"Average Confidence: {gaze.get('average_confidence', 'N/A')}")
                logger.info(f"Off-screen Time Percentage: {gaze.get('off_screen_time_percentage', 'N/A')}%")
        
        # Lip sync results
        if 'lip_sync' in module_results:
            lip_sync = module_results['lip_sync']
            logger.info("\nLip Sync Analysis:")
            if 'error' in lip_sync:
                logger.error(f"Error in lip sync analysis: {lip_sync['error']}")
            else:
                logger.info(f"Lip Sync Score: {lip_sync.get('lip_sync_score', 'N/A')}")
                logger.info(f"Major Lip Desync Detected: {lip_sync.get('major_lip_desync_detected', 'N/A')}")
        
        # Multi-person results
        if 'multi_person' in module_results:
            multi = module_results['multi_person']
            logger.info("\nMulti-Person Analysis:")
            if 'error' in multi:
                logger.error(f"Error in multi-person analysis: {multi['error']}")
            else:
                logger.info(f"Max People Detected: {multi.get('max_people_detected', 'N/A')}")
                logger.info(f"Time with Multiple People: {multi.get('time_with_multiple_people', 'N/A')}s")
                logger.info(f"Different Faces Detected: {multi.get('different_faces_detected', 'N/A')}")
        
        # Audio results
        if 'audio' in module_results:
            audio = module_results['audio']
            logger.info("\nAudio Analysis:")
            if 'error' in audio:
                logger.error(f"Error in audio analysis: {audio['error']}")
            else:
                logger.info(f"Multiple Speakers: {audio.get('multiple_speakers', 'N/A')}")
                logger.info(f"Keyboard Typing Count: {audio.get('keyboard_typing_count', 'N/A')}")
                logger.info(f"Silence Percentage: {audio.get('silence_percentage', 'N/A')}%")
                logger.info(f"Background Noise Level: {audio.get('background_noise_level', 'N/A')}")
        
        # Print evaluation metrics
        metrics.print_metrics()
        
        # Save metrics to file
        metrics.save_metrics()
        
        return final_results
        
    except Exception as e:
        logger.error(f"Error analyzing video: {str(e)}")
        raise

if __name__ == "__main__":
    # Use raw string to handle spaces in path
    video_path = r"/Users/darshanjijhuvadia/Documents/FINAL/Test_data/VID20250429211732.mp4"
    results = asyncio.run(test_video(video_path)) 