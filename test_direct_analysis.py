#!/usr/bin/env python3
"""
Direct test of ProctorAnalyzer to debug the 500 error
"""

from run_all import ProctorAnalyzer
import os
import traceback

def main():
    video_path = 'Test_data/Movie on 22-05-25 at 9.25 PM.mp4'
    
    if not os.path.exists(video_path):
        print(f'❌ Video file not found: {video_path}')
        return
    
    print(f'✅ Video file exists: {video_path}')
    print('🚀 Starting direct analysis...')
    
    try:
        analyzer = ProctorAnalyzer()
        results = analyzer.run_comprehensive_analysis(video_path)
        
        print('✅ Analysis completed successfully!')
        cheat_score = results.get('cheat_score_analysis', {}).get('cheat_score', 'N/A')
        risk_level = results.get('cheat_score_analysis', {}).get('risk_level', 'N/A')
        print(f'📊 Cheat score: {cheat_score}')
        print(f'⚠️  Risk level: {risk_level}')
        
    except Exception as e:
        print(f'❌ Error during analysis: {str(e)}')
        print('📋 Full traceback:')
        traceback.print_exc()

if __name__ == "__main__":
    main() 