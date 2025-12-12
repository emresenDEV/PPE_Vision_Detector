import google.generativeai as genai
from typing import List, Dict
import os
import dotenv


class GeminiRecommendations:
    """Generate safety recommendations using Google Gemini"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Google Gemini API key (or set GEMINI_API_KEY env var)
        """
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        
        if not self.api_key:
            print("Warning: No Gemini API key found. Recommendations will use fallback mode.")
            self.enabled = False
        else:
            genai.configure(api_key=self.api_key)
            # Use gemini-2.5-flash for fast, accurate recommendations
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            self.enabled = True
            print("Gemini API configured successfully with gemini-2.5-flash")
    
    def generate_recommendations(self, summary: Dict, detections: List[Dict]) -> List[str]:
        """
        Generate safety recommendations based on violations.
        
        Args:
            summary: Compliance summary dictionary
            detections: List of detection dictionaries
            
        Returns:
            List of recommendation strings
        """
        if not self.enabled:
            return self._fallback_recommendations(summary)
        
        try:
            # Create prompt for Gemini
            prompt = self._create_prompt(summary, detections)
            
            # Generate content
            response = self.model.generate_content(prompt)
            
            # Parse response into list
            recommendations = self._parse_response(response.text)
            
            return recommendations
            
        except Exception as e:
            print(f"Gemini API error: {e}")
            print("Falling back to rule-based recommendations")
            return self._fallback_recommendations(summary)
    
    def _create_prompt(self, summary: Dict, detections: List[Dict]) -> str:
        """Create structured prompt for Gemini"""
        violations = summary['violations']
        total = summary['total_people']
        rate = summary['compliance_rate']
        
        prompt = f"""You are a workplace safety expert. Based on this PPE compliance analysis, provide exactly 3 specific, actionable safety recommendations.

COMPLIANCE DATA:
- Total workers detected: {total}
- Wearing hard hats: {summary['compliant']}
- NOT wearing hard hats: {violations}
- Compliance rate: {rate:.1f}%
- Status: {summary['status']}

REQUIREMENTS:
1. Provide exactly 3 recommendations
2. Each recommendation should be 1-2 sentences
3. Focus on immediate, actionable steps
4. Prioritize by safety impact
5. Be specific to hard hat compliance
6. Use professional safety language

Format each recommendation as a separate line starting with a number.
"""
        return prompt
    
    def _parse_response(self, response_text: str) -> List[str]:
        """Parse Gemini response into list of recommendations"""
        lines = response_text.strip().split('\n')
        
        recommendations = []
        for line in lines:
            line = line.strip()
            # Remove numbering (1., 2., etc.) if present
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('*')):
                # Remove leading number/bullet and whitespace
                clean_line = line.lstrip('0123456789.-* ')
                if clean_line:
                    recommendations.append(clean_line)
        
        # Ensure we have at least 3 recommendations
        if len(recommendations) < 3:
            recommendations.extend(self._fallback_recommendations({'violations': 1})[:3-len(recommendations)])
        
        return recommendations[:3]  # Return exactly 3
    
    def _fallback_recommendations(self, summary: Dict) -> List[str]:
        """Rule-based fallback recommendations"""
        violations = summary.get('violations', 0)
        
        if violations == 0:
            return [
                "Maintain current safety protocols and continue regular compliance checks",
                "Recognize workers for maintaining excellent PPE compliance standards",
                "Document this compliant inspection for safety records"
            ]
        elif violations <= 2:
            return [
                "Address identified violations immediately with affected workers",
                "Review and reinforce hard hat policy with entire team during next briefing",
                "Increase signage visibility at site entry points"
            ]
        else:
            return [
                "PRIORITY: Halt non-essential operations for immediate safety meeting",
                "Deploy dedicated safety officer for next 24 hours to enforce compliance",
                "Investigate root cause - ensure adequate hard hat availability and accessibility"
            ]


if __name__ == "__main__":
    # Test without API key (fallback mode)
    recommender = GeminiRecommendations()
    
    test_summary = {
        'violations': 2,
        'total_people': 5,
        'compliant': 3,
        'compliance_rate': 60.0,
        'status': 'VIOLATION DETECTED'
    }
    
    recs = recommender.generate_recommendations(test_summary, [])
    
    print("\nTest Recommendations:")
    for i, rec in enumerate(recs, 1):
        print(f"{i}. {rec}")