# ai_insights.py
import re
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

class AIInsightsEngine:
    def __init__(self):
        pass
    
    async def detect_appointment_intent(self, message: str) -> bool:
        """Detect if user message contains appointment scheduling intent"""
        try:
            appointment_keywords = [
                'schedule', 'appointment', 'visit', 'tour', 'viewing', 'meeting',
                'see the property', 'look at', 'check out', 'show me', 'when can',
                'available', 'book', 'reserve', 'set up', 'arrange'
            ]
            
            time_indicators = [
                'today', 'tomorrow', 'this week', 'next week', 'monday', 'tuesday',
                'wednesday', 'thursday', 'friday', 'weekend', 'morning', 'afternoon',
                'evening', 'am', 'pm', 'time', 'date'
            ]
            
            message_lower = message.lower()
            
            # Check for appointment keywords
            has_appointment_keyword = any(keyword in message_lower for keyword in appointment_keywords)
            
            # Check for time indicators
            has_time_indicator = any(indicator in message_lower for indicator in time_indicators)
            
            # Higher confidence if both are present
            return has_appointment_keyword or (has_appointment_keyword and has_time_indicator)
            
        except Exception as e:
            logger.error(f"Appointment intent detection error: {str(e)}")
            return False
    
    async def generate_user_insights(self, user_profile, conversation_history: List, current_message: str) -> List[dict]:
        """Generate simple AI insights for a user"""
        try:
            insights = []
            
            if not user_profile:
                return insights
            
            # Simple sentiment analysis
            positive_words = ['great', 'excellent', 'perfect', 'love', 'interested', 'amazing']
            negative_words = ['terrible', 'awful', 'hate', 'bad', 'disappointed']
            
            message_text = current_message.lower()
            positive_count = sum(1 for word in positive_words if word in message_text)
            negative_count = sum(1 for word in negative_words if word in message_text)
            
            if positive_count > negative_count and positive_count > 0:
                insights.append({
                    'title': 'Positive Sentiment Detected',
                    'description': 'Customer shows positive sentiment in recent message.',
                    'priority': 'medium',
                    'confidence': 0.7
                })
            
            # Check for urgency
            urgency_words = ['urgent', 'asap', 'immediately', 'soon', 'deadline']
            if any(word in message_text for word in urgency_words):
                insights.append({
                    'title': 'Urgency Detected',
                    'description': 'Customer has expressed urgency in their request.',
                    'priority': 'high',
                    'confidence': 0.8
                })
            
            # Check engagement level
            if len(conversation_history) > 5:
                insights.append({
                    'title': 'Highly Engaged Customer',
                    'description': f'Customer has had {len(conversation_history)} interactions showing sustained interest.',
                    'priority': 'medium',
                    'confidence': 0.9
                })
            
            return insights
            
        except Exception as e:
            logger.error(f"Generate user insights error: {str(e)}")
            return []
    
    async def calculate_lead_score(self, user_profile) -> int:
        """Calculate simple lead score for a user (0-100)"""
        try:
            score = 20  # Base score
            
            if hasattr(user_profile, 'email') and user_profile.email:
                score += 15
            if hasattr(user_profile, 'phone') and user_profile.phone:
                score += 15
            if hasattr(user_profile, 'company') and user_profile.company:
                score += 10
            if hasattr(user_profile, 'preferences') and user_profile.preferences:
                score += 20
                if user_profile.preferences.get('budget'):
                    score += 20
            
            return min(score, 100)
            
        except Exception as e:
            logger.error(f"Calculate lead score error: {str(e)}")
            return 0
    
    async def generate_appointment_prep(self, appointment_request: dict, broker: dict) -> str:
        """Generate simple preparation notes for an appointment"""
        try:
            prep_notes = f"""
🎯 APPOINTMENT PREPARATION

📋 CUSTOMER OVERVIEW:
• Name: {appointment_request.get('customer_name', 'N/A')}
• Email: {appointment_request.get('customer_email', 'N/A')}
• Phone: {appointment_request.get('customer_phone', 'N/A')}
• Property Interest: {appointment_request.get('property_address', 'General inquiry')}

💡 KEY TALKING POINTS:
• Build rapport and understand their needs
• Present property benefits and features
• Discuss pricing and lease terms
• Address any concerns or questions

📞 RECOMMENDED APPROACH:
• Listen actively to their requirements
• Show enthusiasm and expertise
• Be prepared with property details
• Follow up with next steps

🎯 SUCCESS TIPS:
• Arrive 5 minutes early
• Bring property brochures
• Have lease terms ready
• Ask for feedback and next steps
"""
            
            if appointment_request.get('message'):
                prep_notes += f"\n💬 CUSTOMER MESSAGE:\n{appointment_request['message']}"
            
            return prep_notes
            
        except Exception as e:
            logger.error(f"Generate appointment prep error: {str(e)}")
            return "Preparation notes could not be generated."