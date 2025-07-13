import requests
import os
import json
import re
from typing import List, Dict, Any, Optional
import logging
from models import ConversationMessage, UserProfile

logger = logging.getLogger(__name__)

class EnhancedLLMService:
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.model = "gpt-4o-mini"
        self.max_tokens = 1500
        self.temperature = 0.7
        self.api_url = "https://api.openai.com/v1/chat/completions"
        
        if self.api_key:
            logger.info(f"OpenAI API key loaded successfully. Model: {self.model}")
        else:
            logger.warning("No OpenAI API key found. Using fallback responses.")
    
    async def generate_response(self, user_message: str, 
                              conversation_history: List[ConversationMessage],
                              rag_context: List[Dict[str, Any]],
                              user_profile: Optional[UserProfile] = None) -> str:
        """Generate response using enhanced multi-agent approach"""
        
        # Step 1: Determine user intent and agent type needed
        intent_analysis = await self.analyze_user_intent(user_message, conversation_history, user_profile)
        
        # Step 2: Route to appropriate agent
        if intent_analysis['agent_type'] == 'appointment_scheduler':
            return "TRIGGER_APPOINTMENT_FORM"
        elif intent_analysis['agent_type'] == 'maintenance_agent':
            return await self.handle_maintenance_request(user_message, intent_analysis, rag_context, user_profile)
        elif intent_analysis['agent_type'] == 'property_agent':
            return await self.handle_property_search(user_message, conversation_history, rag_context, user_profile)
        elif intent_analysis['agent_type'] == 'customer_service_agent':
            return await self.handle_customer_service(user_message, conversation_history, rag_context, user_profile)
        else:
            # Default property search agent
            return await self.handle_property_search(user_message, conversation_history, rag_context, user_profile)
    
    async def analyze_user_intent(self, user_message: str, 
                                conversation_history: List[ConversationMessage],
                                user_profile: Optional[UserProfile] = None) -> Dict[str, Any]:
        """Analyze user intent to determine appropriate agent"""
        
        message_lower = user_message.lower()
        
        # Appointment/scheduling intent
        appointment_phrases = [
            'book', 'schedule', 'appointment', 'visit', 'tour', 'meeting',
            'arrange', 'set up', 'book a', 'schedule a', 'when can i',
            'available times', 'calendar', 'availability'
        ]
        
        if any(phrase in message_lower for phrase in appointment_phrases):
            return {
                'agent_type': 'appointment_scheduler',
                'intent': 'schedule_appointment',
                'confidence': 0.9,
                'context': self._extract_appointment_context(user_message)
            }
        
        # Maintenance/repair intent
        maintenance_phrases = [
            'maintenance', 'repair', 'fix', 'broken', 'not working', 'issue with',
            'problem with', 'leak', 'heating', 'cooling', 'electrical', 'plumbing',
            'hvac', 'air conditioning', 'tenant', 'my unit', 'my apartment', 'my office',
            'emergency', 'urgent', 'flooding', 'no power', 'no heat'
        ]
        
        # Check if user is likely a tenant (has previous maintenance history or mentions unit)
        is_tenant = any(phrase in message_lower for phrase in ['my unit', 'my apartment', 'my office', 'tenant', 'lease'])
        has_maintenance_need = any(phrase in message_lower for phrase in maintenance_phrases)
        
        if has_maintenance_need or (is_tenant and any(phrase in message_lower for phrase in ['help', 'need', 'request'])):
            return {
                'agent_type': 'maintenance_agent',
                'intent': 'maintenance_request',
                'confidence': 0.8,
                'urgency': self._determine_urgency_level(user_message),
                'issue_type': self._classify_maintenance_issue(user_message)
            }
        
        # Property search intent
        property_phrases = [
            'property', 'properties', 'office', 'space', 'rent', 'lease',
            'square feet', 'sq ft', 'sf', 'building', 'floor', 'suite',
            'looking for', 'need', 'find', 'search', 'show me', 'available'
        ]
        
        if any(phrase in message_lower for phrase in property_phrases):
            return {
                'agent_type': 'property_agent',
                'intent': 'property_search',
                'confidence': 0.8,
                'search_criteria': self._extract_property_criteria(user_message)
            }
        
        # Customer service intent
        service_phrases = [
            'help', 'support', 'question', 'information', 'how to', 'what is',
            'explain', 'tell me about', 'contact', 'speak with', 'talk to'
        ]
        
        if any(phrase in message_lower for phrase in service_phrases):
            return {
                'agent_type': 'customer_service_agent',
                'intent': 'general_inquiry',
                'confidence': 0.6
            }
        
        # Default to property agent
        return {
            'agent_type': 'property_agent',
            'intent': 'general_property_inquiry',
            'confidence': 0.5
        }
    
    def _extract_appointment_context(self, message: str) -> Dict[str, Any]:
        """Extract appointment-specific context from message"""
        context = {}
        message_lower = message.lower()
        
        # Extract property mentions
        if 'property' in message_lower:
            property_match = re.search(r'property\s+(\w+)', message_lower)
            if property_match:
                context['property_reference'] = property_match.group(1)
        
        # Extract time preferences
        time_words = ['today', 'tomorrow', 'this week', 'next week', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday']
        for word in time_words:
            if word in message_lower:
                context['time_preference'] = word
                break
        
        # Extract appointment type
        if any(word in message_lower for word in ['virtual', 'online', 'video']):
            context['appointment_type'] = 'virtual_tour'
        elif any(word in message_lower for word in ['maintenance', 'repair', 'fix']):
            context['appointment_type'] = 'maintenance'
        else:
            context['appointment_type'] = 'site_visit'
        
        return context
    
    def _determine_urgency_level(self, message: str) -> str:
        """Determine urgency level for maintenance requests"""
        message_lower = message.lower()
        
        emergency_keywords = ['emergency', 'urgent', 'asap', 'immediately', 'flooding', 'leak', 'no power', 'no heat', 'gas leak']
        if any(keyword in message_lower for keyword in emergency_keywords):
            return 'emergency'
        
        urgent_keywords = ['urgent', 'soon', 'quickly', 'broken', 'not working']
        if any(keyword in message_lower for keyword in urgent_keywords):
            return 'urgent'
        
        return 'normal'
    
    def _classify_maintenance_issue(self, message: str) -> str:
        """Classify the type of maintenance issue"""
        message_lower = message.lower()
        
        categories = {
            'plumbing': ['plumbing', 'leak', 'water', 'pipe', 'drain', 'toilet', 'sink', 'faucet'],
            'electrical': ['electrical', 'power', 'electricity', 'light', 'outlet', 'switch', 'wiring'],
            'hvac': ['heating', 'cooling', 'hvac', 'air conditioning', 'thermostat', 'temperature', 'ac'],
            'structural': ['door', 'window', 'wall', 'ceiling', 'floor', 'stairs'],
            'general': ['cleaning', 'painting', 'maintenance', 'general']
        }
        
        for category, keywords in categories.items():
            if any(keyword in message_lower for keyword in keywords):
                return category
        
        return 'general'
    
    def _extract_property_criteria(self, message: str) -> Dict[str, Any]:
        """Extract property search criteria from message"""
        criteria = {}
        message_lower = message.lower()
        
        # Extract size requirements
        size_patterns = [
            r'(\d+)\s*(?:\+)?\s*(?:sq\s*ft|sqft|square\s*feet|sf)',
            r'(\d+)\s*(?:sq\s*ft|sqft|square\s*feet|sf)\s*(?:\+|plus|or more)'
        ]
        
        for pattern in size_patterns:
            match = re.search(pattern, message_lower)
            if match:
                criteria['size'] = int(match.group(1))
                break
        
        # Extract location preferences
        location_keywords = ['downtown', 'midtown', 'uptown', 'manhattan', 'brooklyn', 'queens']
        for location in location_keywords:
            if location in message_lower:
                criteria['location'] = location
                break
        
        # Extract budget
        budget_pattern = r'\$(\d+(?:,\d{3})*)'
        budget_match = re.search(budget_pattern, message)
        if budget_match:
            criteria['budget'] = budget_match.group(1)
        
        return criteria
    
    async def handle_maintenance_request(self, user_message: str, intent_analysis: Dict[str, Any], 
                                       rag_context: List[Dict[str, Any]], 
                                       user_profile: Optional[UserProfile] = None) -> str:
        """Handle maintenance and repair requests with specialized agent"""
        
        urgency = intent_analysis.get('urgency', 'normal')
        issue_type = intent_analysis.get('issue_type', 'general')
        
        # Build maintenance-specific response
        if urgency == 'emergency':
            response = f"🚨 EMERGENCY MAINTENANCE REQUEST RECEIVED\n\n"
            response += f"I understand you have an urgent {issue_type} issue. This has been flagged as an emergency.\n\n"
            response += f"For immediate assistance:\n"
            response += f"• Emergency Hotline: (555) 911-HELP\n"
            response += f"• Text 'EMERGENCY' to (555) 999-1111\n\n"
            response += f"I'm also scheduling an emergency maintenance appointment for you right away. "
            response += f"Our emergency maintenance team will contact you within 15 minutes.\n\n"
            response += f"Issue reported: {user_message}\n"
            response += f"Priority: EMERGENCY\n"
            response += f"Response time: Within 1 hour"
        
        elif urgency == 'urgent':
            response = f"⚡ URGENT MAINTENANCE REQUEST\n\n"
            response += f"I see you have an urgent {issue_type} issue that needs immediate attention.\n\n"
            response += f"I'm escalating this to our maintenance team with high priority. "
            response += f"You should expect contact within 2-4 hours.\n\n"
            response += f"Issue: {user_message}\n"
            response += f"Category: {issue_type.title()}\n"
            response += f"Priority: HIGH\n\n"
            response += f"Would you like me to schedule a maintenance visit for you?"
        
        else:
            response = f"🔧 MAINTENANCE REQUEST RECEIVED\n\n"
            response += f"Thank you for reporting the {issue_type} issue. I've logged your maintenance request.\n\n"
            response += f"Issue: {user_message}\n"
            response += f"Category: {issue_type.title()}\n"
            response += f"Priority: Standard\n\n"
            response += f"Our maintenance team will review your request and contact you within 24-48 hours to schedule a visit.\n\n"
            response += f"Would you like to schedule a specific time for the maintenance visit?"
        
        # Add tenant-specific information if available
        if user_profile:
            response += f"\n\nTenant Information:\n"
            if user_profile.name:
                response += f"• Name: {user_profile.name}\n"
            if user_profile.email:
                response += f"• Email: {user_profile.email}\n"
            if user_profile.phone:
                response += f"• Phone: {user_profile.phone}\n"
        
        # Add maintenance tips based on issue type
        tips = self._get_maintenance_tips(issue_type, urgency)
        if tips:
            response += f"\n\n💡 In the meantime:\n{tips}"
        
        return response
    
    def _get_maintenance_tips(self, issue_type: str, urgency: str) -> str:
        """Get maintenance tips based on issue type"""
        
        if urgency == 'emergency':
            emergency_tips = {
                'plumbing': "• Turn off water main if flooding\n• Move valuables to dry areas\n• Do not use electrical appliances near water",
                'electrical': "• Turn off power at breaker box if safe\n• Do not touch electrical components\n• Use flashlights, not candles",
                'hvac': "• Check thermostat settings\n• Ensure vents are not blocked\n• If gas smell, evacuate immediately"
            }
            return emergency_tips.get(issue_type, "• Stay safe and avoid the affected area\n• Document the issue with photos if safe")
        
        general_tips = {
            'plumbing': "• Turn off water supply to affected area\n• Place buckets under leaks\n• Avoid using chemical drain cleaners",
            'electrical': "• Check circuit breakers\n• Test GFCI outlets\n• Avoid overloading outlets",
            'hvac': "• Replace air filters if dirty\n• Check thermostat batteries\n• Ensure vents are unblocked",
            'general': "• Take photos of the issue\n• Note when the problem started\n• Keep area clear for maintenance access"
        }
        
        return general_tips.get(issue_type, general_tips['general'])
    
    async def handle_property_search(self, user_message: str, 
                                   conversation_history: List[ConversationMessage],
                                   rag_context: List[Dict[str, Any]],
                                   user_profile: Optional[UserProfile] = None) -> str:
        """Handle property search requests with specialized agent"""
        
        if self.api_key:
            try:
                return await self.call_openai_api(user_message, conversation_history, rag_context, user_profile)
            except Exception as e:
                logger.error(f"OpenAI API failed: {e}")
                logger.info("Falling back to enhanced local responses")
        
        return await self.generate_enhanced_property_response(user_message, rag_context, user_profile)
    
    async def handle_customer_service(self, user_message: str, 
                                    conversation_history: List[ConversationMessage],
                                    rag_context: List[Dict[str, Any]],
                                    user_profile: Optional[UserProfile] = None) -> str:
        """Handle general customer service inquiries"""
        
        message_lower = user_message.lower()
        
        # FAQ handling
        if any(word in message_lower for word in ['hours', 'open', 'contact']):
            return self._handle_contact_inquiry()
        elif any(word in message_lower for word in ['lease', 'contract', 'agreement']):
            return self._handle_lease_inquiry()
        elif any(word in message_lower for word in ['payment', 'rent', 'bill']):
            return self._handle_payment_inquiry()
        elif any(word in message_lower for word in ['amenities', 'facilities', 'features']):
            return self._handle_amenities_inquiry(rag_context)
        else:
            return self._handle_general_inquiry(user_message, rag_context)
    
    def _handle_contact_inquiry(self) -> str:
        return """📞 CONTACT INFORMATION

Office Hours:
• Monday - Friday: 9:00 AM - 6:00 PM
• Saturday: 10:00 AM - 4:00 PM
• Sunday: Closed
• Emergency Maintenance: 24/7

Contact Methods:
• Phone: (555) 123-RENT
• Email: leasing@realestate.com
• Emergency: (555) 911-HELP

Office Location:
123 Business Center Drive
Suite 100
Downtown Business District

How else can I help you today?"""
    
    def _handle_lease_inquiry(self) -> str:
        return """📋 LEASE INFORMATION

Our leasing team can help you with:
• Lease terms and conditions
• Rental agreements
• Lease renewals
• Early termination options
• Subletting policies

For specific lease questions, I can connect you with:
• Leasing Manager: sarah.johnson@realty.com
• Legal Department: legal@realty.com

Would you like me to schedule a consultation with our leasing team?"""
    
    def _handle_payment_inquiry(self) -> str:
        return """💳 PAYMENT INFORMATION

Payment Options:
• Online portal: portal.realty.com
• Automatic bank transfer
• Check (payable to: Property Management Co.)
• Money order

Payment Due Dates:
• Rent: 1st of each month
• Late fee after: 5th of month
• Grace period: 3 days

For payment issues:
• Accounting: accounting@realty.com
• Phone: (555) 123-PAY

Need help setting up online payments?"""
    
    def _handle_amenities_inquiry(self, rag_context: List[Dict[str, Any]]) -> str:
        amenities_info = """🏢 BUILDING AMENITIES

Common amenities in our properties include:
• High-speed fiber internet
• 24/7 security and key card access
• Elevator access
• On-site parking (varies by location)
• Conference rooms and meeting spaces
• Break rooms and kitchenettes
• Modern HVAC systems
• Energy-efficient lighting

Property-specific amenities vary by location. """
        
        if rag_context:
            amenities_info += "Would you like me to show you specific properties with the amenities you're looking for?"
        
        return amenities_info
    
    def _handle_general_inquiry(self, user_message: str, rag_context: List[Dict[str, Any]]) -> str:
        base_response = """I'm here to help with your real estate needs! 

I can assist you with:
• Finding office spaces and commercial properties
• Scheduling property visits
• Maintenance requests for current tenants
• General information about our services
• Connecting you with the right team member

"""
        
        if rag_context:
            base_response += "I found some relevant information that might help. "
            
        base_response += "What specific information are you looking for?"
        
        return base_response
    
    async def generate_enhanced_property_response(self, user_message: str,
                                                rag_context: List[Dict[str, Any]],
                                                user_profile: Optional[UserProfile] = None) -> str:
        """Generate enhanced property search responses"""
        
        message_lower = user_message.lower()
        
        # Check for exact size matches
        exact_size_match = re.search(r'exactly\s+(\d+)\s*(?:sq\s*ft|sqft|square\s*feet|sf)', message_lower)
        if exact_size_match:
            exact_size = exact_size_match.group(1)
            if rag_context:
                response = f"Searching for properties with exactly {exact_size} square feet...\n\n"
                
                # Filter for exact matches
                exact_matches = []
                for prop in rag_context:
                    content = prop.get('content', '')
                    if f"size (sf): {exact_size}" in content:
                        exact_matches.append(prop)
                
                if exact_matches:
                    response += f"Found {len(exact_matches)} properties with exactly {exact_size} sq ft:\n\n"
                    for i, prop in enumerate(exact_matches[:10]):
                        response += f"{i+1}. {prop['content']}\n"
                    
                    if len(exact_matches) > 10:
                        response += f"\n...and {len(exact_matches) - 10} more properties with exactly {exact_size} sq ft."
                else:
                    response += f"No properties found with exactly {exact_size} square feet.\n\n"
                    response += "Here are some similar-sized properties:\n\n"
                    for i, prop in enumerate(rag_context[:5]):
                        response += f"{i+1}. {prop['content']}\n"
                
                return response
        
        # Regular property search handling
        if rag_context:
            response = "Here are the available commercial properties that match your criteria:\n\n"
            
            # Sort properties based on user preferences
            sorted_properties = self.sort_properties_by_preferences(rag_context, user_profile, user_message)
            
            # Return more properties for better selection
            max_properties = min(len(sorted_properties), 25)
            
            for i, doc in enumerate(sorted_properties[:max_properties]):
                content = doc.get('content', '')
                property_data = self.extract_property_from_content(content)
                
                if property_data:
                    formatted_line = self.format_property_line(property_data, i+1)
                    response += formatted_line + "\n"
            
            response += f"\nShowing {max_properties} properties out of {len(sorted_properties)} total matches"
            if user_profile and user_profile.preferences:
                response += " prioritized based on your preferences"
            response += ".\n\nWould you like to:\n"
            response += "• Schedule a visit to any of these properties?\n"
            response += "• Get more details about a specific property?\n"
            response += "• Refine your search criteria?"
            
            return response
        else:
            return """I'd be happy to help you find the perfect office space! 

To provide you with the most relevant properties, could you tell me:
• What size space do you need? (square feet)
• Preferred location or area?
• Budget range?
• Any specific requirements? (parking, amenities, etc.)

You can also say things like:
• "Show me properties with exactly 18650 sq ft"
• "I need 3000+ sq ft downtown with parking"
• "Find me office space under $50,000 annually"

What are you looking for?"""
    
    def build_system_prompt(self, rag_context: List[Dict[str, Any]], 
                           user_profile: Optional[UserProfile] = None) -> str:
        """Build enhanced system prompt for multi-agent support"""
        
        base_prompt = """You are a professional AI assistant for a comprehensive real estate company. You are part of a multi-agent system that helps with property searches, maintenance requests, and customer service.

CRITICAL PROPERTY FORMATTING: When displaying property information, format each property as a SINGLE line with pipe separators (|) in this EXACT format:
[NUMBER]. unique id: [ID] | property address: [ADDRESS] | floor: [FLOOR] | suite: [SUITE] | size (sf): [SIZE] | rent/sf/year: [RENT] | associate 1: [ASSOCIATE1] | associate 2: [ASSOCIATE2] | broker email id: [EMAIL] | phone: [PHONE]

MULTI-AGENT CAPABILITIES:
1. Property Search Agent: Find and recommend office spaces
2. Maintenance Agent: Handle repair and maintenance requests for tenants
3. Appointment Scheduler: Book property visits and maintenance appointments
4. Customer Service Agent: General inquiries and support

ENHANCED FEATURES:
- Exact property size matching (e.g., "exactly 18650 sq ft")
- Maintenance request handling with urgency classification
- Tenant support for existing renters
- Emergency response protocols
- Personalized property recommendations

Your responsibilities:
- Search and recommend properties from our database of 225+ properties
- Handle maintenance and repair requests from current tenants
- Provide detailed property information (size, rent, location, contacts)
- Help with rental calculations and comparisons
- Connect users with appropriate staff (brokers, maintenance, leasing)
- Schedule appointments through our booking system
- Handle emergency maintenance requests with appropriate urgency
- Support both prospective tenants and current renters

Guidelines:
- Be professional, helpful, and knowledgeable
- Use the provided property data for accurate information
- ALWAYS format property listings using the exact pipe-separated format
- For maintenance requests, assess urgency and respond appropriately
- For emergencies, provide immediate contact information
- Include specific details like size, rent, contact information
- When showing multiple properties, number them and use the pipe format
- Prioritize properties based on user preferences and conversation history
- Handle both commercial property searches and residential tenant services
"""
        
        # Add user context
        if user_profile:
            if user_profile.name:
                base_prompt += f"\nUser: {user_profile.name}"
            if user_profile.company:
                base_prompt += f" from {user_profile.company}"
            
            if user_profile.preferences:
                base_prompt += f"\nUser preferences: {json.dumps(user_profile.preferences)}"
        
        # Add RAG context
        if rag_context:
            base_prompt += "\n\nAvailable Properties:\n"
            for i, doc in enumerate(rag_context[:10]):
                base_prompt += f"\n{i+1}. {doc['content'][:500]}...\n"
        
        base_prompt += "\n\nRemember: Format ALL property responses using the exact pipe-separated format. Handle maintenance requests with appropriate urgency."
        return base_prompt
    
    def sort_properties_by_preferences(self, rag_context: List[Dict[str, Any]], 
                                     user_profile: Optional[UserProfile], 
                                     current_message: str) -> List[Dict[str, Any]]:
        """Sort properties based on user preferences and message context"""
        
        # Extract criteria from current message
        message_criteria = self._extract_property_criteria(current_message)
        
        # Score properties based on relevance
        scored_properties = []
        
        for prop in rag_context:
            score = 0
            content = prop.get('content', '').lower()
            
            # Size matching (high priority)
            if 'size' in message_criteria:
                target_size = message_criteria['size']
                size_match = re.search(r'size \(sf\):\s*(\d+)', content)
                if size_match:
                    prop_size = int(size_match.group(1))
                    size_diff = abs(prop_size - target_size)
                    if size_diff == 0:
                        score += 100  # Exact match
                    elif size_diff <= target_size * 0.1:  # Within 10%
                        score += 80
                    elif size_diff <= target_size * 0.2:  # Within 20%
                        score += 60
                    else:
                        score += max(0, 40 - (size_diff / target_size * 100))
            
            # Location matching
            if 'location' in message_criteria:
                if message_criteria['location'] in content:
                    score += 50
            
            # User profile preferences
            if user_profile and user_profile.preferences:
                prefs = user_profile.preferences
                if 'preferred_size' in prefs:
                    # Similar logic for stored preferences
                    pass
                if 'preferred_location' in prefs:
                    if prefs['preferred_location'].lower() in content:
                        score += 30
            
            scored_properties.append((score, prop))
        
        # Sort by score (descending) and return properties
        scored_properties.sort(key=lambda x: x[0], reverse=True)
        return [prop for score, prop in scored_properties]
    
    def extract_property_from_content(self, content: str) -> Optional[Dict[str, str]]:
        """Extract property details from content with enhanced parsing"""
        if not content:
            return None
            
        property_data = {}
        
        # Enhanced patterns for better extraction
        patterns = {
            'id': r'(?:unique\s*id|id)[:\s]+([^\s,|]+)',
            'address': r'(?:property\s*address|address)[:\s]+([^,\n|]+?)(?:\s*\||$)',
            'floor': r'(?:floor)[:\s]+([^\s,\n|]+)',
            'suite': r'(?:suite|unit)[:\s]+([^\s,\n|]+)',
            'size': r'(?:size\s*\(?sf\)?|square\s*feet|sf)[:\s]*([0-9,]+)',
            'rent': r'(?:rent/?sf/?year|rent|price)[:\s]*\$?([0-9,.$]+)',
            'associate1': r'(?:associate\s*1|agent\s*1)[:\s]+([^|,\n]+?)(?:\s*\||$)',
            'associate2': r'(?:associate\s*2|agent\s*2)[:\s]+([^|,\n]+?)(?:\s*\||$)',
            'email': r'(?:broker\s*email\s*id|email)[:\s]*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
            'phone': r'(?:phone|contact)[:\s]*(\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4})',
        }
        
        content_lower = content.lower()
        
        for key, pattern in patterns.items():
            match = re.search(pattern, content_lower, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                # Clean up value
                if '|' in value:
                    value = value.split('|')[0].strip()
                if value and value != 'n/a':
                    property_data[key] = value
        
        # Ensure we have minimum required data
        if 'id' in property_data or 'address' in property_data:
            return property_data
        
        return None
    
    def format_property_line(self, property_data: Dict[str, str], index: int) -> str:
        """Format property data as pipe-separated line with enhanced defaults"""
        
        defaults = {
            'id': f'P{index:03d}',
            'address': 'Address not specified',
            'floor': 'N/A',
            'suite': 'N/A', 
            'size': 'N/A',
            'rent': 'Contact for pricing',
            'associate1': 'Contact office',
            'associate2': 'Contact office',
            'email': 'info@company.com',
            'phone': 'Contact for details'
        }
        
        data = {}
        for key, default in defaults.items():
            value = property_data.get(key, default)
            if isinstance(value, str):
                value = value.strip().replace('|', '').strip()
            data[key] = value or default
        
        formatted_line = (
            f"{index}. unique id: {data['id']} | "
            f"property address: {data['address']} | "
            f"floor: {data['floor']} | "
            f"suite: {data['suite']} | "
            f"size (sf): {data['size']} | "
            f"rent/sf/year: {data['rent']} | "
            f"associate 1: {data['associate1']} | "
            f"associate 2: {data['associate2']} | "
            f"broker email id: {data['email']} | "
            f"phone: {data['phone']}"
        )
        
        return formatted_line
    
    async def call_openai_api(self, user_message: str,
                            conversation_history: List[ConversationMessage],
                            rag_context: List[Dict[str, Any]],
                            user_profile: Optional[UserProfile] = None) -> str:
        """Enhanced OpenAI API call with better context management"""
        
        system_prompt = self.build_system_prompt(rag_context, user_profile)
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add relevant conversation history (last 5 exchanges)
        for msg in conversation_history[-5:]:
            messages.append({"role": "user", "content": msg.user_message})
            messages.append({"role": "assistant", "content": msg.bot_response})
        
        messages.append({"role": "user", "content": user_message})
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        logger.info(f"Calling OpenAI API with model: {self.model}")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                logger.info("✅ OpenAI API call successful")
                return content
            
            elif response.status_code == 404 and "gpt-4o-mini" in str(response.text):
                logger.warning(f"Model {self.model} not available, trying gpt-3.5-turbo")
                payload["model"] = "gpt-3.5-turbo"
                
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    logger.info("✅ OpenAI API call successful with gpt-3.5-turbo")
                    return content
            
            logger.error(f"OpenAI API error {response.status_code}: {response.text}")
            raise Exception(f"API error: {response.status_code}")
            
        except requests.exceptions.Timeout:
            logger.error("OpenAI API timeout")
            raise Exception("API timeout")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            raise Exception(f"Request failed: {e}")
    
    async def extract_user_info(self, message: str) -> Dict[str, Any]:
        """Enhanced user information extraction"""
        extracted_info = {}
        
        # Extract email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, message)
        if emails:
            extracted_info['email'] = emails[0]
        
        # Extract phone numbers (enhanced patterns)
        phone_patterns = [
            r'(\+?1?[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',
            r'(\d{3})[-.](\d{3})[-.](\d{4})',
            r'\((\d{3})\)\s*(\d{3})[-.](\d{4})'
        ]
        
        for pattern in phone_patterns:
            phones = re.findall(pattern, message)
            if phones:
                if isinstance(phones[0], tuple):
                    extracted_info['phone'] = ''.join(phones[0])
                else:
                    extracted_info['phone'] = phones[0]
                break
        
        # Extract names (enhanced patterns)
        name_patterns = [
            r"(?:i'm|my name is|i am|this is)\s+([a-z]+(?:\s+[a-z]+)?)",
            r"name[:\s]+([a-z]+(?:\s+[a-z]+)?)",
            r"^([A-Z][a-z]+\s+[A-Z][a-z]+)"  # First Last pattern
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                extracted_info['name'] = match.group(1).title()
                break
        
        # Extract company information (enhanced)
        company_patterns = [
            r"(?:i work at|i'm from|my company is|i represent|company[:\s]+)\s+([a-z\s&.,0-9]+?)(?:\.|$|,|\s+(?:and|in|on))",
            r"(?:at|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)\s+(?:company|corp|inc|llc)",
        ]
        
        for pattern in company_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                company = match.group(1).strip()
                if len(company) > 2 and company not in ['the', 'and', 'for']:
                    extracted_info['company'] = company.title()
                    break
        
        # Extract enhanced preferences
        preferences = {}
        
        # Size preferences (multiple formats)
        size_patterns = [
            r"(?:need|looking for|want|require)\s+(?:about|around|approximately)?\s*(\d+)\s*(?:sq ft|square feet|sf)",
            r"(\d+)\s*(?:sq ft|square feet|sf)\s*(?:office|space|building)",
            r"(?:size|space)\s*(?:of|around|about)?\s*(\d+)\s*(?:sq ft|square feet|sf)"
        ]
        
        for pattern in size_patterns:
            match = re.search(pattern, message.lower())
            if match:
                preferences['preferred_size'] = int(match.group(1))
                break
        
        # Budget preferences (enhanced)
        budget_patterns = [
            r"budget\s*(?:is|of|around)?\s*(?:up to|under|below|around)?\s*\$?(\d+(?:,\d{3})*)",
            r"(?:can afford|willing to pay|max|maximum)\s*(?:up to)?\s*\$?(\d+(?:,\d{3})*)",
            r"\$(\d+(?:,\d{3})*)\s*(?:budget|max|maximum|per year|annually)"
        ]
        
        for pattern in budget_patterns:
            match = re.search(pattern, message.lower())
            if match:
                preferences['budget'] = match.group(1).replace(',', '')
                break
        
        # Location preferences (enhanced)
        location_patterns = [
            r"(?:in|near|around|at)\s+([a-z\s]+?)(?:\.|$|,|\s+(?:area|district|neighborhood|with))",
            r"location[:\s]*([a-z\s]+?)(?:\.|$|,)",
            r"(?:downtown|midtown|uptown)\s+([a-z\s]+?)(?:\.|$|,)"
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, message.lower())
            if match:
                location = match.group(1).strip()
                if len(location) > 2 and location not in ['the', 'and', 'or', 'but', 'with', 'for']:
                    preferences['preferred_location'] = location.title()
                    break
        
        # Amenity preferences
        amenities = []
        amenity_keywords = ['parking', 'garage', 'elevator', 'gym', 'fitness', 'restaurant', 'cafeteria', 'conference', 'meeting']
        
        for amenity in amenity_keywords:
            if amenity in message.lower():
                amenities.append(amenity)
        
        if amenities:
            preferences['preferred_amenities'] = amenities
        
        # Tenant status detection
        tenant_indicators = ['my unit', 'my apartment', 'my office', 'tenant', 'renter', 'lease', 'rent from you']
        if any(indicator in message.lower() for indicator in tenant_indicators):
            extracted_info['tenant_status'] = 'current_tenant'
        
        if preferences:
            extracted_info['preferences'] = preferences
        
        return extracted_info