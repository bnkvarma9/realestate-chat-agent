from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
from datetime import datetime, timedelta
import json
import uuid
import time
import threading
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import webbrowser
# Load environment variables
load_dotenv()

# Import our enhanced modules
from crm_system import CRMSystem
from rag_system import EnhancedRAGSystem
from llm_service import EnhancedLLMService
from appointment_system import EnhancedAppointmentSystem
from ai_insights import AIInsightsEngine
from models import (
    ChatRequest, ChatResponse, UserProfile, ConversationHistory
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize enhanced services on startup"""
    logger.info("Starting Enhanced Multi-Agentic Real Estate AI System v2.0...")
    
    # Initialize enhanced systems
    app.state.crm = CRMSystem()
    app.state.rag = EnhancedRAGSystem()
    app.state.llm = EnhancedLLMService()
    app.state.appointments = EnhancedAppointmentSystem()
    app.state.ai_insights = AIInsightsEngine()
    
    # Load initial data if CSV exists
    if os.path.exists("HackathonInternalKnowledgeBase.csv"):
        logger.info("Loading enhanced knowledge base...")
        await app.state.rag.load_csv_data("HackathonInternalKnowledgeBase.csv")
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down enhanced system...")

app = FastAPI(
    title="Enhanced Multi-Agentic Real Estate AI System",
    description="Advanced AI-powered real estate platform with voice support, multi-agent handling, and enhanced property search",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

@app.get("/")
async def root():
    return {
        "message": "Enhanced Multi-Agentic Real Estate AI System is running!", 
        "version": "2.0.0",
        "features": [
            "Voice-enabled chat interface",
            "Multi-agent conversation handling",
            "Enhanced property search with exact matching",
            "Maintenance and repair request handling",
            "Tenant support services",
            "Advanced appointment scheduling",
            "CRM integration with user preference tracking"
        ]
    }

@app.post("/chat", response_model=ChatResponse)
async def enhanced_chat(request: ChatRequest):
    """Enhanced chat endpoint with multi-agent routing and voice support"""
    try:
        start_time = datetime.now()
        
        # Get or create user session
        user_id = request.user_id or str(uuid.uuid4())
        
        # Get user profile from CRM
        user_profile = await app.state.crm.get_user_profile(user_id)
        
        # Get conversation history
        conversation_history = await app.state.crm.get_conversation_history(user_id)
        
        # Extract relevant context using enhanced RAG
        rag_context = await app.state.rag.get_relevant_context(
            query=request.message,
            top_k=10,
            user_id=user_id
        )
        
        # Generate response using enhanced LLM with multi-agent routing
        llm_response = await app.state.llm.generate_response(
            user_message=request.message,
            conversation_history=conversation_history,
            rag_context=rag_context,
            user_profile=user_profile
        )
        
        # Check if LLM wants to trigger appointment form
        if llm_response == "TRIGGER_APPOINTMENT_FORM":
            # Get user preferences and recent property interactions
            user_preferences = await app.state.rag.get_user_property_preferences(user_id)
            recent_property = await get_most_recent_property_mention(conversation_history)
            
            return ChatResponse(
                response="",  # Empty response, frontend will handle
                user_id=user_id,
                conversation_id=f"conv_{user_id}_{int(datetime.now().timestamp())}",
                suggested_actions=["open_appointment_form"],
                metadata={
                    "response_time": (datetime.now() - start_time).total_seconds(),
                    "trigger_appointment_form": True,
                    "suggested_property": recent_property,
                    "user_profile": {
                        "name": user_profile.name if user_profile else "",
                        "email": user_profile.email if user_profile else "",
                        "phone": user_profile.phone if user_profile else ""
                    },
                    "user_preferences": user_preferences
                }
            )
        
        # Enhanced user information extraction
        extracted_info = await app.state.llm.extract_user_info(request.message)
        
        if extracted_info:
            await app.state.crm.update_user_profile(user_id, extracted_info)
        
        # Generate enhanced AI insights
        user_insights = await app.state.ai_insights.generate_user_insights(
            user_profile, conversation_history, request.message
        )
        
        # Determine suggested actions based on message and context
        suggested_actions = await determine_suggested_actions(request.message, rag_context, user_profile)
        
        # Save conversation with enhanced metadata
        await app.state.crm.save_conversation(
            user_id=user_id,
            user_message=request.message,
            bot_response=llm_response,
            rag_context=rag_context,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "suggested_actions": suggested_actions,
                "ai_insights": user_insights,
                "user_info_extracted": extracted_info,
                "agent_type": determine_agent_type(request.message),
                "has_voice_support": True
            }
        )
        
        response_time = (datetime.now() - start_time).total_seconds()
        
        return ChatResponse(
            response=llm_response,
            user_id=user_id,
            conversation_id=f"conv_{user_id}_{int(datetime.now().timestamp())}",
            suggested_actions=suggested_actions,
            metadata={
                "response_time": response_time,
                "rag_sources": len(rag_context),
                "user_info_extracted": bool(extracted_info),
                "ai_insights": user_insights,
                "agent_type": determine_agent_type(request.message)
            }
        )
        
    except Exception as e:
        logger.error(f"Enhanced chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

# ENHANCED APPOINTMENT ENDPOINTS

@app.post("/appointments/schedule")
async def schedule_appointment(appointment_data: dict):
    """Enhanced appointment scheduling with maintenance support"""
    try:
        # Validate required fields
        required_fields = ['customer_name', 'customer_email']
        for field in required_fields:
            if not appointment_data.get(field):
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Check if user is a tenant (for maintenance requests)
        user_id = appointment_data.get('user_id')
        if user_id:
            tenant_info = await app.state.appointments.get_tenant_info(user_id)
            if tenant_info:
                appointment_data['tenant_id'] = tenant_info['tenant_id']
                appointment_data['unit_number'] = tenant_info['unit_number']
        
        # Create appointment with enhanced system
        appointment_id, confirmation_message = await app.state.appointments.create_appointment(appointment_data)
        
        # Find and assign appropriate staff
        if appointment_data.get('appointment_type') in ['maintenance', 'repair', 'emergency_repair']:
            staff = await app.state.appointments._assign_maintenance_staff(appointment_data, datetime.now())
        else:
            staff = await app.state.appointments._assign_broker(appointment_data)
        
        # Generate AI preparation notes
        preparation_notes = await app.state.ai_insights.generate_appointment_prep(
            appointment_data, staff
        )
        
        return {
            "success": True,
            "appointment_id": appointment_id,
            "message": confirmation_message,
            "assigned_staff": staff['name'],
            "staff_email": staff['email'],
            "staff_phone": staff['phone'],
            "staff_type": staff.get('type', 'staff'),
            "preparation_notes": preparation_notes
        }
        
    except ValueError as e:
        # Handle validation errors (conflicts, duplicates, etc.)
        error_message = str(e)
        
        if "conflict" in error_message.lower():
            return {
                "success": False,
                "error": "conflict",
                "message": error_message
            }
        
        raise HTTPException(status_code=400, detail=error_message)
        
    except Exception as e:
        logger.error(f"Enhanced appointment scheduling error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to schedule appointment: {str(e)}")

@app.get("/appointments/available-slots")
async def get_available_slots_enhanced(
    date: str = Query(..., description="Date in YYYY-MM-DD format"),
    staff_type: str = Query("broker", description="Type of staff (broker or maintenance)")
):
    """Get available time slots with enhanced logic for different staff types"""
    try:
        slots = await app.state.appointments.get_available_time_slots(date, staff_type)
        return {
            "available_slots": slots, 
            "date": date,
            "staff_type": staff_type
        }
    except Exception as e:
        logger.error(f"Get available slots error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch available slots: {str(e)}")

@app.get("/appointments/available-properties")
async def get_available_properties():
    """Get available properties for appointment scheduling dropdown"""
    try:
        properties = await app.state.rag.get_property_addresses_for_dropdown()
        return {"properties": properties}
    except Exception as e:
        logger.error(f"Get available properties error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch properties: {str(e)}")

@app.delete("/appointments/{appointment_id}")
async def cancel_appointment_enhanced(appointment_id: str, user_id: str = Query(...)):
    """Cancel an appointment with enhanced handling"""
    try:
        success, message = await app.state.appointments.cancel_appointment(appointment_id, user_id)
        return {"success": success, "message": message}
    except Exception as e:
        logger.error(f"Cancel appointment error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel appointment: {str(e)}")

@app.get("/appointments/user/{user_id}")
async def get_user_appointments_enhanced(user_id: str):
    """Get enhanced appointments for a user"""
    try:
        appointments = await app.state.appointments.get_user_appointments(user_id)
        return {
            "appointments": appointments,
            "total": len(appointments)
        }
    except Exception as e:
        logger.error(f"Get user appointments error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch appointments: {str(e)}")

# MAINTENANCE AND TENANT ENDPOINTS

@app.post("/maintenance/register-tenant")
async def register_tenant(tenant_data: dict):
    """Register a new tenant for maintenance requests"""
    try:
        tenant_id = await app.state.appointments.register_tenant(tenant_data)
        return {
            "success": True,
            "tenant_id": tenant_id,
            "message": "Tenant registered successfully"
        }
    except Exception as e:
        logger.error(f"Register tenant error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to register tenant: {str(e)}")

@app.get("/maintenance/history/{user_id}")
async def get_maintenance_history(user_id: str):
    """Get maintenance history for a tenant"""
    try:
        history = await app.state.appointments.get_maintenance_history(user_id)
        return {
            "maintenance_history": history,
            "total": len(history)
        }
    except Exception as e:
        logger.error(f"Get maintenance history error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch maintenance history: {str(e)}")

@app.post("/maintenance/escalate/{appointment_id}")
async def escalate_maintenance_request(appointment_id: str, escalation_data: dict):
    """Escalate a maintenance request to higher priority"""
    try:
        success = await app.state.appointments.escalate_maintenance_request(
            appointment_id, 
            escalation_data.get('reason', 'Customer requested escalation')
        )
        
        if success:
            return {
                "success": True,
                "message": "Maintenance request escalated successfully"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to escalate request")
            
    except Exception as e:
        logger.error(f"Escalate maintenance error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to escalate request: {str(e)}")

# ENHANCED USER AND CRM ENDPOINTS

@app.get("/crm/user/{user_id}")
async def get_user_profile_enhanced(user_id: str):
    """Get enhanced user profile with preferences and interaction history"""
    try:
        user_profile = await app.state.crm.get_user_profile(user_id)
        
        if not user_profile:
            # Create basic profile if none exists
            basic_profile = {
                "user_id": user_id,
                "name": "",
                "email": "",
                "phone": "",
                "company": "",
                "preferences": {},
                "created_at": datetime.now().isoformat()
            }
            return basic_profile
        
        # Get user preferences from RAG system
        user_preferences = await app.state.rag.get_user_property_preferences(user_id)
        
        # Convert UserProfile to dict and enhance
        profile_dict = {
            "user_id": user_profile.user_id,
            "name": user_profile.name,
            "email": user_profile.email,
            "phone": user_profile.phone,
            "company": user_profile.company,
            "preferences": {**(user_profile.preferences or {}), **user_preferences},
            "created_at": user_profile.created_at.isoformat() if user_profile.created_at else datetime.now().isoformat()
        }
        
        return profile_dict
        
    except Exception as e:
        logger.error(f"Get enhanced user profile error: {str(e)}")
        # Return basic profile on error
        return {
            "user_id": user_id,
            "name": "",
            "email": "", 
            "phone": "",
            "company": "",
            "preferences": {},
            "created_at": datetime.now().isoformat()
        }
@app.put("/appointments/{appointment_id}/update")
async def update_appointment(appointment_id: str, user_id: str = Query(...), updates: dict = Body(...)):
    """Update an existing appointment"""
    try:
        success, message = await app.state.appointments.update_appointment(appointment_id, user_id, updates)
        
        if success:
            return {
                "success": True,
                "message": message,
                "appointment_id": appointment_id
            }
        else:
            raise HTTPException(status_code=404, detail=message)
            
    except Exception as e:
        logger.error(f"Update appointment error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update appointment: {str(e)}")
@app.put("/crm/user/{user_id}/update")
async def update_user_profile_enhanced(user_id: str, user_data: dict):
    """Update user profile with enhanced data handling"""
    try:
        success = await app.state.crm.update_user_profile(user_id, user_data)
        
        if success:
            return {"message": "User profile updated successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to update user profile")
            
    except Exception as e:
        logger.error(f"Update enhanced user profile error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"User update failed: {str(e)}")

@app.get("/crm/conversations/{user_id}")
async def get_conversations_enhanced(user_id: str):
    """Fetch enhanced conversation history for a user"""
    try:
        conversations = await app.state.crm.get_conversation_history(user_id)
        
        # Add conversation analysis
        analysis = {
            "total_messages": len(conversations),
            "first_contact": conversations[0].timestamp.isoformat() if conversations else None,
            "last_contact": conversations[-1].timestamp.isoformat() if conversations else None,
            "engagement_level": determine_engagement_level(conversations),
            "topics_discussed": extract_conversation_topics(conversations),
            "user_intent_history": analyze_user_intents(conversations)
        }
        
        return {
            "user_id": user_id,
            "conversations": conversations,
            "total_messages": len(conversations),
            "analysis": analysis
        }
    except Exception as e:
        logger.error(f"Get enhanced conversations error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch conversations: {str(e)}")

@app.get("/crm/users")
async def get_all_users_enhanced():
    """Get all users with enhanced analytics"""
    try:
        users = await app.state.crm.get_all_users()
        
        enhanced_users = []
        for user in users:
            # Calculate enhanced lead score
            lead_score = await app.state.ai_insights.calculate_lead_score(user)
            
            # Get user interaction summary
            conversations = await app.state.crm.get_conversation_history(user.user_id)
            
            user_dict = {
                'user_id': getattr(user, 'user_id', ''),
                'name': getattr(user, 'name', ''),
                'email': getattr(user, 'email', ''),
                'company': getattr(user, 'company', ''),
                'phone': getattr(user, 'phone', ''),
                'preferences': getattr(user, 'preferences', {}),
                'created_at': getattr(user, 'created_at', datetime.now()).isoformat() if getattr(user, 'created_at', None) else None,
                'updated_at': getattr(user, 'updated_at', datetime.now()).isoformat() if getattr(user, 'updated_at', None) else None,
                'lead_score': lead_score,
                'total_conversations': len(conversations),
                'engagement_level': determine_engagement_level(conversations),
                'last_activity': conversations[-1].timestamp.isoformat() if conversations else None
            }
            
            enhanced_users.append(user_dict)
        
        return {"users": enhanced_users, "total_users": len(enhanced_users)}
    except Exception as e:
        logger.error(f"Get enhanced users error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch users: {str(e)}")

# DOCUMENT UPLOAD AND RAG ENDPOINTS

@app.post("/upload_docs")
async def upload_docs_enhanced(files: List[UploadFile] = File(...)):
    """Upload documents to populate the enhanced RAG knowledge base"""
    try:
        results = []
        
        for file in files:
            # Save uploaded file
            file_path = f"uploads/{file.filename}"
            
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Process file based on type with enhanced RAG
            if file.filename.endswith('.csv'):
                await app.state.rag.load_csv_data(file_path)
            elif file.filename.endswith('.txt'):
                await app.state.rag.load_text_data(file_path)
            elif file.filename.endswith('.json'):
                await app.state.rag.load_json_data(file_path)
            elif file.filename.endswith('.pdf'):
                await app.state.rag.load_pdf_data(file_path)
            
            results.append({
                "filename": file.filename,
                "size": len(content),
                "status": "processed",
                "type": file.filename.split('.')[-1]
            })
        
        return {"message": "Documents uploaded and processed successfully", "files": results}
        
    except Exception as e:
        logger.error(f"Enhanced upload docs error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Document upload failed: {str(e)}")

# ANALYTICS AND INSIGHTS ENDPOINTS

@app.get("/analytics/dashboard")
async def get_dashboard_analytics():
    """Get enhanced dashboard analytics"""
    try:
        # Get all users and conversations for analytics
        users = await app.state.crm.get_all_users()
        
        total_conversations = 0
        total_appointments = 0
        engagement_levels = {"high": 0, "medium": 0, "low": 0}
        
        for user in users:
            conversations = await app.state.crm.get_conversation_history(user.user_id)
            appointments = await app.state.appointments.get_user_appointments(user.user_id)
            
            total_conversations += len(conversations)
            total_appointments += len(appointments)
            
            engagement = determine_engagement_level(conversations)
            engagement_levels[engagement] += 1
        
        analytics = {
            "total_users": len(users),
            "total_conversations": total_conversations,
            "total_appointments": total_appointments,
            "engagement_distribution": engagement_levels,
            "avg_conversations_per_user": total_conversations / len(users) if users else 0,
            "system_health": {
                "api_status": "active",
                "rag_documents": 225,  # From CSV
                "voice_support": True,
                "multi_agent": True
            }
        }
        
        return analytics
        
    except Exception as e:
        logger.error(f"Get dashboard analytics error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch analytics: {str(e)}")

@app.get("/health")
async def enhanced_health_check():
    """Enhanced health check endpoint with detailed system status"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "services": {
            "crm": "active",
            "rag": "active",
            "llm": "active", 
            "appointments": "active",
            "ai_insights": "active",
            "maintenance": "active"
        },
        "features": [
            "Multi-agent conversation routing",
            "Enhanced RAG with exact property matching", 
            "Voice-enabled chat interface",
            "Maintenance and repair request handling",
            "Tenant support services",
            "Advanced appointment scheduling",
            "Real-time property search",
            "CRM integration with user tracking",
            "AI-powered insights and recommendations"
        ],
        "capabilities": {
            "voice_recognition": True,
            "text_to_speech": True,
            "exact_property_matching": True,
            "maintenance_requests": True,
            "multi_agent_routing": True,
            "real_time_scheduling": True
        }
    }

# UTILITY FUNCTIONS

async def get_most_recent_property_mention(conversation_history: List) -> Optional[str]:
    """Extract the most recently mentioned property from conversation history"""
    try:
        for message in reversed(conversation_history[-10:]):  # Check last 10 messages
            if 'property address:' in message.bot_response:
                import re
                property_match = re.search(r'property address:\s*([^|]+)', message.bot_response)
                if property_match:
                    return property_match.group(1).strip()
        return None
    except Exception:
        return None

async def determine_suggested_actions(message: str, rag_context: List, user_profile) -> List[str]:
    """Determine suggested actions based on message context"""
    actions = []
    message_lower = message.lower()
    
    # Property interest indicators
    if any(word in message_lower for word in ['interested', 'perfect', 'love', 'great', 'excellent']):
        actions.append('schedule_property_visit')
    
    # Pricing interest
    if any(word in message_lower for word in ['price', 'cost', 'budget', 'rent', 'lease']):
        actions.append('send_pricing_proposal')
    
    # Contact requests
    if any(word in message_lower for word in ['contact', 'speak', 'talk', 'call', 'broker']):
        actions.append('contact_broker')
    
    # Maintenance requests
    if any(word in message_lower for word in ['maintenance', 'repair', 'fix', 'broken', 'issue']):
        actions.append('schedule_maintenance')
    
    return actions

def determine_agent_type(message: str) -> str:
    """Determine which agent type should handle the message"""
    message_lower = message.lower()
    
    if any(word in message_lower for word in ['maintenance', 'repair', 'fix', 'broken', 'tenant']):
        return 'maintenance_agent'
    elif any(word in message_lower for word in ['schedule', 'appointment', 'book', 'visit']):
        return 'appointment_scheduler'
    elif any(word in message_lower for word in ['property', 'office', 'space', 'rent', 'lease']):
        return 'property_agent'
    else:
        return 'customer_service_agent'

def determine_engagement_level(conversations: List) -> str:
    """Determine user engagement level based on conversation history"""
    if not conversations:
        return "low"
    
    message_count = len(conversations)
    
    if message_count >= 10:
        return "high"
    elif message_count >= 5:
        return "medium"
    else:
        return "low"

def extract_conversation_topics(conversations: List) -> List[str]:
    """Extract main topics from conversation history"""
    topics = set()
    
    for conv in conversations:
        message_text = conv.user_message.lower()
        
        if any(word in message_text for word in ['property', 'office', 'space']):
            topics.add('Property Search')
        if any(word in message_text for word in ['price', 'cost', 'budget']):
            topics.add('Pricing')
        if any(word in message_text for word in ['maintenance', 'repair']):
            topics.add('Maintenance')
        if any(word in message_text for word in ['schedule', 'appointment']):
            topics.add('Scheduling')
        if any(word in message_text for word in ['location', 'downtown', 'midtown']):
            topics.add('Location')
    
    return list(topics)

def analyze_user_intents(conversations: List) -> Dict[str, int]:
    """Analyze user intents from conversation history"""
    intents = {
        'property_search': 0,
        'appointment_scheduling': 0,
        'maintenance_request': 0,
        'general_inquiry': 0
    }
    
    for conv in conversations:
        message = conv.user_message.lower()
        
        if any(word in message for word in ['property', 'office', 'space', 'find', 'search']):
            intents['property_search'] += 1
        elif any(word in message for word in ['schedule', 'appointment', 'book', 'visit']):
            intents['appointment_scheduling'] += 1
        elif any(word in message for word in ['maintenance', 'repair', 'fix', 'broken']):
            intents['maintenance_request'] += 1
        else:
            intents['general_inquiry'] += 1
    
    return intents

# LEGACY ENDPOINTS (for backward compatibility)

@app.post("/reset")
async def reset_conversation(user_id: Optional[str] = None):
    """Clear conversation memory (optional: per user)"""
    try:
        if user_id:
            await app.state.crm.clear_user_conversations(user_id)
            return {"message": f"Conversation memory cleared for user {user_id}"}
        else:
            await app.state.crm.clear_all_conversations()
            return {"message": "All conversation memory cleared"}
    except Exception as e:
        logger.error(f"Reset conversation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")
@app.get("/crm/properties/dropdown")
async def get_properties_for_dropdown():
    """Get all properties for dropdown/autocomplete"""
    try:
        properties = await app.state.rag.get_property_addresses_for_dropdown()
        return {"properties": properties}
    except Exception as e:
        logger.error(f"Get properties dropdown error: {str(e)}")
        # Fallback to basic property list
        return {
            "properties": [
                {"id": "1", "address": "123 Main St", "display": "123 Main St - 2500 sq ft"},
                {"id": "2", "address": "456 Business Ave", "display": "456 Business Ave - 3200 sq ft"},
                {"id": "3", "address": "789 Corporate Blvd", "display": "789 Corporate Blvd - 18650 sq ft"}
            ]
        }

# Enhanced LLM service with better user info extraction
def enhance_llm_user_extraction():
    """Add this method to your EnhancedLLMService class in llm_service.py"""
    
    async def extract_user_info_with_openai(self, message: str) -> Dict[str, Any]:
        """Use OpenAI to intelligently extract user information"""
        if not self.api_key:
            return await self.extract_user_info(message)  # Fallback to regex
        
        try:
            extraction_prompt = f"""
            Extract personal information from this message. Only extract clear, explicit information.
            Do NOT extract question words, greetings, or partial phrases as names.
            
            Message: "{message}"
            
            Extract:
            - name (only if clearly stated like "I'm John" or "My name is Sarah", NOT from questions like "can you")
            - email (valid email addresses)
            - phone (phone numbers)
            - company (company names mentioned as "I work at X" or "from X company")
            
            Return JSON with only the fields that are clearly present:
            """
            
            response = await self.call_openai_for_extraction(extraction_prompt, message)
            
            if response:
                try:
                    import json
                    extracted = json.loads(response)
                    # Validate extracted name - reject if it's a question word
                    if 'name' in extracted:
                        name = extracted['name'].lower()
                        question_words = ['can', 'could', 'would', 'will', 'do', 'does', 'how', 'what', 'when', 'where', 'why']
                        if any(word in name for word in question_words) or len(name) < 2:
                            del extracted['name']
                    
                    return extracted
                except json.JSONDecodeError:
                    pass
            
            # Fallback to regex extraction
            return await self.extract_user_info(message)
            
        except Exception as e:
            logger.error(f"OpenAI user extraction error: {str(e)}")
            return await self.extract_user_info(message)
    
    async def call_openai_for_extraction(self, prompt: str, message: str) -> str:
        """Call OpenAI for user info extraction"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": message}
                ],
                "max_tokens": 200,
                "temperature": 0.1
            }
            
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            
            return None
            
        except Exception as e:
            logger.error(f"OpenAI extraction API error: {str(e)}")
            return None

# Updated Enhanced Chat Endpoint with Fixes and Improvements
@app.post("/chat", response_model=ChatResponse)
async def enhanced_chat_with_history(request: ChatRequest):
    """Enhanced chat endpoint with better name extraction, accurate property mapping, and appointment control."""
    try:
        start_time = datetime.now()

        # Generate or get user ID
        user_id = request.user_id or str(uuid.uuid4())

        # Retrieve user profile and conversation history
        user_profile = await app.state.crm.get_user_profile(user_id)
        conversation_history = await app.state.crm.get_conversation_history(user_id, limit=10)

        # Extract context using RAG
        rag_context = await app.state.rag.get_relevant_context(
            query=request.message,
            top_k=15,
            user_id=user_id
        )

        # More robust name extraction
        extracted_info = {}
        if hasattr(app.state.llm, 'extract_user_info_with_openai'):
            extracted_info = await app.state.llm.extract_user_info_with_openai(request.message)
        else:
            extracted_info = await app.state.llm.extract_user_info(request.message)

        # Avoid setting invalid names like "Already A"
        if extracted_info.get("name") and len(extracted_info["name"].split()) < 2:
            extracted_info.pop("name")

        # Update user profile if meaningful info is found
        if extracted_info:
            await app.state.crm.update_user_profile(user_id, extracted_info)
            user_profile = await app.state.crm.get_user_profile(user_id)

        # Build context from last 5 exchanges
        conversation_context = ""
        if conversation_history:
            conversation_context = "\n\nPrevious conversation context:\n"
            for msg in conversation_history[-5:]:
                conversation_context += f"User: {msg.user_message}\nAssistant: {msg.bot_response[:200]}...\n"

        # Add user profile and context to message
        enhanced_message = request.message
        if user_profile:
            user_context = f"\nUser profile: Name: {user_profile.name or 'Unknown'}, Email: {user_profile.email or 'Unknown'}"
            if user_profile.preferences:
                user_context += f", Preferences: {user_profile.preferences}"
            enhanced_message += user_context + conversation_context

        # Generate LLM response
        llm_response = await app.state.llm.generate_response(
            user_message=enhanced_message,
            conversation_history=conversation_history,
            rag_context=rag_context,
            user_profile=user_profile
        )

        # Appointment flow handling
        if llm_response == "TRIGGER_APPOINTMENT_FORM":
            user_preferences = await app.state.rag.get_user_property_preferences(user_id)
            recent_property = await get_most_recent_property_mention(conversation_history)
            if not recent_property:
                recent_property = await app.state.rag.extract_property_from_message(request.message)

            # Save appointment
            appointment = await app.state.crm.save_appointment(user_id, {
                "property": recent_property,
                "datetime": parse_appointment_datetime(request.message),
                "name": user_profile.name,
                "email": user_profile.email
            })

            return ChatResponse(
                response="",
                user_id=user_id,
                conversation_id=f"conv_{user_id}_{int(datetime.now().timestamp())}",
                suggested_actions=["open_appointment_form"],
                metadata={
                    "response_time": (datetime.now() - start_time).total_seconds(),
                    "trigger_appointment_form": True,
                    "suggested_property": recent_property,
                    "user_profile": {
                        "name": user_profile.name or "",
                        "email": user_profile.email or "",
                        "phone": user_profile.phone or ""
                    },
                    "user_preferences": user_preferences
                }
            )

        # Determine next actions
        suggested_actions = await determine_suggested_actions(request.message, rag_context, user_profile)

        # Save chat
        await app.state.crm.save_conversation(
            user_id=user_id,
            user_message=request.message,
            bot_response=llm_response,
            rag_context=rag_context,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "suggested_actions": suggested_actions,
                "user_info_extracted": extracted_info,
                "agent_type": determine_agent_type(request.message),
                "has_conversation_history": len(conversation_history) > 0,
                "personalized": bool(user_profile and user_profile.preferences)
            }
        )

        response_time = (datetime.now() - start_time).total_seconds()

        return ChatResponse(
            response=llm_response,
            user_id=user_id,
            conversation_id=f"conv_{user_id}_{int(datetime.now().timestamp())}",
            suggested_actions=suggested_actions,
            metadata={
                "response_time": response_time,
                "rag_sources": len(rag_context),
                "user_info_extracted": bool(extracted_info),
                "agent_type": determine_agent_type(request.message),
                "conversation_length": len(conversation_history),
                "personalized": bool(user_profile and user_profile.preferences)
            }
        )

    except Exception as e:
        logger.error(f"Enhanced chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


# Enhanced recommendations endpoint
@app.get("/recommendations/{user_id}")
async def get_personalized_recommendations(user_id: str):
    """Get personalized property recommendations based on user history"""
    try:
        # Get user profile and conversation history
        user_profile = await app.state.crm.get_user_profile(user_id)
        conversation_history = await app.state.crm.get_conversation_history(user_id)
        
        if not conversation_history:
            return {"message": "No conversation history found for personalized recommendations"}
        
        # Analyze user preferences from conversation history
        user_interests = []
        mentioned_sizes = []
        mentioned_locations = []
        
        for msg in conversation_history:
            content = msg.user_message.lower()
            
            # Extract size preferences
            import re
            size_matches = re.findall(r'(\d+)\s*(?:sq\s*ft|square\s*feet|sf)', content)
            mentioned_sizes.extend([int(size) for size in size_matches])
            
            # Extract location preferences
            locations = ['downtown', 'midtown', 'uptown', 'manhattan', 'brooklyn']
            for location in locations:
                if location in content:
                    mentioned_locations.append(location)
            
            # Extract interest indicators
            if any(word in content for word in ['interested', 'like', 'perfect', 'good']):
                user_interests.append(msg.bot_response)
        
        # Get relevant properties based on user history
        if mentioned_sizes:
            avg_size = sum(mentioned_sizes) / len(mentioned_sizes)
            size_query = f"properties around {int(avg_size)} square feet"
        else:
            size_query = "office properties"
        
        if mentioned_locations:
            location_query = f"{mentioned_locations[0]} {size_query}"
        else:
            location_query = size_query
        
        # Get recommendations using RAG
        recommendations = await app.state.rag.get_relevant_context(
            query=location_query,
            top_k=15,
            user_id=user_id
        )
        
        return {
            "user_id": user_id,
            "recommendations": recommendations,
            "based_on": {
                "conversation_count": len(conversation_history),
                "size_preferences": mentioned_sizes,
                "location_preferences": mentioned_locations,
                "interests": len(user_interests)
            },
            "message": f"Based on your {len(conversation_history)} previous conversations, here are our recommendations:"
        }
        
    except Exception as e:
        logger.error(f"Get recommendations error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")
@app.post("/appointments/schedule")
async def schedule_appointment_enhanced(appointment_data: dict):
    """Enhanced appointment scheduling with better error handling"""
    try:
        # Validate required fields
        required_fields = ['customer_name', 'customer_email']
        for field in required_fields:
            if not appointment_data.get(field):
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Clean and validate data
        appointment_data['customer_name'] = str(appointment_data.get('customer_name', '')).strip()
        appointment_data['customer_email'] = str(appointment_data.get('customer_email', '')).strip()
        
        # Handle property address safely
        if appointment_data.get('property_address'):
            appointment_data['property_address'] = str(appointment_data['property_address']).strip()
        
        # Handle notes safely
        if appointment_data.get('notes'):
            appointment_data['notes'] = str(appointment_data['notes']).strip()
        
        # Combine date and time properly
        if appointment_data.get('appointment_date') and appointment_data.get('appointment_time'):
            date_str = appointment_data['appointment_date']
            time_str = appointment_data['appointment_time']
            appointment_data['preferred_date'] = f"{date_str} {time_str}"
        
        # Check if user is a tenant (for maintenance requests)
        user_id = appointment_data.get('user_id')
        if user_id:
            try:
                tenant_info = await app.state.appointments.get_tenant_info(user_id)
                if tenant_info:
                    appointment_data['tenant_id'] = tenant_info['tenant_id']
                    appointment_data['unit_number'] = tenant_info['unit_number']
            except Exception as tenant_error:
                logger.warning(f"Could not fetch tenant info: {tenant_error}")
        
        # Create appointment with enhanced system
        appointment_id, confirmation_message = await app.state.appointments.create_appointment(appointment_data)
        
        # Find and assign appropriate staff (FIXED: removed extra datetime argument)
        appointment_type = appointment_data.get('appointment_type', 'site_visit')
        if appointment_type in ['maintenance', 'repair', 'emergency_repair']:
            staff = await app.state.appointments._assign_maintenance_staff(
                appointment_data, 
                datetime.now()
            )
        else:
            # FIXED: Only pass appointment_data, no datetime
            staff = await app.state.appointments._assign_broker(appointment_data)
        
        # Generate preparation notes if AI insights available
        preparation_notes = ""
        try:
            if hasattr(app.state, 'ai_insights'):
                preparation_notes = await app.state.ai_insights.generate_appointment_prep(
                    appointment_data, staff
                )
        except Exception as ai_error:
            logger.warning(f"Could not generate AI preparation notes: {ai_error}")
            preparation_notes = "Standard appointment preparation recommended."
        
        return {
            "success": True,
            "appointment_id": appointment_id,
            "message": confirmation_message,
            "assigned_staff": staff['name'],
            "staff_email": staff['email'],
            "staff_phone": staff['phone'],
            "staff_type": staff.get('type', 'staff'),
            "preparation_notes": preparation_notes
        }
        
    except ValueError as e:
        # Handle validation errors (conflicts, duplicates, etc.)
        error_message = str(e)
        
        if "conflict" in error_message.lower():
            return {
                "success": False,
                "error": "conflict",
                "message": error_message
            }
        
        raise HTTPException(status_code=400, detail=error_message)
        
    except Exception as e:
        logger.error(f"Enhanced appointment scheduling error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to schedule appointment: {str(e)}")

@app.post("/appointments/schedule-simple")
async def schedule_appointment_simple(appointment_data: dict):
    """Simplified appointment scheduling for debugging"""
    try:
        # Basic validation
        if not appointment_data.get('customer_name'):
            raise HTTPException(status_code=400, detail="Customer name is required")
        if not appointment_data.get('customer_email'):
            raise HTTPException(status_code=400, detail="Customer email is required")
        
        # Generate appointment ID
        appointment_id = str(uuid.uuid4())
        
        # Default staff assignment
        default_staff = {
            'staff_id': 'staff_001',
            'name': 'John Smith',
            'email': 'john.smith@company.com',
            'phone': '(555) 123-4567',
            'type': 'broker'
        }
        
        # Simple confirmation message
        confirmation_message = f"Appointment scheduled for {appointment_data.get('customer_name')} on {appointment_data.get('appointment_date', 'TBD')}"
        
        return {
            "success": True,
            "appointment_id": appointment_id,
            "message": confirmation_message,
            "assigned_staff": default_staff['name'],
            "staff_email": default_staff['email'],
            "staff_phone": default_staff['phone'],
            "staff_type": default_staff['type']
        }
        
    except Exception as e:
        logger.error(f"Simple appointment scheduling error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to schedule appointment: {str(e)}")

# Conversation analytics endpoint
@app.get("/analytics/conversations/{user_id}")
async def get_conversation_analytics(user_id: str):
    """Get detailed conversation analytics for a user"""
    try:
        conversations = await app.state.crm.get_conversation_history(user_id, limit=100)
        
        if not conversations:
            return {"message": "No conversation data found"}
        
        # Analyze conversation patterns
        topics = {}
        sentiment_scores = []
        response_times = []
        
        for msg in conversations:
            # Topic analysis
            content = msg.user_message.lower()
            if 'property' in content or 'office' in content:
                topics['property_search'] = topics.get('property_search', 0) + 1
            elif 'appointment' in content or 'schedule' in content:
                topics['appointments'] = topics.get('appointments', 0) + 1
            elif 'maintenance' in content or 'repair' in content:
                topics['maintenance'] = topics.get('maintenance', 0) + 1
            else:
                topics['general'] = topics.get('general', 0) + 1
            
            # Extract response times if available
            if msg.metadata and 'response_time' in msg.metadata:
                response_times.append(msg.metadata['response_time'])
        
        return {
            "user_id": user_id,
            "total_conversations": len(conversations),
            "topics_discussed": topics,
            "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
            "engagement_level": determine_engagement_level(conversations),
            "first_contact": conversations[0].timestamp.isoformat() if conversations else None,
            "last_contact": conversations[-1].timestamp.isoformat() if conversations else None
        }
        
    except Exception as e:
        logger.error(f"Conversation analytics error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")
@app.get("/chatbot", response_class=HTMLResponse)
async def root():
    with open("chatbot_frontend.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/broker", response_class=HTMLResponse)
async def broker_dashboard():
    with open("broker_dashboard.html", "r", encoding="utf-8") as f:
        return f.read()

# Function to open browser after server starts
def open_browser():
    time.sleep(60)  # Wait for server to start
    webbrowser.open("http://127.0.0.1:8000/chatbot")             # Opens chatbot_frontend.html
    webbrowser.open("http://127.0.0.1:8000/broker")        # Opens broker_dashboard.html
if __name__ == "__main__":
    
    threading.Thread(target=open_browser).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)