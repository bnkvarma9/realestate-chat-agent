from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class ConversationStatus(str, Enum):
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    INQUIRING = "inquiring"
    FOLLOW_UP = "follow_up"
    HOT_LEAD = "hot_lead"
    SCHEDULED = "scheduled"

class ConversationCategory(str, Enum):
    REAL_ESTATE = "real_estate"
    GENERAL = "general"
    SUPPORT = "support"
    SALES = "sales"
    INFORMATION = "information"
    APPOINTMENT_REQUEST = "appointment_request"

class LeadScore(str, Enum):
    HOT = "hot"           # 80-100 points
    WARM = "warm"         # 60-79 points  
    COLD = "cold"         # 40-59 points
    INACTIVE = "inactive" # 0-39 points

class AppointmentType(str, Enum):
    SITE_VISIT = "site_visit"
    VIRTUAL_TOUR = "virtual_tour"
    CONSULTATION = "consultation"
    FOLLOW_UP = "follow_up"
    SIGNING = "signing"

class AppointmentStatus(str, Enum):
    SCHEDULED = "scheduled"
    CONFIRMED = "confirmed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    NO_SHOW = "no_show"
    RESCHEDULED = "rescheduled"

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class TaskPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

# Basic models
class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    user_id: Optional[str] = Field(None, description="User ID for session tracking")
    session_id: Optional[str] = Field(None, description="Session ID")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")

class ChatResponse(BaseModel):
    response: str = Field(..., description="AI response")
    user_id: str = Field(..., description="User ID")
    conversation_id: str = Field(..., description="Conversation ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")
    suggested_actions: Optional[List[str]] = Field(default_factory=list, description="AI suggested next actions")

class UserProfile(BaseModel):
    user_id: Optional[str] = Field(None, description="Unique user identifier")
    name: Optional[str] = Field(None, description="User's name")
    email: Optional[str] = Field(None, description="User's email")
    company: Optional[str] = Field(None, description="User's company")
    phone: Optional[str] = Field(None, description="User's phone number")
    preferences: Optional[Dict[str, Any]] = Field(default_factory=dict, description="User preferences")
    tags: Optional[List[str]] = Field(default_factory=list, description="User tags")
    lead_score: Optional[int] = Field(0, description="Lead scoring 0-100")
    lead_status: Optional[LeadScore] = Field(LeadScore.COLD, description="Lead temperature")
    created_at: Optional[datetime] = Field(None, description="Profile creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Profile update timestamp")
    last_contact: Optional[datetime] = Field(None, description="Last contact timestamp")
    assigned_broker: Optional[str] = Field(None, description="Assigned broker ID")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ConversationMessage(BaseModel):
    message_id: str = Field(..., description="Unique message identifier")
    user_id: str = Field(..., description="User ID")
    user_message: str = Field(..., description="User's message")
    bot_response: str = Field(..., description="Bot's response")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")
    rag_context: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="RAG context used")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    sentiment_score: Optional[float] = Field(None, description="Message sentiment score -1 to 1")
    extracted_entities: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Extracted entities")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# MISSING MODEL - ConversationHistory
class ConversationHistory(BaseModel):
    user_id: str = Field(..., description="User ID")
    messages: List[ConversationMessage] = Field(default_factory=list, description="List of messages")
    status: ConversationStatus = Field(default=ConversationStatus.INQUIRING, description="Conversation status")
    category: ConversationCategory = Field(default=ConversationCategory.GENERAL, description="Conversation category")
    created_at: datetime = Field(default_factory=datetime.now, description="Conversation start time")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update time")
    total_messages: int = Field(0, description="Total number of messages")
    last_activity: Optional[datetime] = Field(None, description="Last activity timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# Appointment models
class AppointmentRequest(BaseModel):
    customer_name: str = Field(..., description="Customer name")
    customer_email: str = Field(..., description="Customer email")
    customer_phone: Optional[str] = Field(None, description="Customer phone")
    property_id: Optional[str] = Field(None, description="Property ID of interest")
    property_address: Optional[str] = Field(None, description="Property address")
    preferred_date: datetime = Field(..., description="Preferred appointment date/time")
    alternative_date: Optional[datetime] = Field(None, description="Alternative date/time")
    appointment_type: AppointmentType = Field(AppointmentType.SITE_VISIT, description="Type of appointment")
    message: Optional[str] = Field(None, description="Additional message from customer")
    broker_id: Optional[str] = Field(None, description="Requested broker ID")

class Appointment(BaseModel):
    appointment_id: str = Field(..., description="Unique appointment identifier")
    user_id: str = Field(..., description="Customer user ID")
    broker_id: str = Field(..., description="Assigned broker ID")
    property_id: Optional[str] = Field(None, description="Property ID")
    property_address: Optional[str] = Field(None, description="Property address")
    
    # Appointment details
    appointment_type: AppointmentType = Field(..., description="Type of appointment")
    status: AppointmentStatus = Field(AppointmentStatus.SCHEDULED, description="Appointment status")
    scheduled_datetime: datetime = Field(..., description="Scheduled date and time")
    duration_minutes: int = Field(60, description="Expected duration in minutes")
    
    # Customer information
    customer_name: str = Field(..., description="Customer name")
    customer_email: str = Field(..., description="Customer email")
    customer_phone: Optional[str] = Field(None, description="Customer phone")
    
    # Appointment content
    agenda: Optional[List[str]] = Field(default_factory=list, description="Meeting agenda items")
    notes: Optional[str] = Field(None, description="Additional notes")
    preparation_notes: Optional[str] = Field(None, description="AI-generated preparation notes")
    
    # Tracking
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    confirmed_at: Optional[datetime] = Field(None, description="Confirmation timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    
    # Follow-up
    follow_up_required: bool = Field(True, description="Whether follow-up is required")
    outcome: Optional[str] = Field(None, description="Appointment outcome")
    next_steps: Optional[List[str]] = Field(default_factory=list, description="Agreed next steps")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class Task(BaseModel):
    task_id: str = Field(..., description="Unique task identifier")
    title: str = Field(..., description="Task title")
    description: Optional[str] = Field(None, description="Task description")
    
    # Assignment
    assigned_to: str = Field(..., description="Broker/user ID assigned to task")
    created_by: str = Field(..., description="Who created the task")
    customer_id: Optional[str] = Field(None, description="Related customer ID")
    appointment_id: Optional[str] = Field(None, description="Related appointment ID")
    property_id: Optional[str] = Field(None, description="Related property ID")
    
    # Status and priority
    status: TaskStatus = Field(TaskStatus.PENDING, description="Task status")
    priority: TaskPriority = Field(TaskPriority.MEDIUM, description="Task priority")
    
    # Timing
    due_date: Optional[datetime] = Field(None, description="Task due date")
    estimated_duration: Optional[int] = Field(None, description="Estimated duration in minutes")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    
    # Content
    checklist: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Task checklist items")
    attachments: Optional[List[str]] = Field(default_factory=list, description="File attachments")
    tags: Optional[List[str]] = Field(default_factory=list, description="Task tags")
    
    # AI-generated
    ai_suggestions: Optional[List[str]] = Field(default_factory=list, description="AI suggestions for task completion")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class BrokerProfile(BaseModel):
    broker_id: str = Field(..., description="Unique broker identifier")
    name: str = Field(..., description="Broker name")
    email: str = Field(..., description="Broker email")
    phone: Optional[str] = Field(None, description="Broker phone")
    
    # Professional info
    license_number: Optional[str] = Field(None, description="Real estate license number")
    specializations: Optional[List[str]] = Field(default_factory=list, description="Areas of specialization")
    experience_years: Optional[int] = Field(None, description="Years of experience")
    
    # Performance metrics
    total_deals: int = Field(0, description="Total deals closed")
    deals_this_month: int = Field(0, description="Deals closed this month")
    average_deal_size: Optional[float] = Field(None, description="Average deal size")
    customer_satisfaction: Optional[float] = Field(None, description="Customer satisfaction score")
    
    # Availability
    is_active: bool = Field(True, description="Whether broker is active")
    max_appointments_per_day: int = Field(8, description="Maximum appointments per day")
    working_hours: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Working hours schedule")
    
    # Preferences
    preferred_property_types: Optional[List[str]] = Field(default_factory=list, description="Preferred property types")
    territory: Optional[List[str]] = Field(default_factory=list, description="Coverage areas")
    
    created_at: datetime = Field(default_factory=datetime.now, description="Profile creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Profile update timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class CustomerInsight(BaseModel):
    insight_id: str = Field(..., description="Unique insight identifier")
    user_id: str = Field(..., description="Customer user ID")
    insight_type: str = Field(..., description="Type of insight")
    title: str = Field(..., description="Insight title")
    description: str = Field(..., description="Insight description")
    confidence_score: float = Field(..., description="AI confidence score 0-1")
    priority: str = Field(..., description="Insight priority")
    generated_at: datetime = Field(default_factory=datetime.now, description="Generation timestamp")
    is_actionable: bool = Field(True, description="Whether insight requires action")
    suggested_actions: Optional[List[str]] = Field(default_factory=list, description="Suggested actions")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class PropertyInteraction(BaseModel):
    interaction_id: str = Field(..., description="Unique interaction identifier")
    user_id: str = Field(..., description="Customer user ID")
    property_id: str = Field(..., description="Property ID")
    interaction_type: str = Field(..., description="Type of interaction")
    timestamp: datetime = Field(default_factory=datetime.now, description="Interaction timestamp")
    duration_seconds: Optional[int] = Field(None, description="Duration of interaction")
    details: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Interaction details")
    sentiment: Optional[str] = Field(None, description="Customer sentiment")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# Response models for API endpoints
class AppointmentResponse(BaseModel):
    appointment: Appointment
    ai_preparation_notes: Optional[str] = None
    customer_insights: Optional[List[CustomerInsight]] = None

class DashboardStats(BaseModel):
    total_leads: int
    scheduled_visits: int
    deals_closed: int
    revenue: float
    leads_this_week: int
    conversion_rate: float
    average_response_time: float
    top_performing_properties: List[Dict[str, Any]]

class AIRecommendation(BaseModel):
    recommendation_id: str
    user_id: Optional[str] = None
    recommendation_type: str
    title: str
    description: str
    confidence_score: float
    estimated_impact: str
    suggested_actions: List[str]
    expires_at: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# Additional models for RAG and CRM
class RAGDocument(BaseModel):
    doc_id: str = Field(..., description="Document ID")
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    embeddings: Optional[List[float]] = Field(None, description="Document embeddings")
    created_at: datetime = Field(default_factory=datetime.now, description="Document creation time")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class RAGQueryResult(BaseModel):
    query: str = Field(..., description="Original query")
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Retrieved documents")
    scores: List[float] = Field(default_factory=list, description="Relevance scores")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Query metadata")

class PropertyListing(BaseModel):
    """Model for real estate property data from CSV"""
    unique_id: int
    property_address: str
    floor: Optional[str] = None
    suite: Optional[str] = None
    size_sf: Optional[int] = None
    rent_sf_year: Optional[str] = None
    associate_1: Optional[str] = None
    broker_email: Optional[str] = None
    associate_2: Optional[str] = None
    associate_3: Optional[str] = None
    associate_4: Optional[str] = None
    annual_rent: Optional[str] = None
    monthly_rent: Optional[str] = None
    gci_on_3_years: Optional[str] = None

class CalendarEvent(BaseModel):
    """Optional calendar integration model"""
    event_id: str = Field(..., description="Event ID")
    user_id: str = Field(..., description="User ID")
    title: str = Field(..., description="Event title")
    description: Optional[str] = Field(None, description="Event description")
    start_time: datetime = Field(..., description="Event start time")
    end_time: datetime = Field(..., description="Event end time")
    location: Optional[str] = Field(None, description="Event location")
    attendees: Optional[List[str]] = Field(default_factory=list, description="Event attendees")
    created_at: datetime = Field(default_factory=datetime.now, description="Event creation time")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class SystemMetrics(BaseModel):
    """System performance metrics"""
    total_users: int = Field(default=0, description="Total number of users")
    total_conversations: int = Field(default=0, description="Total number of conversations")
    total_messages: int = Field(default=0, description="Total number of messages")
    avg_response_time: float = Field(default=0.0, description="Average response time")
    rag_documents: int = Field(default=0, description="Number of RAG documents")
    active_sessions: int = Field(default=0, description="Active user sessions")
    last_updated: datetime = Field(default_factory=datetime.now, description="Metrics last updated")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }