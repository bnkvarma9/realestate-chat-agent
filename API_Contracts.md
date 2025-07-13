# API Contracts Documentation

## Base URL
```
http://localhost:8000
```

## Authentication
Currently no authentication required. All endpoints are open for hackathon demo purposes.

## Content Types
- Request: `application/json`
- Response: `application/json`
- File Upload: `multipart/form-data`

---

## Core Endpoints

### 1. Chat Endpoint

**Endpoint:** `POST /chat`

**Description:** Main conversational interface with RAG and CRM integration

**Request Format:**
```json
{
    "message": "string (required)",
    "user_id": "string (optional)",
    "session_id": "string (optional)",
    "context": {
        "key": "value (optional)"
    }
}
```

**Response Format:**
```json
{
    "response": "string",
    "user_id": "string",
    "conversation_id": "string",
    "metadata": {
        "response_time": "float",
        "rag_sources": "integer",
        "user_info_extracted": "boolean",
        "intent_category": "string",
        "conversation_status": "string"
    }
}
```

**Sample Request:**
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I need a 2000 sq ft office space in downtown with parking",
    "user_id": "user_123"
  }'
```

**Sample Response:**
```json
{
    "response": "I found several office spaces that match your criteria. Here are the best options: Property at 123 Main St offers 2100 sq ft with dedicated parking, annual rent $45,000. Would you like more details about this or other available properties?",
    "user_id": "user_123",
    "conversation_id": "conv_user_123_1704123456",
    "metadata": {
        "response_time": 1.23,
        "rag_sources": 3,
        "user_info_extracted": true,
        "intent_category": "property_search",
        "conversation_status": "inquiring"
    }
}
```

---

### 2. Document Upload

**Endpoint:** `POST /upload_docs`

**Description:** Upload documents to populate RAG knowledge base

**Request Format:** `multipart/form-data`
- Field: `files` (array of files)
- Supported formats: CSV, JSON, TXT, PDF

**Response Format:**
```json
{
    "message": "string",
    "files": [
        {
            "filename": "string",
            "size": "integer",
            "status": "string",
            "doc_type": "string",
            "processed_chunks": "integer"
        }
    ],
    "total_documents_added": "integer"
}
```

**Sample Request:**
```bash
curl -X POST "http://localhost:8000/upload_docs" \
  -F "files=@properties.csv" \
  -F "files=@company_info.txt"
```

**Sample Response:**
```json
{
    "message": "Documents uploaded successfully",
    "files": [
        {
            "filename": "properties.csv",
            "size": 15234,
            "status": "processed",
            "doc_type": "csv",
            "processed_chunks": 45
        },
        {
            "filename": "company_info.txt",
            "size": 2048,
            "status": "processed", 
            "doc_type": "text",
            "processed_chunks": 3
        }
    ],
    "total_documents_added": 48
}
```

---

## CRM Endpoints

### 3. Create User

**Endpoint:** `POST /crm/create_user`

**Description:** Create new user profile in CRM

**Request Format:**
```json
{
    "user_id": "string (optional)",
    "name": "string (optional)",
    "email": "string (optional)",
    "company": "string (optional)",
    "phone": "string (optional)",
    "preferences": {
        "preferred_size": "integer",
        "budget": "string",
        "location": "string",
        "move_in_date": "string"
    },
    "tags": ["string"]
}
```

**Response Format:**
```json
{
    "message": "string",
    "user_id": "string",
    "created_at": "string (ISO datetime)"
}
```

**Sample Request:**
```bash
curl -X POST "http://localhost:8000/crm/create_user" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "John Doe",
    "email": "john@techcorp.com",
    "company": "Tech Corp",
    "phone": "+1234567890",
    "preferences": {
        "preferred_size": 2000,
        "budget": "50000",
        "location": "downtown"
    }
  }'
```

### 4. Update User

**Endpoint:** `PUT /crm/update_user/{user_id}`

**Description:** Update existing user profile

**Path Parameters:**
- `user_id`: string (required)

**Request Format:** Same as create user (all fields optional)

**Response Format:**
```json
{
    "message": "string",
    "user_id": "string",
    "updated_fields": ["string"],
    "updated_at": "string (ISO datetime)"
}
```

### 5. Get Conversations

**Endpoint:** `GET /crm/conversations/{user_id}`

**Description:** Retrieve conversation history for specific user

**Path Parameters:**
- `user_id`: string (required)

**Query Parameters:**
- `limit`: integer (optional, default: 50)
- `offset`: integer (optional, default: 0)

**Response Format:**
```json
{
    "user_id": "string",
    "conversations": [
        {
            "message_id": "string",
            "user_message": "string",
            "bot_response": "string",
            "timestamp": "string (ISO datetime)",
            "rag_context": [
                {
                    "content": "string",
                    "source": "string",
                    "relevance_score": "float"
                }
            ],
            "metadata": {
                "response_time": "float",
                "intent": "string"
            }
        }
    ],
    "total_messages": "integer",
    "conversation_summary": "string"
}
```

### 6. Get All Users

**Endpoint:** `GET /crm/users`

**Description:** Retrieve all users in the system

**Query Parameters:**
- `limit`: integer (optional, default: 100)
- `offset`: integer (optional, default: 0)

**Response Format:**
```json
{
    "users": [
        {
            "user_id": "string",
            "name": "string",
            "email": "string",
            "company": "string",
            "phone": "string",
            "preferences": {},
            "tags": ["string"],
            "created_at": "string",
            "updated_at": "string",
            "total_conversations": "integer",
            "last_interaction": "string"
        }
    ],
    "total_users": "integer",
    "pagination": {
        "limit": "integer",
        "offset": "integer",
        "has_more": "boolean"
    }
}
```

---

## System Management Endpoints

### 7. Reset Conversations

**Endpoint:** `POST /reset`

**Description:** Clear conversation memory

**Request Format:**
```json
{
    "user_id": "string (optional - if provided, clears only that user's conversations)"
}
```

**Response Format:**
```json
{
    "message": "string",
    "affected_users": "integer",
    "conversations_cleared": "integer"
}
```

### 8. Health Check

**Endpoint:** `GET /health`

**Description:** System health and status check

**Response Format:**
```json
{
    "status": "string",
    "timestamp": "string (ISO datetime)",
    "services": {
        "crm": "string",
        "rag": "string", 
        "llm": "string",
        "database": "string"
    },
    "metrics": {
        "total_users": "integer",
        "total_conversations": "integer",
        "total_documents": "integer",
        "uptime": "string"
    },
    "version": "string"
}
```

### 9. System Metrics

**Endpoint:** `GET /metrics`

**Description:** Detailed system performance metrics

**Response Format:**
```json
{
    "performance": {
        "avg_response_time": "float",
        "total_requests": "integer",
        "requests_per_minute": "float"
    },
    "usage": {
        "active_users": "integer",
        "conversations_today": "integer",
        "documents_processed": "integer"
    },
    "resources": {
        "memory_usage": "string",
        "cpu_usage": "float",
        "disk_usage": "string"
    }
}
```

---

## Error Handling

### Error Response Format
```json
{
    "detail": "string",
    "error_code": "string",
    "timestamp": "string (ISO datetime)",
    "request_id": "string"
}
```

### Common HTTP Status Codes
- `200`: Success
- `201`: Created
- `400`: Bad Request
- `404`: Not Found
- `422`: Validation Error
- `500`: Internal Server Error

### Sample Error Response
```json
{
    "detail": "User not found with ID: user_123",
    "error_code": "USER_NOT_FOUND",
    "timestamp": "2024-01-01T12:00:00Z",
    "request_id": "req_abc123"
}
```

---

## Advanced Features

### Property Search

**Endpoint:** `GET /properties/search`

**Query Parameters:**
- `q`: string (search query)
- `min_size`: integer (minimum square feet)
- `max_size`: integer (maximum square feet)
- `location`: string (location filter)
- `max_rent`: float (maximum annual rent)
- `limit`: integer (default: 20)

**Response Format:**
```json
{
    "properties": [
        {
            "unique_id": "integer",
            "property_address": "string",
            "floor": "string",
            "suite": "string",
            "size_sf": "integer",
            "annual_rent": "string",
            "monthly_rent": "string",
            "broker_email": "string",
            "associates": ["string"]
        }
    ],
    "total_results": "integer",
    "search_metadata": {
        "query": "string",
        "filters_applied": {},
        "search_time": "float"
    }
}
```

### Calendar Integration (Optional)

**Endpoint:** `POST /calendar/events`

**Description:** Create calendar event for property viewing

**Request Format:**
```json
{
    "user_id": "string",
    "title": "string",
    "description": "string",
    "start_time": "string (ISO datetime)",
    "end_time": "string (ISO datetime)",
    "location": "string",
    "attendees": ["string"],
    "property_id": "integer (optional)"
}
```

---

## Usage Examples

### Complete Conversation Flow

1. **Initial Contact**
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hi, I am Sarah from ABC Corp looking for office space"}'
```

2. **Follow-up with Requirements**
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I need 1500-2000 sq ft in downtown area, budget around $40k annually",
    "user_id": "user_id_from_previous_response"
  }'
```

3. **Check Conversation History**
```bash
curl "http://localhost:8000/crm/conversations/user_id_from_previous_response"
```

### Data Upload and Search
```bash
# Upload property data
curl -X POST "http://localhost:8000/upload_docs" \
  -F "files=@property_listings.csv"

# Search for properties
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Show me all properties over 2000 sq ft with parking"}'
```

---

## Rate Limits
- No rate limits currently implemented
- All endpoints are open for hackathon demo
- Production deployment should implement appropriate rate limiting

## API Versioning
- Current version: v1.0
- No versioning in endpoints (for simplicity)
- Future versions will use `/v2/` prefix

## Support
For issues or questions about the API, check:
1. Interactive documentation: `http://localhost:8000/docs`
2. Health endpoint: `http://localhost:8000/health`
3. Run test suite: `python test_system.py`