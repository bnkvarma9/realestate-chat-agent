import sqlite3
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any
import logging
from models import UserProfile, ConversationMessage, ConversationHistory, ConversationStatus, ConversationCategory
import re
logger = logging.getLogger(__name__)

class CRMSystem:
    def __init__(self, db_path: str = "crm_database.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with required tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    name TEXT,
                    email TEXT,
                    company TEXT,
                    phone TEXT,
                    preferences TEXT,
                    tags TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Conversations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    status TEXT DEFAULT 'inquiring',
                    category TEXT DEFAULT 'general',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            
            # Messages table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    user_id TEXT,
                    user_message TEXT,
                    bot_response TEXT,
                    rag_context TEXT,
                    metadata TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id),
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            
            # Calendar events table (optional feature)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS calendar_events (
                    event_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    title TEXT,
                    description TEXT,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    location TEXT,
                    attendees TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON messages(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_conversation_id ON messages(conversation_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON messages(timestamp)')
            
            conn.commit()
            conn.close()
            logger.info("CRM database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization error: {str(e)}")
            raise
    
    async def create_user(self, user_profile: UserProfile) -> str:
        """Create a new user profile"""
        try:
            user_id = user_profile.user_id or str(uuid.uuid4())
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO users 
                (user_id, name, email, company, phone, preferences, tags, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                user_profile.name,
                user_profile.email,
                user_profile.company,
                user_profile.phone,
                json.dumps(user_profile.preferences or {}),
                json.dumps(user_profile.tags or []),
                datetime.now().isoformat(),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"User created: {user_id}")
            return user_id
            
        except Exception as e:
            logger.error(f"Create user error: {str(e)}")
            raise
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile by user ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return UserProfile(
                    user_id=row[0],
                    name=row[1],
                    email=row[2],
                    company=row[3],
                    phone=row[4],
                    preferences=json.loads(row[5] or '{}'),
                    tags=json.loads(row[6] or '[]'),
                    created_at=datetime.fromisoformat(row[7]) if row[7] else None,
                    updated_at=datetime.fromisoformat(row[8]) if row[8] else None
                )
            return None
            
        except Exception as e:
            logger.error(f"Get user profile error: {str(e)}")
            return None
    def is_valid_email(self, email: str) -> bool:
        # Simple regex to validate email format
        pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        return re.match(pattern, email) is not None
    def is_valid_phone(self, phone: str) -> bool:
        """
        Validates phone numbers in a simple way:
        - Allows digits, spaces, dashes, parentheses, and plus sign at start
        - Requires at least 10 digits (typical US number length)
        """
        if not isinstance(phone, str):
            return False
        
        # Remove common formatting characters
        digits_only = re.sub(r'[\s\-\(\)+]', '', phone)
        
        # Check if all remaining characters are digits and length >= 10
        return digits_only.isdigit() and len(digits_only) >= 10
    async def update_user_profile(self, user_id: str, updates: Dict[str, Any], source: str = 'chat') -> bool:
        """Update user profile with new information. Only allow name update from 'form' source."""
        try:
            existing_profile = await self.get_user_profile(user_id)
            if not existing_profile:
                # Only allow name if source is 'form'
                if source != 'form' and 'name' in updates:
                    updates.pop('name')  # prevent chat from setting name
                user_profile = UserProfile(user_id=user_id, **updates)
                await self.create_user(user_profile)
                return True

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            update_fields = []
            values = []

            for field, value in updates.items():
                if field == 'name' and source != 'form':
                    continue  # skip name from chat
                elif field == 'email' and self.is_valid_email(value):
                    update_fields.append("email = ?")
                    values.append(value)
                elif field == 'company' and len(value.strip()) > 2:
                    update_fields.append("company = ?")
                    values.append(value)
                elif field == 'phone' and self.is_valid_phone(value):
                    update_fields.append("phone = ?")
                    values.append(value)
                elif field == 'preferences':
                    update_fields.append("preferences = ?")
                    values.append(json.dumps(value))
                elif field == 'tags':
                    update_fields.append("tags = ?")
                    values.append(json.dumps(value))
                elif field == 'name' and source == 'form':
                    update_fields.append("name = ?")
                    values.append(value)

            if update_fields:
                update_fields.append("updated_at = ?")
                values.append(datetime.now().isoformat())
                values.append(user_id)

                query = f"UPDATE users SET {', '.join(update_fields)} WHERE user_id = ?"
                cursor.execute(query, values)
                conn.commit()

            conn.close()
            return True

        except Exception as e:
            logger.error(f"Update user profile error: {str(e)}")
            return False

    async def save_conversation(self, user_id: str, user_message: str, bot_response: str, 
                              rag_context: List[Dict[str, Any]], metadata: Dict[str, Any]) -> str:
        """Save a conversation exchange"""
        try:
            # Get or create conversation
            conversation_id = await self.get_or_create_conversation(user_id)
            message_id = str(uuid.uuid4())
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert message
            cursor.execute('''
                INSERT INTO messages 
                (message_id, conversation_id, user_id, user_message, bot_response, rag_context, metadata, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                message_id,
                conversation_id,
                user_id,
                user_message,
                bot_response,
                json.dumps(rag_context),
                json.dumps(metadata),
                datetime.now().isoformat()
            ))
            
            # Update conversation timestamp
            cursor.execute('''
                UPDATE conversations 
                SET updated_at = ?, category = ?, status = ?
                WHERE conversation_id = ?
            ''', (
                datetime.now().isoformat(),
                self.categorize_conversation(user_message),
                self.determine_status(user_message, bot_response),
                conversation_id
            ))
            
            conn.commit()
            conn.close()
            
            return message_id
            
        except Exception as e:
            logger.error(f"Save conversation error: {str(e)}")
            raise
    
    async def get_or_create_conversation(self, user_id: str) -> str:
        """Get existing conversation or create new one"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check for existing active conversation
            cursor.execute('''
                SELECT conversation_id FROM conversations 
                WHERE user_id = ? AND status != 'resolved'
                ORDER BY updated_at DESC LIMIT 1
            ''', (user_id,))
            
            row = cursor.fetchone()
            
            if row:
                conversation_id = row[0]
            else:
                # Create new conversation
                conversation_id = str(uuid.uuid4())
                cursor.execute('''
                    INSERT INTO conversations (conversation_id, user_id, status, category)
                    VALUES (?, ?, ?, ?)
                ''', (conversation_id, user_id, 'inquiring', 'general'))
                
                conn.commit()
            
            conn.close()
            return conversation_id
            
        except Exception as e:
            logger.error(f"Get or create conversation error: {str(e)}")
            raise
    
    async def get_conversation_history(self, user_id: str, limit: int = 50) -> List[ConversationMessage]:
        """Get conversation history for a user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT message_id, conversation_id, user_id, user_message, bot_response, 
                       rag_context, metadata, timestamp
                FROM messages 
                WHERE user_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (user_id, limit))
            
            rows = cursor.fetchall()
            conn.close()
            
            messages = []
            for row in rows:
                message = ConversationMessage(
                    message_id=row[0],
                    user_id=row[2],
                    user_message=row[3],
                    bot_response=row[4],
                    rag_context=json.loads(row[5] or '[]'),
                    metadata=json.loads(row[6] or '{}'),
                    timestamp=datetime.fromisoformat(row[7])
                )
                messages.append(message)
            
            return list(reversed(messages))  # Return in chronological order
            
        except Exception as e:
            logger.error(f"Get conversation history error: {str(e)}")
            return []
    
    async def get_all_users(self) -> List[UserProfile]:
        """Get all users in the system"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM users ORDER BY created_at DESC')
            rows = cursor.fetchall()
            conn.close()
            
            users = []
            for row in rows:
                user = UserProfile(
                    user_id=row[0],
                    name=row[1],
                    email=row[2],
                    company=row[3],
                    phone=row[4],
                    preferences=json.loads(row[5] or '{}'),
                    tags=json.loads(row[6] or '[]'),
                    created_at=datetime.fromisoformat(row[7]) if row[7] else None,
                    updated_at=datetime.fromisoformat(row[8]) if row[8] else None
                )
                users.append(user)
            
            return users
            
        except Exception as e:
            logger.error(f"Get all users error: {str(e)}")
            return []
    
    async def clear_user_conversations(self, user_id: str) -> bool:
        """Clear conversations for a specific user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM messages WHERE user_id = ?', (user_id,))
            cursor.execute('DELETE FROM conversations WHERE user_id = ?', (user_id,))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Clear user conversations error: {str(e)}")
            return False
    
    async def clear_all_conversations(self) -> bool:
        """Clear all conversations"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM messages')
            cursor.execute('DELETE FROM conversations')
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Clear all conversations error: {str(e)}")
            return False
    
    def categorize_conversation(self, message: str) -> str:
        """Categorize conversation based on message content"""
        message_lower = message.lower()
        
        # Real estate keywords
        real_estate_keywords = ['property', 'rent', 'lease', 'office', 'space', 'building', 
                               'floor', 'suite', 'sf', 'square feet', 'broker', 'real estate']
        
        # Support keywords
        support_keywords = ['help', 'issue', 'problem', 'error', 'support', 'bug', 'fix']
        
        # Sales keywords
        sales_keywords = ['buy', 'purchase', 'price', 'cost', 'deal', 'offer', 'quote']
        
        if any(keyword in message_lower for keyword in real_estate_keywords):
            return ConversationCategory.REAL_ESTATE
        elif any(keyword in message_lower for keyword in support_keywords):
            return ConversationCategory.SUPPORT
        elif any(keyword in message_lower for keyword in sales_keywords):
            return ConversationCategory.SALES
        else:
            return ConversationCategory.GENERAL
    
    def determine_status(self, user_message: str, bot_response: str) -> str:
        """Determine conversation status based on messages"""
        user_lower = user_message.lower()
        bot_lower = bot_response.lower()
        
        # Check for resolution indicators
        resolution_keywords = ['thank you', 'thanks', 'solved', 'resolved', 'perfect', 'great']
        question_keywords = ['?', 'how', 'what', 'when', 'where', 'why', 'can you']
        
        if any(keyword in user_lower for keyword in resolution_keywords):
            return ConversationStatus.RESOLVED
        elif any(keyword in user_lower for keyword in question_keywords):
            return ConversationStatus.INQUIRING
        elif 'follow up' in bot_lower or 'contact' in bot_lower:
            return ConversationStatus.FOLLOW_UP
        else:
            return ConversationStatus.UNRESOLVED
    
    # Calendar integration methods (optional feature)
    async def create_calendar_event(self, user_id: str, title: str, start_time: datetime, 
                                  end_time: datetime, description: str = None, 
                                  location: str = None, attendees: List[str] = None) -> str:
        """Create a calendar event for a user"""
        try:
            event_id = str(uuid.uuid4())
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO calendar_events 
                (event_id, user_id, title, description, start_time, end_time, location, attendees)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event_id,
                user_id,
                title,
                description,
                start_time.isoformat(),
                end_time.isoformat(),
                location,
                json.dumps(attendees or [])
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Calendar event created: {event_id}")
            return event_id
            
        except Exception as e:
            logger.error(f"Create calendar event error: {str(e)}")
            raise
    
    async def get_user_calendar_events(self, user_id: str, 
                                     start_date: datetime = None, 
                                     end_date: datetime = None) -> List[Dict[str, Any]]:
        """Get calendar events for a user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = 'SELECT * FROM calendar_events WHERE user_id = ?'
            params = [user_id]
            
            if start_date:
                query += ' AND start_time >= ?'
                params.append(start_date.isoformat())
            
            if end_date:
                query += ' AND end_time <= ?'
                params.append(end_date.isoformat())
            
            query += ' ORDER BY start_time'
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            events = []
            for row in rows:
                event = {
                    'event_id': row[0],
                    'user_id': row[1],
                    'title': row[2],
                    'description': row[3],
                    'start_time': row[4],
                    'end_time': row[5],
                    'location': row[6],
                    'attendees': json.loads(row[7] or '[]'),
                    'created_at': row[8]
                }
                events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Get calendar events error: {str(e)}")
            return []