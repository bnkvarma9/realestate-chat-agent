import sqlite3
import json
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

class EnhancedAppointmentSystem:
    def __init__(self, db_path: str = "appointments.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize enhanced appointment management database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Enhanced appointments table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS appointments (
                    appointment_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    broker_id TEXT NOT NULL,
                    property_id TEXT,
                    property_address TEXT,
                    appointment_type TEXT NOT NULL,
                    status TEXT DEFAULT 'scheduled',
                    scheduled_datetime TIMESTAMP NOT NULL,
                    duration_minutes INTEGER DEFAULT 60,
                    customer_name TEXT NOT NULL,
                    customer_email TEXT NOT NULL,
                    customer_phone TEXT,
                    agenda TEXT,
                    notes TEXT,
                    preparation_notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    confirmed_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    follow_up_required BOOLEAN DEFAULT 1,
                    outcome TEXT,
                    next_steps TEXT,
                    priority TEXT DEFAULT 'medium',
                    maintenance_type TEXT,
                    urgency_level TEXT DEFAULT 'normal',
                    tenant_id TEXT,
                    unit_number TEXT,
                    issue_category TEXT,
                    preferred_contact_method TEXT DEFAULT 'email'
                )
            ''')
            
            # Enhanced brokers table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS brokers (
                    broker_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    phone TEXT,
                    specialization TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    max_daily_appointments INTEGER DEFAULT 8,
                    working_hours_start TIME DEFAULT '09:00',
                    working_hours_end TIME DEFAULT '18:00',
                    working_days TEXT DEFAULT 'mon,tue,wed,thu,fri',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Maintenance staff table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS maintenance_staff (
                    staff_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    phone TEXT,
                    specialties TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    max_daily_appointments INTEGER DEFAULT 12,
                    working_hours_start TIME DEFAULT '08:00',
                    working_hours_end TIME DEFAULT '17:00',
                    working_days TEXT DEFAULT 'mon,tue,wed,thu,fri',
                    emergency_contact BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Property tenants table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS property_tenants (
                    tenant_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    property_id TEXT,
                    property_address TEXT,
                    unit_number TEXT,
                    tenant_name TEXT NOT NULL,
                    tenant_email TEXT NOT NULL,
                    tenant_phone TEXT,
                    lease_start_date DATE,
                    lease_end_date DATE,
                    is_active BOOLEAN DEFAULT 1,
                    emergency_contact TEXT,
                    move_in_date DATE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            
            # Appointment conflicts tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS appointment_conflicts (
                    conflict_id TEXT PRIMARY KEY,
                    original_appointment_id TEXT,
                    conflicting_appointment_id TEXT,
                    conflict_type TEXT,
                    resolution_status TEXT DEFAULT 'pending',
                    resolved_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (original_appointment_id) REFERENCES appointments (appointment_id),
                    FOREIGN KEY (conflicting_appointment_id) REFERENCES appointments (appointment_id)
                )
            ''')
            
            # Insert default brokers and maintenance staff
            cursor.execute('SELECT COUNT(*) FROM brokers')
            if cursor.fetchone()[0] == 0:
                default_brokers = [
                    ('broker_001', 'John Smith', 'john.smith@realty.com', '(555) 123-4567', 'commercial'),
                    ('broker_002', 'Sarah Johnson', 'sarah.johnson@realty.com', '(555) 234-5678', 'residential'),
                    ('broker_003', 'Mike Chen', 'mike.chen@realty.com', '(555) 345-6789', 'industrial')
                ]
                
                for broker in default_brokers:
                    cursor.execute('''
                        INSERT INTO brokers (broker_id, name, email, phone, specialization)
                        VALUES (?, ?, ?, ?, ?)
                    ''', broker)
            
            cursor.execute('SELECT COUNT(*) FROM maintenance_staff')
            if cursor.fetchone()[0] == 0:
                default_maintenance = [
                    ('maint_001', 'Bob Wilson', 'bob.wilson@maintenance.com', '(555) 111-2222', 'plumbing,electrical', 1, 12, '08:00', '17:00', 'mon,tue,wed,thu,fri', 1),
                    ('maint_002', 'Alice Brown', 'alice.brown@maintenance.com', '(555) 333-4444', 'hvac,general', 1, 10, '09:00', '18:00', 'mon,tue,wed,thu,fri', 0),
                    ('maint_003', 'Tom Davis', 'tom.davis@maintenance.com', '(555) 555-6666', 'carpentry,painting', 1, 8, '08:00', '16:00', 'mon,tue,wed,thu,fri', 0)
                ]
                
                for staff in default_maintenance:
                    cursor.execute('''
                        INSERT INTO maintenance_staff 
                        (staff_id, name, email, phone, specialties, is_active, max_daily_appointments, 
                         working_hours_start, working_hours_end, working_days, emergency_contact)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', staff)
            
            conn.commit()
            conn.close()
            logger.info("Enhanced appointment system database initialized successfully")
            
        except Exception as e:
            logger.error(f"Enhanced appointment database initialization error: {str(e)}")
            raise
    async def update_appointment(self, appointment_id: str, user_id: str, updates: Dict[str, Any]) -> tuple[bool, str]:
        """Update an existing appointment"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            fields = []
            values = []

            for key, value in updates.items():
                fields.append(f"{key} = ?")
                values.append(value)

            fields.append("updated_at = ?")
            values.append(datetime.now().isoformat())

            # These are WHERE clause parameters
            where_values = [appointment_id, user_id]

            sql = f'''
                UPDATE appointments 
                SET {", ".join(fields)}
                WHERE appointment_id = ? AND user_id = ? AND status NOT IN ('cancelled', 'completed')
            '''

            cursor.execute(sql, values + where_values)

            conn.commit()
            conn.close()

            if cursor.rowcount == 0:
                return False, "Appointment not found or already finalized"
            return True, "Appointment updated successfully"

        except Exception as e:
            logger.error(f"Update appointment error: {str(e)}")
            return False, f"Error updating appointment: {str(e)}"

    async def delete_appointment(self, appointment_id: str, user_id: str) -> tuple[bool, str]:
        """Delete (permanently remove) an appointment"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Ensure the appointment exists
            cursor.execute('''
                SELECT appointment_id FROM appointments 
                WHERE appointment_id = ? AND user_id = ?
            ''', (appointment_id, user_id))

            if not cursor.fetchone():
                conn.close()
                return False, "Appointment not found"

            # Delete the appointment
            cursor.execute('''
                DELETE FROM appointments WHERE appointment_id = ? AND user_id = ?
            ''', (appointment_id, user_id))

            conn.commit()
            conn.close()

            return True, "Appointment deleted successfully"

        except Exception as e:
            logger.error(f"Delete appointment error: {str(e)}")
            return False, f"Error deleting appointment: {str(e)}"
    async def create_appointment(self, request_data: dict) -> tuple[str, str]:
        """Create a new appointment with enhanced validation and error handling"""
        try:
            appointment_id = str(uuid.uuid4())
            appointment_type = request_data.get('appointment_type', 'site_visit')
            
            # Validate and clean input data
            customer_name = request_data.get('customer_name', '').strip()
            customer_email = request_data.get('customer_email', '').strip()
            
            if not customer_name or not customer_email:
                raise ValueError("Customer name and email are required")
            
            # Safe handling of property address
            property_address = request_data.get('property_address')
            if property_address:
                property_address = str(property_address).strip()
            
            # Safe handling of notes
            notes = request_data.get('notes')
            if notes:
                notes = str(notes).strip()
            
            # Determine if this is maintenance/repair vs property viewing
            is_maintenance = appointment_type in ['maintenance', 'repair', 'emergency_repair']
            
            # Parse datetime with error handling
            try:
                scheduled_datetime = self._parse_datetime(request_data)
            except Exception as dt_error:
                logger.error(f"DateTime parsing error: {str(dt_error)}")
                # Use tomorrow at 2 PM as fallback
                tomorrow = datetime.now() + timedelta(days=1)
                scheduled_datetime = tomorrow.replace(hour=14, minute=0, second=0, microsecond=0)
            
            # Check for conflicts
            user_id = request_data.get('user_id')
            if user_id:
                conflict_check = await self._check_appointment_conflicts(
                    user_id, scheduled_datetime, appointment_type
                )
                
                if conflict_check['has_conflict']:
                    raise ValueError(f"Appointment conflict: {conflict_check['message']}")
            
            # Assign appropriate staff
            if is_maintenance:
                assigned_staff = await self._assign_maintenance_staff(request_data, scheduled_datetime)
            else:
                assigned_staff = await self._assign_broker(request_data, scheduled_datetime)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Determine priority and urgency
            priority = self._determine_priority(request_data)
            urgency_level = self._determine_urgency(request_data)
            
            cursor.execute('''
                INSERT INTO appointments (
                    appointment_id, user_id, broker_id, property_id, property_address,
                    appointment_type, scheduled_datetime, customer_name, customer_email,
                    customer_phone, notes, priority, maintenance_type, urgency_level,
                    tenant_id, unit_number, issue_category, preferred_contact_method
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                appointment_id,
                user_id or str(uuid.uuid4()),
                assigned_staff['staff_id'],
                request_data.get('property_id'),
                property_address,
                appointment_type,
                scheduled_datetime.isoformat(),
                customer_name,
                customer_email,
                request_data.get('customer_phone'),
                notes,
                priority,
                request_data.get('maintenance_type'),
                urgency_level,
                request_data.get('tenant_id'),
                request_data.get('unit_number'),
                request_data.get('issue_category'),
                request_data.get('preferred_contact_method', 'email')
            ))
            
            conn.commit()
            conn.close()
            
            # Generate confirmation message
            confirmation_msg = self._generate_confirmation_message(
                appointment_type, scheduled_datetime, assigned_staff, request_data
            )
            
            logger.info(f"appointment created: {appointment_id}")
            return appointment_id, confirmation_msg
            
        except Exception as e:
            logger.error(f"Create  appointment error: {str(e)}")
            raise
    def _generate_confirmation_message(self, appointment_type: str, scheduled_datetime: datetime, 
                                 assigned_staff: dict, request_data: dict) -> str:
        """Generate appointment confirmation message"""
        try:
            staff_name = assigned_staff.get('name', 'Staff Member')
            staff_type = assigned_staff.get('type', 'staff')
            
            date_str = scheduled_datetime.strftime("%B %d, %Y")
            time_str = scheduled_datetime.strftime("%I:%M %p")
            
            if appointment_type in ['maintenance', 'repair', 'emergency_repair']:
                return (f"Maintenance appointment scheduled for {date_str} at {time_str}. "
                    f"Our {staff_type} {staff_name} will handle your request. "
                    f"Contact: {assigned_staff.get('email', 'N/A')}")
            else:
                property_info = ""
                if request_data.get('property_address'):
                    property_info = f" for {request_data['property_address']}"
                
                return (f"Property visit scheduled for {date_str} at {time_str}{property_info}. "
                    f"Your {staff_type} {staff_name} will meet you. "
                    f"Contact: {assigned_staff.get('email', 'N/A')}")
        
        except Exception as e:
            logger.error(f"Generate confirmation message error: {str(e)}")
            return f"Appointment scheduled for {scheduled_datetime.strftime('%B %d, %Y at %I:%M %p')}"
    def _determine_priority(self, request_data: dict) -> str:
        """Determine appointment priority based on request data"""
        try:
            appointment_type = request_data.get('appointment_type', '').lower()
            notes = (request_data.get('notes') or '').lower()
            maintenance_type = (request_data.get('maintenance_type') or '').lower()
            
            # High priority conditions
            high_priority_keywords = ['emergency', 'urgent', 'leak', 'flooding', 'no power', 'no heat']
            if (appointment_type == 'emergency_repair' or 
                any(keyword in notes for keyword in high_priority_keywords) or
                any(keyword in maintenance_type for keyword in high_priority_keywords)):
                return 'high'
            
            # Medium priority for regular maintenance
            if appointment_type in ['maintenance', 'repair']:
                return 'medium'
            
            # Low priority for property visits
            return 'low'
            
        except Exception as e:
            logger.error(f"Determine priority error: {str(e)}")
            return 'medium'

    def _determine_urgency(self, request_data: dict) -> str:
        """Determine urgency level based on request data"""
        try:
            appointment_type = request_data.get('appointment_type', '').lower()
            notes = (request_data.get('notes') or '').lower()
            maintenance_type = (request_data.get('maintenance_type') or '').lower()
            
            # Urgent conditions
            urgent_keywords = ['emergency', 'urgent', 'asap', 'immediate', 'critical']
            if (appointment_type == 'emergency_repair' or
                any(keyword in notes for keyword in urgent_keywords) or
                any(keyword in maintenance_type for keyword in urgent_keywords)):
                return 'urgent'
            
            # High urgency for serious issues
            high_urgency_keywords = ['leak', 'flooding', 'no power', 'no heat', 'broken']
            if any(keyword in notes for keyword in high_urgency_keywords):
                return 'high'
            
            return 'normal'
            
        except Exception as e:
            logger.error(f"Determine urgency error: {str(e)}")
            return 'normal'
    def _parse_datetime(self, request_data: dict) -> datetime:
        """Parse appointment datetime from various formats"""
        try:
            # Try different datetime formats
            date_str = request_data.get('appointment_date')
            time_str = request_data.get('appointment_time')
            
            if date_str and time_str:
                # Combine date and time
                datetime_str = f"{date_str} {time_str}"
                
                # Try different formats
                formats = [
                    '%Y-%m-%d %I:%M %p',  # 2024-12-15 2:00 PM
                    '%Y-%m-%d %H:%M',     # 2024-12-15 14:00
                    '%m/%d/%Y %I:%M %p',  # 12/15/2024 2:00 PM
                    '%m/%d/%Y %H:%M'      # 12/15/2024 14:00
                ]
                
                for fmt in formats:
                    try:
                        return datetime.strptime(datetime_str, fmt)
                    except ValueError:
                        continue
            
            # Fallback to preferred_date if available
            preferred_date = request_data.get('preferred_date')
            if preferred_date:
                if isinstance(preferred_date, str):
                    return datetime.fromisoformat(preferred_date.replace('Z', '+00:00'))
                return preferred_date
            
            # Default to tomorrow at 2 PM
            tomorrow = datetime.now() + timedelta(days=1)
            return tomorrow.replace(hour=14, minute=0, second=0, microsecond=0)
            
        except Exception as e:
            logger.error(f"Datetime parsing error: {str(e)}")
            # Return tomorrow at 2 PM as fallback
            tomorrow = datetime.now() + timedelta(days=1)
            return tomorrow.replace(hour=14, minute=0, second=0, microsecond=0)
    
    async def _check_appointment_conflicts(self, user_id: str, scheduled_datetime: datetime, appointment_type: str) -> dict:
        """Check for appointment conflicts"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check for existing appointments within 2 hours
            time_buffer = timedelta(hours=2)
            start_time = scheduled_datetime - time_buffer
            end_time = scheduled_datetime + time_buffer
            
            cursor.execute('''
                SELECT appointment_id, appointment_type, scheduled_datetime, property_address
                FROM appointments 
                WHERE user_id = ? 
                AND scheduled_datetime BETWEEN ? AND ?
                AND status NOT IN ('cancelled', 'completed')
            ''', (user_id, start_time.isoformat(), end_time.isoformat()))
            
            conflicts = cursor.fetchall()
            conn.close()
            
            if conflicts:
                conflict_details = conflicts[0]
                return {
                    'has_conflict': True,
                    'message': f"You already have a {conflict_details[1]} appointment at {conflict_details[2]} for {conflict_details[3]}",
                    'existing_appointment': {
                        'id': conflict_details[0],
                        'type': conflict_details[1],
                        'datetime': conflict_details[2],
                        'property': conflict_details[3]
                    }
                }
            
            return {'has_conflict': False}
            
        except Exception as e:
            logger.error(f"Conflict check error: {str(e)}")
            return {'has_conflict': False}
    
    async def _assign_maintenance_staff(self, request_data: dict, scheduled_datetime: datetime) -> dict:
        """Assign appropriate maintenance staff based on request"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Determine required specialty
            issue_category = request_data.get('issue_category', '').lower()
            maintenance_type = request_data.get('maintenance_type', '').lower()
            notes = request_data.get('notes', '').lower()
            
            specialty_keywords = {
                'plumbing': ['plumbing', 'leak', 'pipe', 'water', 'drain', 'toilet', 'sink'],
                'electrical': ['electrical', 'electric', 'power', 'light', 'outlet', 'wiring'],
                'hvac': ['hvac', 'heating', 'cooling', 'air', 'temperature', 'thermostat'],
                'carpentry': ['carpentry', 'door', 'window', 'wood', 'cabinet'],
                'general': ['general', 'cleaning', 'painting', 'maintenance']
            }
            
            required_specialty = 'general'
            text_to_check = f"{issue_category} {maintenance_type} {notes}"
            
            for specialty, keywords in specialty_keywords.items():
                if any(keyword in text_to_check for keyword in keywords):
                    required_specialty = specialty
                    break
            
            # Check if it's an emergency
            is_emergency = any(word in text_to_check for word in ['emergency', 'urgent', 'leak', 'flooding', 'no power'])
            
            # Find available maintenance staff
            where_clause = "WHERE is_active = 1"
            params = []
            
            if is_emergency:
                where_clause += " AND emergency_contact = 1"
            
            if required_specialty != 'general':
                where_clause += " AND (specialties LIKE ? OR specialties LIKE ?)"
                params.extend([f"%{required_specialty}%", "%general%"])
            
            cursor.execute(f'''
                SELECT staff_id, name, email, phone, specialties
                FROM maintenance_staff 
                {where_clause}
                ORDER BY emergency_contact DESC, staff_id
                LIMIT 1''', params)
            
            staff = cursor.fetchone()
            conn.close()
            if staff:
                return {
                    'staff_id': staff[0],
                    'name': staff[1],
                    'email': staff[2],
                    'phone': staff[3],
                    'specialties': staff[4]
                }
            else:
                # Fallback: return a default staff or raise an error
                logger.warning("No available maintenance staff found, assigning default staff.")
                return {
                    'staff_id': 'maint_default',
                    'name': 'Default Staff',
                    'email': 'default@maintenance.com',
                    'phone': '',
                    'specialties': required_specialty
                }
        except Exception as e:
            logger.error(f"Error assigning maintenance staff: {str(e)}")
            return {
                'staff_id': 'maint_default',
                'name': 'Default Staff',
                'email': 'default@maintenance.com',
                'phone': '',
                'specialties': 'general'
            }
    # Add this method to your EnhancedAppointmentSystem class in appointment_system.py

    async def get_tenant_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get tenant information for a user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT tenant_id, property_id, property_address, unit_number, tenant_name, tenant_email, tenant_phone
                FROM property_tenants 
                WHERE user_id = ? AND is_active = 1
                LIMIT 1
            ''', (user_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    'tenant_id': row[0],
                    'property_id': row[1],
                    'property_address': row[2],
                    'unit_number': row[3],
                    'tenant_name': row[4],
                    'tenant_email': row[5],
                    'tenant_phone': row[6]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Get tenant info error: {str(e)}")
            return None

    async def register_tenant(self, tenant_data: dict) -> str:
        """Register a new tenant"""
        try:
            tenant_id = str(uuid.uuid4())
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO property_tenants 
                (tenant_id, user_id, property_id, property_address, unit_number, 
                tenant_name, tenant_email, tenant_phone, lease_start_date, lease_end_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                tenant_id,
                tenant_data.get('user_id'),
                tenant_data.get('property_id'),
                tenant_data.get('property_address'),
                tenant_data.get('unit_number'),
                tenant_data.get('tenant_name'),
                tenant_data.get('tenant_email'),
                tenant_data.get('tenant_phone'),
                tenant_data.get('lease_start_date'),
                tenant_data.get('lease_end_date')
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Tenant registered: {tenant_id}")
            return tenant_id
            
        except Exception as e:
            logger.error(f"Register tenant error: {str(e)}")
            raise

    async def get_maintenance_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get maintenance history for a user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT appointment_id, appointment_type, scheduled_datetime, status, 
                    property_address, maintenance_type, urgency_level, notes, outcome
                FROM appointments 
                WHERE user_id = ? AND appointment_type IN ('maintenance', 'repair', 'emergency_repair')
                ORDER BY scheduled_datetime DESC
            ''', (user_id,))
            
            rows = cursor.fetchall()
            conn.close()
            
            history = []
            for row in rows:
                history.append({
                    'appointment_id': row[0],
                    'appointment_type': row[1],
                    'scheduled_datetime': row[2],
                    'status': row[3],
                    'property_address': row[4],
                    'maintenance_type': row[5],
                    'urgency_level': row[6],
                    'notes': row[7],
                    'outcome': row[8]
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Get maintenance history error: {str(e)}")
            return []

    async def escalate_maintenance_request(self, appointment_id: str, reason: str) -> bool:
        """Escalate a maintenance request"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE appointments 
                SET urgency_level = 'urgent', 
                    priority = 'high',
                    notes = COALESCE(notes, '') || '\nESCALATED: ' || ?,
                    updated_at = ?
                WHERE appointment_id = ?
            ''', (reason, datetime.now().isoformat(), appointment_id))
            
            conn.commit()
            conn.close()
            
            return cursor.rowcount > 0
            
        except Exception as e:
            logger.error(f"Escalate maintenance error: {str(e)}")
            return False

    async def _assign_broker(self, appointment_data: dict, scheduled_datetime: datetime = None) -> Dict[str, str]:
        """Assign a broker for property visits"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get available brokers
            cursor.execute('''
                SELECT broker_id, name, email, phone, specialization
                FROM brokers 
                WHERE is_active = 1
                ORDER BY broker_id
                LIMIT 1
            ''', ())
            
            broker = cursor.fetchone()
            conn.close()
            
            if broker:
                return {
                    'staff_id': broker[0],
                    'name': broker[1],
                    'email': broker[2],
                    'phone': broker[3],
                    'specialization': broker[4],
                    'type': 'broker'
                }
            else:
                # Fallback broker
                return {
                    'staff_id': 'broker_default',
                    'name': 'Default Broker',
                    'email': 'broker@company.com',
                    'phone': '(555) 123-RENT',
                    'specialization': 'general',
                    'type': 'broker'
                }
                
        except Exception as e:
            logger.error(f"Assign broker error: {str(e)}")
            return {
                'staff_id': 'broker_default',
                'name': 'Default Broker',
                'email': 'broker@company.com',
                'phone': '(555) 123-RENT',
                'specialization': 'general',
                'type': 'broker'
            }

    async def get_available_time_slots(self, date: str, staff_type: str = "broker") -> List[str]:
        """Get available time slots for a given date"""
        try:
            # Generate default time slots (you can enhance this with real availability)
            base_slots = [
                '9:00 AM', '9:30 AM', '10:00 AM', '10:30 AM', '11:00 AM', '11:30 AM',
                '12:00 PM', '12:30 PM', '1:00 PM', '1:30 PM', '2:00 PM', '2:30 PM',
                '3:00 PM', '3:30 PM', '4:00 PM', '4:30 PM', '5:00 PM', '5:30 PM'
            ]
            
            # You can add logic here to check existing appointments and remove conflicts
            # For now, return all slots as available
            return base_slots
            
        except Exception as e:
            logger.error(f"Get available time slots error: {str(e)}")
            return []

    async def cancel_appointment(self, appointment_id: str, user_id: str) -> tuple[bool, str]:
        """Cancel an appointment"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if appointment exists and belongs to user
            cursor.execute('''
                SELECT appointment_id FROM appointments 
                WHERE appointment_id = ? AND user_id = ? AND status NOT IN ('cancelled', 'completed')
            ''', (appointment_id, user_id))
            
            if not cursor.fetchone():
                conn.close()
                return False, "Appointment not found or already cancelled"
            
            # Cancel the appointment
            cursor.execute('''
                UPDATE appointments 
                SET status = 'cancelled', updated_at = ?
                WHERE appointment_id = ?
            ''', (datetime.now().isoformat(), appointment_id))
            
            conn.commit()
            conn.close()
            
            return True, "Appointment cancelled successfully"
            
        except Exception as e:
            logger.error(f"Cancel appointment error: {str(e)}")
            return False, f"Error cancelling appointment: {str(e)}"

    async def get_user_appointments(self, user_id: str) -> List[Dict[str, Any]]:
        """Get appointments for a user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT appointment_id, appointment_type, scheduled_datetime, status,
                    property_address, customer_name, customer_email, notes
                FROM appointments 
                WHERE user_id = ?
                ORDER BY scheduled_datetime DESC
            ''', (user_id,))
            
            rows = cursor.fetchall()
            conn.close()
            
            appointments = []
            for row in rows:
                appointments.append({
                    'appointment_id': row[0],
                    'appointment_type': row[1],
                    'appointment_date': row[2].split('T')[0] if 'T' in row[2] else row[2],
                    'appointment_time': row[2].split('T')[1] if 'T' in row[2] else '14:00',
                    'status': row[3],
                    'property_address': row[4],
                    'customer_name': row[5],
                    'customer_email': row[6],
                    'notes': row[7]
                })
            
            return appointments
            
        except Exception as e:
            logger.error(f"Get user appointments error: {str(e)}")
            return []