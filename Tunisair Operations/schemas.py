"""
Tunisair Operations - Pydantic Schemas
Data validation and serialization schemas
"""

from pydantic import BaseModel, EmailStr, validator
from typing import List, Optional
from datetime import datetime, date
from models import FlightStatusEnum, CrewRoleEnum, CrewStatusEnum, AircraftStatusEnum, MaintenanceTypeEnum

# User Schemas
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    phone_number: Optional[str] = None
    password: str
    full_name: Optional[str] = None
    
    @validator('username')
    def username_alphanumeric(cls, v):
        assert v.replace('_', '').isalnum(), 'Username must be alphanumeric'
        assert len(v) >= 3, 'Username must be at least 3 characters'
        return v
    
    @validator('password')
    def password_strength(cls, v):
        assert len(v) >= 6, 'Password must be at least 6 characters'
        return v

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    phone_number: Optional[str] = None
    full_name: Optional[str]
    is_active: bool
    is_admin: bool
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: UserResponse

# Aircraft Schemas
class AircraftResponse(BaseModel):
    id: int
    registration: str
    model: str
    manufacturer: str
    year_manufactured: Optional[int]
    total_seats: int
    status: AircraftStatusEnum
    total_flight_hours: float
    last_maintenance_date: Optional[date]
    next_maintenance_date: Optional[date]
    current_location: Optional[str]
    
    class Config:
        from_attributes = True

class AircraftListResponse(BaseModel):
    total: int
    items: List[AircraftResponse]

# Crew Schemas
class CrewMemberResponse(BaseModel):
    id: int
    employee_id: str
    full_name: str
    role: CrewRoleEnum
    license_number: Optional[str]
    total_flight_hours: float
    status: CrewStatusEnum
    base_location: str
    
    class Config:
        from_attributes = True

class CrewMemberListResponse(BaseModel):
    total: int
    items: List[CrewMemberResponse]

# Flight Schemas
class FlightResponse(BaseModel):
    id: int
    flight_number: str
    aircraft_id: int
    origin: str
    destination: str
    scheduled_departure: datetime
    scheduled_arrival: datetime
    actual_departure: Optional[datetime]
    actual_arrival: Optional[datetime]
    status: FlightStatusEnum
    gate: Optional[str]
    delay_minutes: int
    passengers_booked: int
    base_price: Optional[float] = 150.0
    
    class Config:
        from_attributes = True

class FlightDetailResponse(FlightResponse):
    aircraft: Optional[AircraftResponse] = None
    crew_count: int = 0

class FlightListResponse(BaseModel):
    total: int
    items: List[FlightResponse]

# Crew Assignment Schemas
class CrewAssignmentResponse(BaseModel):
    id: int
    flight_id: int
    crew_member_id: int
    crew_member_name: str
    role: CrewRoleEnum
    position: Optional[str]
    
    class Config:
        from_attributes = True

class CrewAssignmentListResponse(BaseModel):
    total: int
    items: List[CrewAssignmentResponse]

# Maintenance Schemas
class MaintenanceRecordCreate(BaseModel):
    aircraft_id: int
    maintenance_type: MaintenanceTypeEnum
    description: str
    scheduled_date: date
    expected_finish_date: date
    technician_name: Optional[str] = None
    cost: float = 0

class MaintenanceRecordResponse(BaseModel):
    id: int
    aircraft_id: int
    aircraft_registration: Optional[str] = None
    maintenance_type: MaintenanceTypeEnum
    description: str
    scheduled_date: Optional[date]
    expected_finish_date: Optional[date]
    completion_date: Optional[datetime]
    completed: bool
    technician_name: Optional[str]
    cost: float
    created_at: datetime
    
    class Config:
        from_attributes = True

class MaintenanceRecordListResponse(BaseModel):
    total: int
    items: List[MaintenanceRecordResponse]

# Statistics Schemas
class StatisticsResponse(BaseModel):
    total_aircraft: int
    operational_aircraft: int
    total_crew: int
    available_crew: int
    total_flights: int
    delayed_flights: int
    maintenance_due: int

class MessageResponse(BaseModel):
    message: str
    detail: Optional[str] = None

# Booking Schemas
class BookingCreate(BaseModel):
    flight_id: int
    passenger_name: str
    passenger_email: str
    passenger_phone: Optional[str] = None
    departure_place: str
    arrival_place: str
    num_passengers: int = 1
    ticket_class: str = "economy"  # economy, business, premium

class BookingResponse(BaseModel):
    id: int
    flight_id: int
    booking_reference: str
    passenger_name: str
    passenger_email: str
    passenger_phone: Optional[str]
    departure_place: str
    arrival_place: str
    num_passengers: int
    ticket_class: str
    price: float
    seat_number: Optional[str]
    status: str
    created_at: datetime
    # Nested flight information (returned by API on booking creation and listings)
    flight: Optional[FlightResponse] = None
    
    class Config:
        from_attributes = True

class BookingListResponse(BaseModel):
    total: int
    items: List[BookingResponse]