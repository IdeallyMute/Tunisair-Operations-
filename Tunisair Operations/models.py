"""
Tunisair Operations - Database Models
SQLAlchemy ORM models for flight operations management
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Enum, Text, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime, timezone

# datetime.utcnow() is deprecated in recent Python versions.
# Keep naive UTC timestamps for SQLite compatibility.
def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)
import enum

Base = declarative_base()

# Enumerations
class FlightStatusEnum(enum.Enum):
    SCHEDULED = "scheduled"
    BOARDING = "boarding"
    DEPARTED = "departed"
    IN_FLIGHT = "in_flight"
    LANDED = "landed"
    DELAYED = "delayed"
    CANCELLED = "cancelled"

class CrewRoleEnum(enum.Enum):
    CAPTAIN = "captain"
    FIRST_OFFICER = "first_officer"
    CABIN_CHIEF = "cabin_chief"
    FLIGHT_ATTENDANT = "flight_attendant"

class CrewStatusEnum(enum.Enum):
    AVAILABLE = "available"
    ON_DUTY = "on_duty"
    OFF_DUTY = "off_duty"
    ON_LEAVE = "on_leave"

class AircraftStatusEnum(enum.Enum):
    OPERATIONAL = "operational"
    MAINTENANCE = "maintenance"
    GROUNDED = "grounded"

class MaintenanceTypeEnum(enum.Enum):
    ROUTINE = "routine"
    SCHEDULED = "scheduled"
    UNSCHEDULED = "unscheduled"
    INSPECTION = "inspection"

# User Model
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    full_name = Column(String)
    phone_number = Column(String)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=utcnow)
    
    bookings = relationship("Booking", back_populates="user")

class Booking(Base):
    __tablename__ = "bookings"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    flight_id = Column(Integer, ForeignKey("flights.id"), nullable=False)
    booking_reference = Column(String, unique=True, index=True, nullable=False)
    passenger_name = Column(String, nullable=False)
    passenger_email = Column(String, nullable=False)
    passenger_phone = Column(String)
    departure_place = Column(String, nullable=False)
    arrival_place = Column(String, nullable=False)
    num_passengers = Column(Integer, default=1)
    ticket_class = Column(String, default="economy")  # economy, business, premium
    price = Column(Float, default=0)
    seat_number = Column(String)
    status = Column(String, default="confirmed")  # confirmed, cancelled
    created_at = Column(DateTime, default=utcnow)
    
    user = relationship("User", back_populates="bookings")
    flight = relationship("Flight", back_populates="bookings")

# Aircraft Model
class Aircraft(Base):
    __tablename__ = "aircraft"
    
    id = Column(Integer, primary_key=True, index=True)
    registration = Column(String, unique=True, nullable=False, index=True)
    model = Column(String, nullable=False)
    manufacturer = Column(String, nullable=False)
    year_manufactured = Column(Integer)
    total_seats = Column(Integer, nullable=False)
    status = Column(Enum(AircraftStatusEnum), default=AircraftStatusEnum.OPERATIONAL)
    total_flight_hours = Column(Float, default=0)
    last_maintenance_date = Column(Date)
    next_maintenance_date = Column(Date)
    current_location = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=utcnow)
    
    flights = relationship("Flight", back_populates="aircraft")
    maintenance_records = relationship("MaintenanceRecord", back_populates="aircraft")

# Crew Member Model
class CrewMember(Base):
    __tablename__ = "crew_members"
    
    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(String, unique=True, nullable=False, index=True)
    full_name = Column(String, nullable=False)
    role = Column(Enum(CrewRoleEnum), nullable=False)
    license_number = Column(String)
    total_flight_hours = Column(Float, default=0)
    status = Column(Enum(CrewStatusEnum), default=CrewStatusEnum.AVAILABLE)
    base_location = Column(String, default="TUN")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=utcnow)
    
    assignments = relationship("CrewAssignment", back_populates="crew_member")

# Flight Model
class Flight(Base):
    __tablename__ = "flights"
    
    id = Column(Integer, primary_key=True, index=True)
    flight_number = Column(String, nullable=False, index=True)
    aircraft_id = Column(Integer, ForeignKey("aircraft.id"), nullable=False)
    origin = Column(String, nullable=False)
    destination = Column(String, nullable=False)
    scheduled_departure = Column(DateTime, nullable=False)
    scheduled_arrival = Column(DateTime, nullable=False)
    actual_departure = Column(DateTime)
    actual_arrival = Column(DateTime)
    status = Column(Enum(FlightStatusEnum), default=FlightStatusEnum.SCHEDULED)
    gate = Column(String)
    delay_minutes = Column(Integer, default=0)
    passengers_booked = Column(Integer, default=0)
    base_price = Column(Float, default=150.0)  # Base price in EUR
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=utcnow)
    
    aircraft = relationship("Aircraft", back_populates="flights")
    crew_assignments = relationship("CrewAssignment", back_populates="flight")
    bookings = relationship("Booking", back_populates="flight")

# Crew Assignment Model
class CrewAssignment(Base):
    __tablename__ = "crew_assignments"
    
    id = Column(Integer, primary_key=True, index=True)
    flight_id = Column(Integer, ForeignKey("flights.id"), nullable=False)
    crew_member_id = Column(Integer, ForeignKey("crew_members.id"), nullable=False)
    role = Column(Enum(CrewRoleEnum), nullable=False)
    position = Column(String)
    confirmed = Column(Boolean, default=True)
    created_at = Column(DateTime, default=utcnow)
    
    flight = relationship("Flight", back_populates="crew_assignments")
    crew_member = relationship("CrewMember", back_populates="assignments")

# Maintenance Record Model
class MaintenanceRecord(Base):
    __tablename__ = "maintenance_records"
    
    id = Column(Integer, primary_key=True, index=True)
    aircraft_id = Column(Integer, ForeignKey("aircraft.id"), nullable=False)
    maintenance_type = Column(Enum(MaintenanceTypeEnum), nullable=False)
    description = Column(Text, nullable=False)
    scheduled_date = Column(Date)
    expected_finish_date = Column(Date)
    completion_date = Column(DateTime)
    completed = Column(Boolean, default=False)
    technician_name = Column(String)
    cost = Column(Float, default=0)
    created_at = Column(DateTime, default=utcnow)
    
    aircraft = relationship("Aircraft", back_populates="maintenance_records")
