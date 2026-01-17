"""
============================================================================
TUNISAIR OPERATIONS API
============================================================================
Flight operations management system 
Run: python main.py
API Docs: http://localhost:8000/docs
FrontEnd : Index.html
============================================================================
"""

# ---------------------------------------------------------------------------
# Bootstrap: ensure dependencies are installed when running `python main.py`.
# This avoids the common "ModuleNotFoundError" experience on first run, while
# keeping subsequent starts fast (a marker file prevents repeated installs).
# ---------------------------------------------------------------------------

import os
import sys
import subprocess

if __name__ == "__main__" and os.getenv("TUNISAIR_SKIP_PIP", "0") != "1":
    try:
        import sqlalchemy  # noqa: F401
    except ModuleNotFoundError:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        marker = os.path.join(base_dir, ".requirements_installed")
        req = os.path.join(base_dir, "requirements.txt")
        if os.path.exists(req) and not os.path.exists(marker):
            print("✓ Installing requirements (first run)...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req])
            with open(marker, "w", encoding="utf-8") as f:
                f.write("installed")
            # Restart the process so imports below succeed.
            os.execv(sys.executable, [sys.executable] + sys.argv)

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import create_engine, func, and_
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.orm import joinedload
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta, date, timezone

# NOTE: datetime.utcnow() is deprecated in recent Python versions.
# This helper keeps the original behaviour (naive UTC datetimes) while avoiding the deprecation.
def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)

from typing import List, Optional
import uvicorn
import csv
import os
import secrets
import requests

from models import (
    Base, User, Aircraft, CrewMember, Flight, CrewAssignment, MaintenanceRecord, Booking,
    FlightStatusEnum, CrewRoleEnum, CrewStatusEnum, AircraftStatusEnum, MaintenanceTypeEnum
)
from schemas import (
    UserCreate, UserLogin, UserResponse, Token,
    AircraftResponse, AircraftListResponse,
    CrewMemberResponse, CrewMemberListResponse,
    FlightResponse, FlightDetailResponse, FlightListResponse,
    CrewAssignmentResponse, CrewAssignmentListResponse,
    MaintenanceRecordCreate, MaintenanceRecordResponse, MaintenanceRecordListResponse,
    StatisticsResponse, MessageResponse,
    BookingCreate, BookingResponse, BookingListResponse
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Always store the SQLite DB next to this file so running from different working
# directories (VS Code, double-click, etc.) doesn't create multiple databases.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'tunisair.db')}"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False}, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

SECRET_KEY = "tunisair-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Tunisair Operations API",
    description="Flight operations management system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # When the frontend is opened via file://, the browser uses an opaque "null" origin.
    # Using allow_credentials=True together with wildcard origins is blocked by browsers.
    # Tokens are sent via the Authorization header, so credentials are not required.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# AUTHENTICATION FUNCTIONS
# ============================================================================

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def create_refresh_token(data: dict):
    to_encode = data.copy()
    expire = utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str, token_type: str = "access"):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != token_type:
            raise HTTPException(status_code=401, detail="Invalid token type")
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Could not validate credentials")

# ============================================================================
# DATABASE DEPENDENCIES
# ============================================================================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    token = credentials.credentials
    payload = verify_token(token, token_type="access")
    username: str = payload.get("sub")
    if username is None:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# ============================================================================
# INITIALIZATION FUNCTIONS
# ============================================================================

def init_db():
    Base.metadata.create_all(bind=engine)
    # Lightweight migration helpers (SQLite)
    # Keep the project simple (no Alembic) while allowing minor schema evolution.
    with engine.connect() as conn:
        try:
            # Add users.phone_number if missing
            cols = [row[1] for row in conn.exec_driver_sql("PRAGMA table_info(users)").fetchall()]
            if "phone_number" not in cols:
                conn.exec_driver_sql("ALTER TABLE users ADD COLUMN phone_number VARCHAR")
        except Exception as e:
            print(f"⚠ DB migration warning: {e}")
    print("✓ Database initialized")

def seed_demo_data(db: Session):
    """Seed extra mock users and bookings for dashboard clarity (concept project)."""
    try:
        # Users
        user_count = db.query(User).filter(User.username != "admin").count()
        # Make user signups look realistic: variable day-to-day with an overall upward trend.
        # We seed (and also normalize existing mock users) across the next 8+ months so the
        # "User Growth" chart is always meaningful.
        import random
        from datetime import timedelta

        def _pick_signup_date(start_dt: datetime, end_dt: datetime) -> datetime:
            """Return a timezone-aware UTC datetime biased toward more recent dates."""
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=timezone.utc)
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=timezone.utc)
            span_days = max(1, (end_dt - start_dt).days)

            # Weighted selection: later days more likely (trend up), with some noise.
            # weights ~ (day_index + 1)^p where p>1 biases toward recent.
            p = 1.6
            weights = [(i + 1) ** p for i in range(span_days + 1)]
            day_offset = random.choices(range(span_days + 1), weights=weights, k=1)[0]
            # random time within the day
            seconds = random.randint(0, 86399)
            return start_dt + timedelta(days=day_offset, seconds=seconds)

        # Seed window: include BOTH history and future.
        # Requirement: 2-year span between historical and future data affecting graphs.
        # We use 1 year back + 1 year forward (total span ~2 years).
        now_utc = datetime.now(timezone.utc)
        start_seed = now_utc - timedelta(days=365)
        end_seed = now_utc + timedelta(days=365)

        # Ensure we always have enough users for the dashboard (>= 527 non-admin users)
        # NOTE: this is demo data only. Make it fast so startup is responsive.
        target_users = 600
        if user_count < 527:
            import string
            domains = ["gmail.com", "outlook.com", "yahoo.com", "proton.me"]
            first_names = ["Yassine", "Sana", "Aymen", "Mariem", "Oussama", "Ines", "Ahmed", "Rania", "Houssem", "Rim", "Khalil", "Nour", "Salma", "Mehdi"]
            last_names = ["Ben Ali", "Trabelsi", "Haddad", "Bouazizi", "Mansouri", "Jebali", "Kammoun", "Sassi", "Chaabane", "Hamdi", "Karray", "Ben Salah", "Brahmi", "Gharbi"]

            # Pre-compute a single bcrypt hash (bcrypt is intentionally slow).
            # Re-using the hash keeps login behaviour correct while avoiding hundreds
            # of expensive hashing operations at startup.
            demo_password_hash = hash_password("user12345")

            # Create users until we reach target_users (best-effort; avoids duplicates)
            i = 1
            created = 0
            # Avoid COUNT() on every iteration (very slow with SQLite)
            current = user_count
            while (current < target_users) and i < (target_users * 3):
                uname = f"user{i}"
                i += 1
                if db.query(User).filter(User.username == uname).first():
                    continue

                fn = random.choice(first_names)
                ln = random.choice(last_names)
                full_name = f"{fn} {ln}"

                u = User(
                    username=uname,
                    email=f"{uname}@{random.choice(domains)}",
                    phone_number=f"+216 {random.randint(20000000, 99999999)}",
                    full_name=full_name,
                    password_hash=demo_password_hash,
                    is_active=True
                )
                # Important: created_at must vary across days and trend upward
                u.created_at = _pick_signup_date(start_seed, end_seed)
                db.add(u)
                created += 1
                current += 1
            db.commit()

        # Normalize existing mock users (common issue: all seeded at the same moment)
        # so the growth chart doesn't look flat/constant.
        try:
            existing = db.query(User).filter(User.username.notin_(["admin", "test"])) .all()
            if existing:
                # If >60% share the same date (day-level), re-distribute.
                from collections import Counter
                days = [u.created_at.date() if u.created_at else None for u in existing]
                c = Counter(days)
                most_common_day, most_common_count = c.most_common(1)[0]
                if most_common_day is not None and most_common_count / max(1, len(existing)) > 0.60:
                    for u in existing:
                        u.created_at = _pick_signup_date(start_seed, end_seed)
                    db.commit()
        except Exception:
            # Best-effort; do not block app start.
            db.rollback()

        # Bookings
        booking_count = db.query(Booking).count()
        # Ensure enough bookings for charts with strong day-to-day variability.
        target_bookings = 1800
        if booking_count < 800:
            import random
            from datetime import timedelta
            users = db.query(User).filter(User.username != "admin").all()
            flights = db.query(Flight).all()
            if users and flights:
                # Helper: choose a date with realistic seasonality and daily noise
                start_seed_b = now_utc - timedelta(days=365)
                end_seed_b = now_utc + timedelta(days=365)
                span_days_b = max(1, (end_seed_b - start_seed_b).days)

                def _pick_booking_date() -> datetime:
                    # Weekly seasonality: weekends busier; plus gradual growth toward future
                    # Weight = base + weekend_boost + trend
                    weights = []
                    for day in range(span_days_b + 1):
                        dt = start_seed_b + timedelta(days=day)
                        weekday = dt.weekday()  # 0=Mon
                        weekend_boost = 1.35 if weekday in (4, 5, 6) else 1.0
                        trend = 0.6 + (day / max(1, span_days_b)) * 1.0
                        noise = random.uniform(0.85, 1.15)
                        weights.append(max(0.05, weekend_boost * trend * noise))
                    day_offset = random.choices(range(span_days_b + 1), weights=weights, k=1)[0]
                    seconds = random.randint(0, 86399)
                    return start_seed_b + timedelta(days=day_offset, seconds=seconds)

                to_create = max(0, target_bookings - booking_count)
                to_create = min(to_create, 2500)  # safety
                for _ in range(to_create):
                    user = random.choice(users)
                    flight = random.choice(flights)
                    # skip impossible
                    if not flight:
                        continue
                    num_pax = random.choice([1,1,2,3])
                    cls = random.choice(["economy","premium","business"])
                    mult = {"economy":1.0,"premium":1.5,"business":2.5}.get(cls,1.0)
                    # Add more variability to price day-to-day and by demand
                    demand_factor = random.uniform(0.85, 1.25)
                    price = float((flight.base_price or 150.0) * mult * num_pax * demand_factor)
                    status = random.choices(
                        population=["confirmed", "cancelled", "pending"],
                        weights=[0.82, 0.10, 0.08],
                        k=1
                    )[0]
                    b = Booking(
                        user_id=user.id,
                        flight_id=flight.id,
                        booking_reference=f"TUDEMO{secrets.token_hex(3).upper()}",
                        passenger_name=user.full_name or user.username,
                        passenger_email=user.email,
                        passenger_phone=user.phone_number,
                        departure_place=flight.origin,
                        arrival_place=flight.destination,
                        num_passengers=num_pax,
                        ticket_class=cls,
                        price=price,
                        seat_number=None,
                        status=status
                    )
                    # Spread bookings across 1 year history + 1 year future
                    b.created_at = _pick_booking_date()
                    db.add(b)
                    # bump passengers_booked a bit (safe)
                    try:
                        if status == "confirmed":
                            flight.passengers_booked = int((flight.passengers_booked or 0) + num_pax)
                    except Exception:
                        pass
                db.commit()
    except Exception as e:
        db.rollback()
        print(f"Demo seed error: {e}")

def seed_admin_user(db: Session):
    # Create admin user
    admin = db.query(User).filter(User.username == "admin").first()
    if not admin:
        admin = User(
            username="admin",
            email="admin@tunisair.com",
            password_hash=hash_password("admin123"),
            full_name="Operations Admin",
            is_active=True,
            is_admin=True
        )
        db.add(admin)
        db.commit()
        print("✓ Created admin user (username: admin, password: admin123)")
    
    # Create test user
    test_user = db.query(User).filter(User.username == "test").first()
    if not test_user:
        test_user = User(
            username="test",
            email="test@tunisair.com",
            password_hash=hash_password("test"),
            full_name="Mahdy",
            is_active=True,
            is_admin=False
        )
        db.add(test_user)
        db.commit()
        print("✓ Created test user (username: test, password: test, name: Mahdy)")

def load_aircraft_from_csv(db: Session):
    count = db.query(Aircraft).count()
    if count > 0:
        print(f"✓ Database already has {count} aircraft")
        return
    
    csv_path = os.path.join(BASE_DIR, "aircraft.csv")
    if not os.path.exists(csv_path):
        print("✗ aircraft.csv not found")
        return
    
    added = 0
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            aircraft = Aircraft(
                registration=row['registration'],
                model=row['model'],
                manufacturer=row['manufacturer'],
                year_manufactured=int(row['year_manufactured']) if row['year_manufactured'] else None,
                total_seats=int(row['total_seats']),
                status=AircraftStatusEnum(row['status']),
                total_flight_hours=float(row['total_flight_hours']),
                last_maintenance_date=datetime.strptime(row['last_maintenance_date'], '%Y-%m-%d').date() if row['last_maintenance_date'] else None,
                next_maintenance_date=datetime.strptime(row['next_maintenance_date'], '%Y-%m-%d').date() if row['next_maintenance_date'] else None,
                current_location=row['current_location'],
                is_active=True
            )
            db.add(aircraft)
            added += 1
    db.commit()
    print(f"✓ Loaded {added} aircraft from CSV")

def load_crew_from_csv(db: Session):
    count = db.query(CrewMember).count()
    if count > 0:
        print(f"✓ Database already has {count} crew members")
        return
    
    csv_path = os.path.join(BASE_DIR, "crew.csv")
    if not os.path.exists(csv_path):
        print("✗ crew.csv not found")
        return
    
    added = 0
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            crew = CrewMember(
                employee_id=row['employee_id'],
                full_name=row['full_name'],
                role=CrewRoleEnum(row['role']),
                license_number=row['license_number'] if row['license_number'] else None,
                total_flight_hours=float(row['total_flight_hours']),
                status=CrewStatusEnum(row['status']),
                base_location=row['base_location'],
                is_active=True
            )
            db.add(crew)
            added += 1
    db.commit()
    print(f"✓ Loaded {added} crew members from CSV")

def load_flights_from_csv(db: Session):
    count = db.query(Flight).count()
    if count > 0:
        print(f"✓ Database already has {count} flights")
        # If all flights are in the past (common during dev), reload from CSV
        try:
            latest = db.query(Flight).order_by(Flight.scheduled_departure.desc()).first()
            if latest and latest.scheduled_departure and latest.scheduled_departure.date() < utcnow().date():
                # keep a 1-day grace
                if (utcnow().date() - latest.scheduled_departure.date()).days >= 1:
                    db.query(Booking).delete()
                    db.query(Flight).delete()
                    db.commit()
                    print("↻ Existing flights were outdated, reloading flights from CSV")
                    count = 0
        except Exception as _:
            pass
        # Update existing flights with prices if they don't have them
        flights_to_update = db.query(Flight).filter(Flight.base_price == 150.0).all()
        if flights_to_update:
            import random
            for flight in flights_to_update:
                # Calculate price based on route (more realistic pricing)
                route_multipliers = {
                    'short': (0.8, 1.2),  # 120-180 EUR
                    'medium': (1.2, 1.8),  # 180-270 EUR
                    'long': (2.0, 3.5),    # 300-525 EUR
                }
                # Determine route length
                long_haul_destinations = ['Dubai', 'Cairo', 'Istanbul', 'Abu Dhabi']
                medium_haul = ['London', 'Frankfurt', 'Madrid', 'Barcelona']
                
                if any(dest in [flight.origin, flight.destination] for dest in long_haul_destinations):
                    mult_range = route_multipliers['long']
                elif any(dest in [flight.origin, flight.destination] for dest in medium_haul):
                    mult_range = route_multipliers['medium']
                else:
                    mult_range = route_multipliers['short']
                
                # Add some randomization for realism
                multiplier = random.uniform(mult_range[0], mult_range[1])
                flight.base_price = round(150 * multiplier, 2)
            
            db.commit()
            print(f"✓ Updated {len(flights_to_update)} flights with variable pricing")
        return
    
    csv_path = os.path.join(BASE_DIR, "flights.csv")
    if not os.path.exists(csv_path):
        print("✗ flights.csv not found")
        return
    
    import random
    added = 0
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            aircraft = db.query(Aircraft).filter(Aircraft.registration == row['aircraft_registration']).first()
            if not aircraft:
                continue
            
            # Calculate variable price based on route
            route_multipliers = {
                'short': (0.8, 1.2),  # 120-180 EUR
                'medium': (1.2, 1.8),  # 180-270 EUR
                'long': (2.0, 3.5),    # 300-525 EUR
            }
            
            # Determine route length
            long_haul_destinations = ['Dubai', 'Cairo', 'Istanbul', 'Abu Dhabi']
            medium_haul = ['London', 'Frankfurt', 'Madrid', 'Barcelona']
            
            if any(dest in [row['origin'], row['destination']] for dest in long_haul_destinations):
                mult_range = route_multipliers['long']
            elif any(dest in [row['origin'], row['destination']] for dest in medium_haul):
                mult_range = route_multipliers['medium']
            else:
                mult_range = route_multipliers['short']
            
            # Add some randomization for realism
            multiplier = random.uniform(mult_range[0], mult_range[1])
            base_price = round(150 * multiplier, 2)
                
            flight = Flight(
                flight_number=row['flight_number'],
                aircraft_id=aircraft.id,
                origin=row['origin'],
                destination=row['destination'],
                scheduled_departure=datetime.strptime(row['scheduled_departure'], '%Y-%m-%d %H:%M:%S'),
                scheduled_arrival=datetime.strptime(row['scheduled_arrival'], '%Y-%m-%d %H:%M:%S'),
                actual_departure=datetime.strptime(row['actual_departure'], '%Y-%m-%d %H:%M:%S') if row['actual_departure'] else None,
                actual_arrival=datetime.strptime(row['actual_arrival'], '%Y-%m-%d %H:%M:%S') if row['actual_arrival'] else None,
                status=FlightStatusEnum(row['status']),
                gate=row['gate'] if row['gate'] else None,
                delay_minutes=int(row['delay_minutes']),
                passengers_booked=int(row['passengers_booked']),
                base_price=base_price,
                is_active=True
            )
            db.add(flight)
            added += 1
    db.commit()
    print(f"✓ Loaded {added} flights from CSV with variable pricing")

def load_crew_assignments_from_csv(db: Session):
    count = db.query(CrewAssignment).count()
    if count > 0:
        print(f"✓ Database already has {count} crew assignments")
        return
    
    csv_path = os.path.join(BASE_DIR, "crew_assignments.csv")
    if not os.path.exists(csv_path):
        print("✗ crew_assignments.csv not found")
        return
    
    added = 0
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            flight = db.query(Flight).filter(Flight.flight_number == row['flight_number']).first()
            crew = db.query(CrewMember).filter(CrewMember.employee_id == row['employee_id']).first()
            
            if not flight or not crew:
                continue
                
            assignment = CrewAssignment(
                flight_id=flight.id,
                crew_member_id=crew.id,
                role=CrewRoleEnum(row['role']),
                position=row['position'] if row['position'] else None,
                confirmed=True
            )
            db.add(assignment)
            added += 1
    db.commit()
    print(f"✓ Loaded {added} crew assignments from CSV")

def load_maintenance_from_csv(db: Session):
    count = db.query(MaintenanceRecord).count()
    if count > 0:
        print(f"✓ Database already has {count} maintenance records")
        return
    
    csv_path = os.path.join(BASE_DIR, "maintenance.csv")
    if not os.path.exists(csv_path):
        print("✗ maintenance.csv not found")
        return
    
    added = 0
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            aircraft = db.query(Aircraft).filter(Aircraft.registration == row['aircraft_registration']).first()
            if not aircraft:
                continue
                
            record = MaintenanceRecord(
                aircraft_id=aircraft.id,
                maintenance_type=MaintenanceTypeEnum(row['maintenance_type']),
                description=row['description'],
                scheduled_date=datetime.strptime(row['scheduled_date'], '%Y-%m-%d').date() if row['scheduled_date'] else None,
                expected_finish_date=datetime.strptime(row['expected_finish_date'], '%Y-%m-%d').date() if row['expected_finish_date'] else None,
                completion_date=datetime.strptime(row['completion_date'], '%Y-%m-%d %H:%M:%S') if row['completion_date'] else None,
                completed=row['completed'].lower() == 'true',
                technician_name=row['technician_name'] if row['technician_name'] else None,
                cost=float(row['cost']) if row['cost'] else 0
            )
            db.add(record)
            added += 1
    db.commit()
    print(f"✓ Loaded {added} maintenance records from CSV")

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Serve the frontend (index.html) from the same origin as the API."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return FileResponse(os.path.join(base_dir, "index.html"))

@app.get("/api/status")
async def api_status():
    return {
        "application": "Tunisair Operations API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

# Authentication Endpoints
@app.post("/api/auth/register", response_model=Token, status_code=201)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == user_data.username).first():
        raise HTTPException(status_code=400, detail="Username already registered")
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user = User(
        username=user_data.username,
        email=user_data.email,
        phone_number=getattr(user_data, 'phone_number', None),
        password_hash=hash_password(user_data.password),
        full_name=user_data.full_name,
        is_active=True
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    access_token = create_access_token(data={"sub": user.username})
    refresh_token = create_refresh_token(data={"sub": user.username})
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        user=UserResponse.from_orm(user)
    )

@app.post("/api/auth/login", response_model=Token)
async def login(credentials: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == credentials.username).first()
    
    if not user or not verify_password(credentials.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    access_token = create_access_token(data={"sub": user.username})
    refresh_token = create_refresh_token(data={"sub": user.username})
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        user=UserResponse.from_orm(user)
    )

@app.get("/api/auth/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_active_user)):
    return UserResponse.from_orm(current_user)

# Aircraft Endpoints
@app.get("/api/aircraft", response_model=AircraftListResponse)
async def get_aircraft(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    aircraft = db.query(Aircraft).filter(Aircraft.is_active == True).all()
    return AircraftListResponse(
        total=len(aircraft),
        items=[AircraftResponse.from_orm(a) for a in aircraft]
    )

@app.get("/api/aircraft/{aircraft_id}", response_model=AircraftResponse)
async def get_aircraft_detail(
    aircraft_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    aircraft = db.query(Aircraft).filter(Aircraft.id == aircraft_id).first()
    if not aircraft:
        raise HTTPException(status_code=404, detail="Aircraft not found")
    return AircraftResponse.from_orm(aircraft)

# Crew Endpoints
@app.get("/api/crew", response_model=CrewMemberListResponse)
async def get_crew(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    crew = db.query(CrewMember).filter(CrewMember.is_active == True).all()
    return CrewMemberListResponse(
        total=len(crew),
        items=[CrewMemberResponse.from_orm(c) for c in crew]
    )

@app.get("/api/crew/{crew_id}", response_model=CrewMemberResponse)
async def get_crew_detail(
    crew_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    crew = db.query(CrewMember).filter(CrewMember.id == crew_id).first()
    if not crew:
        raise HTTPException(status_code=404, detail="Crew member not found")
    return CrewMemberResponse.from_orm(crew)

# Flight Endpoints
@app.get("/api/flights", response_model=FlightListResponse)
async def get_flights(
    date: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    query = db.query(Flight).filter(Flight.is_active == True)
    
    if date:
        target_date = datetime.strptime(date, '%Y-%m-%d').date()
        query = query.filter(func.date(Flight.scheduled_departure) == target_date)
    
    flights = query.order_by(Flight.scheduled_departure).all()
    return FlightListResponse(
        total=len(flights),
        items=[FlightResponse.from_orm(f) for f in flights]
    )

@app.get("/api/flights/{flight_id}", response_model=FlightDetailResponse)
async def get_flight_detail(
    flight_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    flight = db.query(Flight).filter(Flight.id == flight_id).first()
    if not flight:
        raise HTTPException(status_code=404, detail="Flight not found")
    
    response = FlightDetailResponse.from_orm(flight)
    response.aircraft = AircraftResponse.from_orm(flight.aircraft) if flight.aircraft else None
    response.crew_count = len(flight.crew_assignments)
    return response

# Crew Assignment Endpoints
@app.get("/api/assignments/flight/{flight_id}", response_model=CrewAssignmentListResponse)
async def get_flight_crew(
    flight_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    assignments = db.query(CrewAssignment).filter(CrewAssignment.flight_id == flight_id).all()
    
    items = []
    for assignment in assignments:
        crew = db.query(CrewMember).filter(CrewMember.id == assignment.crew_member_id).first()
        items.append({
            "id": assignment.id,
            "flight_id": assignment.flight_id,
            "crew_member_id": assignment.crew_member_id,
            "crew_member_name": crew.full_name if crew else "Unknown",
            "role": assignment.role,
            "position": assignment.position
        })
    
    return CrewAssignmentListResponse(total=len(items), items=items)

# Maintenance Endpoints
@app.post("/api/maintenance", response_model=MaintenanceRecordResponse, status_code=201)
async def create_maintenance(
    maintenance: MaintenanceRecordCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    aircraft = db.query(Aircraft).filter(Aircraft.id == maintenance.aircraft_id).first()
    if not aircraft:
        raise HTTPException(status_code=404, detail="Aircraft not found")
    
    record = MaintenanceRecord(
        aircraft_id=maintenance.aircraft_id,
        maintenance_type=maintenance.maintenance_type,
        description=maintenance.description,
        scheduled_date=maintenance.scheduled_date,
        expected_finish_date=maintenance.expected_finish_date,
        technician_name=maintenance.technician_name,
        cost=maintenance.cost,
        completed=False
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    
    response = MaintenanceRecordResponse.from_orm(record)
    response.aircraft_registration = aircraft.registration
    return response

@app.get("/api/maintenance", response_model=MaintenanceRecordListResponse)
async def get_maintenance(
    date: Optional[str] = None,
    completed: Optional[bool] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    query = db.query(MaintenanceRecord)
    
    if date:
        target_date = datetime.strptime(date, '%Y-%m-%d').date()
        # Show maintenance that is in progress on this date (scheduled_date <= target_date <= expected_finish_date)
        # OR completed on this date
        query = query.filter(
            and_(
                MaintenanceRecord.scheduled_date <= target_date,
                MaintenanceRecord.expected_finish_date >= target_date
            )
        )
    
    if completed is not None:
        query = query.filter(MaintenanceRecord.completed == completed)
    
    records = query.order_by(MaintenanceRecord.created_at.desc()).all()
    
    items = []
    for record in records:
        aircraft = db.query(Aircraft).filter(Aircraft.id == record.aircraft_id).first()
        response_dict = {
            "id": record.id,
            "aircraft_id": record.aircraft_id,
            "aircraft_registration": aircraft.registration if aircraft else None,
            "maintenance_type": record.maintenance_type,
            "description": record.description,
            "scheduled_date": record.scheduled_date,
            "expected_finish_date": record.expected_finish_date,
            "completion_date": record.completion_date,
            "completed": record.completed,
            "technician_name": record.technician_name,
            "cost": record.cost,
            "created_at": record.created_at
        }
        items.append(response_dict)
    
    return MaintenanceRecordListResponse(total=len(items), items=items)

# Statistics Endpoint
@app.get("/api/stats", response_model=StatisticsResponse)
async def get_statistics(
    date: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    target_date = datetime.strptime(date, '%Y-%m-%d').date() if date else utcnow().date()
    
    total_aircraft = db.query(func.count(Aircraft.id)).filter(Aircraft.is_active == True).scalar()
    operational_aircraft = db.query(func.count(Aircraft.id)).filter(
        Aircraft.is_active == True,
        Aircraft.status == AircraftStatusEnum.OPERATIONAL
    ).scalar()
    
    total_crew = db.query(func.count(CrewMember.id)).filter(CrewMember.is_active == True).scalar()
    available_crew = db.query(func.count(CrewMember.id)).filter(
        CrewMember.is_active == True,
        CrewMember.status == CrewStatusEnum.AVAILABLE
    ).scalar()
    
    total_flights = db.query(func.count(Flight.id)).filter(
        Flight.is_active == True,
        func.date(Flight.scheduled_departure) == target_date
    ).scalar()
    
    delayed_flights = db.query(func.count(Flight.id)).filter(
        Flight.is_active == True,
        func.date(Flight.scheduled_departure) == target_date,
        Flight.delay_minutes > 0
    ).scalar()
    
    maintenance_due = db.query(func.count(Aircraft.id)).filter(
        Aircraft.is_active == True,
        Aircraft.next_maintenance_date <= target_date
    ).scalar()
    
    return StatisticsResponse(
        total_aircraft=total_aircraft,
        operational_aircraft=operational_aircraft,
        total_crew=total_crew,
        available_crew=available_crew,
        total_flights=total_flights,
        delayed_flights=delayed_flights,
        maintenance_due=maintenance_due
    )

# ============================================================================
# ADMIN KPI & ANALYTICS ENDPOINTS
# ============================================================================

@app.get("/api/admin/kpis")
async def get_admin_kpis(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get comprehensive KPI metrics for admin dashboard"""
    if not _is_admin(current_user):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        # Get total counts
        total_users = db.query(func.count(User.id)).filter(User.is_admin == False).scalar()
        total_bookings = db.query(func.count(Booking.id)).scalar()
        confirmed_bookings = db.query(func.count(Booking.id)).filter(Booking.status == "confirmed").scalar()
        total_revenue = db.query(func.sum(Booking.price)).filter(Booking.status == "confirmed").scalar() or 0
        
        total_flights = db.query(func.count(Flight.id)).filter(Flight.is_active == True).scalar()
        total_aircraft = db.query(func.count(Aircraft.id)).filter(Aircraft.is_active == True).scalar()
        operational_aircraft = db.query(func.count(Aircraft.id)).filter(
            Aircraft.is_active == True,
            Aircraft.status == AircraftStatusEnum.OPERATIONAL
        ).scalar()
        
        total_crew = db.query(func.count(CrewMember.id)).filter(CrewMember.is_active == True).scalar()
        available_crew = db.query(func.count(CrewMember.id)).filter(
            CrewMember.is_active == True,
            CrewMember.status == CrewStatusEnum.AVAILABLE
        ).scalar()
        
        # Get average booking value
        avg_booking_value = total_revenue / confirmed_bookings if confirmed_bookings > 0 else 0
        
        # Get total passengers
        total_passengers = db.query(func.sum(Booking.num_passengers)).filter(Booking.status == "confirmed").scalar() or 0
        
        # Calculate occupancy rate (assuming average 200 seats per flight)
        total_seats_available = total_flights * 200
        occupancy_rate = (total_passengers / total_seats_available * 100) if total_seats_available > 0 else 0
        
        return {
            "total_users": total_users,
            "total_bookings": total_bookings,
            "confirmed_bookings": confirmed_bookings,
            "total_revenue": round(total_revenue, 2),
            "total_flights": total_flights,
            "total_aircraft": total_aircraft,
            "operational_aircraft": operational_aircraft,
            "total_crew": total_crew,
            "available_crew": available_crew,
            "avg_booking_value": round(avg_booking_value, 2),
            "total_passengers": total_passengers,
            "occupancy_rate": round(occupancy_rate, 2)
        }
    except Exception as e:
        print(f"KPI error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching KPIs: {str(e)}")



def _is_admin(user) -> bool:
    """Runtime-safe admin check (avoids SQLAlchemy typing warnings)."""
    try:
        return bool(getattr(user, 'is_admin', False))
    except Exception:
        return False


def _parse_iso_date(s: str):
    """Parse YYYY-MM-DD to datetime.date; returns None on failure."""
    try:
        from datetime import date
        return date.fromisoformat(s)
    except Exception:
        return None


def _date_range_from_params(days: int | None, start_date: str | None, end_date: str | None):
    """Compute (start_dt, end_dt_exclusive) for filtering Booking.created_at."""
    from datetime import datetime, timedelta

    sd = _parse_iso_date(start_date) if start_date else None
    ed = _parse_iso_date(end_date) if end_date else None

    if sd and ed:
        # inclusive end date -> exclusive end datetime
        start_dt = datetime.combine(sd, datetime.min.time())
        end_dt = datetime.combine(ed + timedelta(days=1), datetime.min.time())
        return start_dt, end_dt

    # fallback to last N days
    try:
        d = int(days) if days is not None else 30
        d = max(1, min(d, 365))
    except Exception:
        d = 30
    end_dt = utcnow() + timedelta(seconds=1)
    start_dt = utcnow() - timedelta(days=d)
    return start_dt, end_dt

@app.get("/api/admin/analytics/bookings-over-time")
async def get_bookings_over_time(
    days: int = 30,
    start_date: str | None = None,
    end_date: str | None = None,
    ticket_class: Optional[str] = None,
    origin: Optional[str] = None,
    destination: Optional[str] = None,
    aircraft_registration: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get booking trends over time for charts"""
    if not _is_admin(current_user):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        # Get bookings grouped by date (with optional slicers)
        from sqlalchemy import func
        from datetime import timedelta

        date_key = func.strftime('%Y-%m-%d', Booking.created_at)
        q = db.query(
            date_key.label('date'),
            func.count(Booking.id).label('count'),
            func.coalesce(func.sum(Booking.price), 0).label('revenue')
        ).join(Flight, Flight.id == Booking.flight_id).filter(Booking.status == "confirmed")

        if ticket_class:
            q = q.filter(Booking.ticket_class == ticket_class)
        if origin:
            q = q.filter(Booking.departure_place == origin)
        if destination:
            q = q.filter(Booking.arrival_place == destination)
        if aircraft_registration:
            # match by flight aircraft registration
            q = q.join(Aircraft, Aircraft.id == Flight.aircraft_id).filter(Aircraft.registration == aircraft_registration)

        # date range filter (start/end) or fallback to last N days
        start_dt, end_dt = _date_range_from_params(days, start_date, end_date)
        q = q.filter(Booking.created_at >= start_dt, Booking.created_at < end_dt)

        bookings_by_date = q.group_by(date_key).order_by(date_key.asc()).all()
        
        return [{
            "date": str(row.date),
            "count": row.count,
            "revenue": round(row.revenue, 2)
        } for row in bookings_by_date]
    except Exception as e:
        print(f"Analytics error: {str(e)}")
        return []

@app.get("/api/admin/analytics/popular-routes")
async def get_popular_routes(
    days: int = 30,
    start_date: str | None = None,
    end_date: str | None = None,
    ticket_class: Optional[str] = None,
    origin: Optional[str] = None,
    destination: Optional[str] = None,
    aircraft_registration: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get most popular routes for charts"""
    if not _is_admin(current_user):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        routes_q = db.query(
            Booking.departure_place,
            Booking.arrival_place,
            func.count(Booking.id).label('booking_count'),
            func.coalesce(func.sum(Booking.price), 0).label('revenue')
        ).filter(Booking.status == "confirmed")

        # slicers
        if ticket_class:
            routes_q = routes_q.filter(Booking.ticket_class == ticket_class)
        if origin:
            routes_q = routes_q.filter(Booking.departure_place == origin)
        if destination:
            routes_q = routes_q.filter(Booking.arrival_place == destination)
        if aircraft_registration:
            routes_q = routes_q.join(Flight, Flight.id == Booking.flight_id).join(Aircraft, Aircraft.id == Flight.aircraft_id).filter(Aircraft.registration == aircraft_registration)

        start_dt, end_dt = _date_range_from_params(days, start_date, end_date)
        routes_q = routes_q.filter(Booking.created_at >= start_dt, Booking.created_at < end_dt)

        routes = routes_q.group_by(
            Booking.departure_place,
            Booking.arrival_place
        ).order_by(
            func.count(Booking.id).desc()
        ).limit(10).all()
        
        return [{
            "route": f"{row.departure_place} → {row.arrival_place}",
            "bookings": row.booking_count,
            "revenue": round(row.revenue, 2)
        } for row in routes]
    except Exception as e:
        print(f"Routes analytics error: {str(e)}")
        return []

@app.get("/api/admin/analytics/bookings-by-class")
async def get_bookings_by_class(
    days: int = 30,
    start_date: str | None = None,
    end_date: str | None = None,
    origin: Optional[str] = None,
    destination: Optional[str] = None,
    aircraft_registration: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get bookings distribution by ticket class"""
    if not _is_admin(current_user):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        q = db.query(
            Booking.ticket_class,
            func.count(Booking.id).label('booking_count'),
            func.coalesce(func.sum(Booking.price), 0).label('revenue')
        ).filter(Booking.status == "confirmed")

        if origin:
            q = q.filter(Booking.departure_place == origin)
        if destination:
            q = q.filter(Booking.arrival_place == destination)
        if aircraft_registration:
            q = q.join(Flight, Flight.id == Booking.flight_id).join(Aircraft, Aircraft.id == Flight.aircraft_id).filter(Aircraft.registration == aircraft_registration)

        start_dt, end_dt = _date_range_from_params(days, start_date, end_date)
        q = q.filter(Booking.created_at >= start_dt, Booking.created_at < end_dt)

        classes = q.group_by(Booking.ticket_class).all()
        
        return [{
            "class": row.ticket_class.capitalize(),
            "bookings": row.booking_count,
            "revenue": round(row.revenue, 2)
        } for row in classes]
    except Exception as e:
        print(f"Class analytics error: {str(e)}")
        return []

@app.get("/api/admin/analytics/user-growth")
async def get_user_growth(
    days: int = 30,
    start_date: str | None = None,
    end_date: str | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get user registration trends (cumulative).

    The admin UI chart expects a continuous day-by-day series.
    We therefore fill missing dates and also include a baseline count of users
    created before the selected range so the cumulative curve starts correctly.
    """
    if not _is_admin(current_user):
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        from sqlalchemy import func
        from datetime import timedelta

        start_dt, end_dt = _date_range_from_params(days, start_date, end_date)

        # Baseline: users created before the range (so cumulative is truly cumulative)
        baseline = db.query(func.count(User.id)).filter(
            User.is_admin == False,
            User.created_at < start_dt
        ).scalar() or 0

        # Daily registrations within the selected range
        date_key = func.strftime('%Y-%m-%d', User.created_at)
        rows = db.query(
            date_key.label('date'),
            func.count(User.id).label('count')
        ).filter(
            User.is_admin == False,
            User.created_at >= start_dt,
            User.created_at < end_dt
        ).group_by(date_key).all()

        per_day = {str(r.date): int(r.count or 0) for r in rows}

        # Fill every date in [start_dt, end_dt)
        result = []
        cumulative = int(baseline)
        cursor = start_dt.date()
        end_date_only = (end_dt - timedelta(seconds=1)).date()
        while cursor <= end_date_only:
            key = cursor.strftime('%Y-%m-%d')
            cumulative += per_day.get(key, 0)
            result.append({"date": key, "count": cumulative})
            cursor += timedelta(days=1)

        return result
    except Exception as e:
        print(f"User growth analytics error: {str(e)}")
        return []


@app.get("/api/admin/analytics/passengers-per-day")
async def get_passengers_per_day(
    days: int = 30,
    start_date: str | None = None,
    end_date: str | None = None,
    ticket_class: Optional[str] = None,
    origin: Optional[str] = None,
    destination: Optional[str] = None,
    aircraft_registration: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get passengers per day for charts."""
    if not _is_admin(current_user):
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        from sqlalchemy import func

        date_key = func.strftime('%Y-%m-%d', Booking.created_at)
        q = db.query(
            date_key.label('date'),
            func.coalesce(func.sum(Booking.num_passengers), 0).label('passengers')
        ).join(Flight, Flight.id == Booking.flight_id).filter(Booking.status == "confirmed")

        if ticket_class:
            q = q.filter(Booking.ticket_class == ticket_class)
        if origin:
            q = q.filter(Booking.departure_place == origin)
        if destination:
            q = q.filter(Booking.arrival_place == destination)
        if aircraft_registration:
            q = q.join(Aircraft, Aircraft.id == Flight.aircraft_id).filter(Aircraft.registration == aircraft_registration)

        start_dt, end_dt = _date_range_from_params(days, start_date, end_date)
        q = q.filter(Booking.created_at >= start_dt, Booking.created_at < end_dt)

        rows = q.group_by(date_key).order_by(date_key.asc()).all()
        return [{"date": str(r.date), "passengers": int(r.passengers or 0)} for r in rows]
    except Exception as e:
        print(f"Passengers analytics error: {str(e)}")
        return []



@app.get("/api/admin/analytics/bookings-by-status")
async def get_bookings_by_status(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Breakdown of bookings by status (confirmed/cancelled/etc.)."""
    if not _is_admin(current_user):
        raise HTTPException(status_code=403, detail="Admin access required")
    try:
        rows = db.query(
            Booking.status.label('status'),
            func.count(Booking.id).label('count'),
            func.sum(Booking.price).label('revenue')
        ).group_by(Booking.status).all()
        return [{
            "status": r.status,
            "count": int(r.count or 0),
            "revenue": float(r.revenue or 0)
        } for r in rows]
    except Exception as e:
        print(f"Analytics error: {str(e)}")
        return []


@app.get("/api/admin/analytics/top-customers")
async def get_top_customers(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Top users by total revenue."""
    if not _is_admin(current_user):
        raise HTTPException(status_code=403, detail="Admin access required")
    try:
        rows = db.query(
            User.username.label('username'),
            func.count(Booking.id).label('bookings'),
            func.sum(Booking.price).label('revenue')
        ).join(Booking, Booking.user_id == User.id).filter(
            Booking.status == "confirmed"
        ).group_by(User.username).order_by(func.sum(Booking.price).desc()).limit(10).all()

        return [{
            "username": r.username,
            "bookings": int(r.bookings or 0),
            "revenue": float(r.revenue or 0)
        } for r in rows]
    except Exception as e:
        print(f"Analytics error: {str(e)}")
        return []

# ============================================================================
# BOOKING ENDPOINTS
# ============================================================================

# Create Booking
@app.post("/api/bookings", response_model=BookingResponse)
async def create_booking(
    booking: BookingCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    try:
        # Check if flight exists
        flight = db.query(Flight).filter(Flight.id == booking.flight_id).first()
        if not flight:
            raise HTTPException(status_code=404, detail="Flight not found")
        
        # Calculate price based on ticket class and number of passengers using flight's base_price
        base_price = flight.base_price if hasattr(flight, 'base_price') and flight.base_price else 150.0
        class_multipliers = {
            "economy": 1.0,
            "business": 2.5,
            "premium": 1.5
        }
        multiplier = class_multipliers.get(booking.ticket_class, 1.0)
        total_price = base_price * multiplier * booking.num_passengers
        
        # Generate booking reference
        booking_ref = f"TU{secrets.token_hex(4).upper()}"
        
        # Create booking
        new_booking = Booking(
            user_id=current_user.id,
            flight_id=booking.flight_id,
            booking_reference=booking_ref,
            passenger_name=booking.passenger_name,
            passenger_email=booking.passenger_email,
            passenger_phone=booking.passenger_phone,
            departure_place=booking.departure_place,
            arrival_place=booking.arrival_place,
            num_passengers=booking.num_passengers,
            ticket_class=booking.ticket_class,
            price=total_price,
            status="confirmed"
        )
        db.add(new_booking)
        
        # Update passenger count
        flight.passengers_booked += booking.num_passengers
        
        db.commit()
        db.refresh(new_booking)
        
        # Add flight info
        new_booking.flight = flight
        return new_booking
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"Booking error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating booking: {str(e)}")

# Get My Bookings
@app.get("/api/bookings/my", response_model=BookingListResponse)
async def get_my_bookings(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    bookings = db.query(Booking).options(joinedload(Booking.flight)).filter(
        Booking.user_id == current_user.id,
        Booking.status != "cancelled"
    ).all()

    return BookingListResponse(total=len(bookings), items=bookings)

# Cancel Booking
@app.delete("/api/bookings/{booking_id}")
async def cancel_booking(
    booking_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    booking = db.query(Booking).filter(
        Booking.id == booking_id,
        Booking.user_id == current_user.id
    ).first()
    
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    
    booking.status = "cancelled"
    
    # Update passenger count
    flight = db.query(Flight).filter(Flight.id == booking.flight_id).first()
    if flight and flight.passengers_booked >= booking.num_passengers:
        flight.passengers_booked -= booking.num_passengers
    
    db.commit()
    return {"message": "Booking cancelled"}

# Delete User Account
@app.delete("/api/users/me")
async def delete_account(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    # Cancel all bookings
    bookings = db.query(Booking).filter(Booking.user_id == current_user.id).all()
    for booking in bookings:
        booking.status = "cancelled"
        flight = db.query(Flight).filter(Flight.id == booking.flight_id).first()
        if flight and flight.passengers_booked >= booking.num_passengers:
            flight.passengers_booked -= booking.num_passengers
    
    # Delete user
    db.delete(current_user)
    db.commit()
    
    return {"message": "Account deleted"}

# Admin: Get All Users
@app.get("/api/admin/users")
async def get_all_users(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    if not _is_admin(current_user):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    users = db.query(User).filter(User.is_admin == False).all()
    return {"total": len(users), "items": users}

# Admin: Get All Bookings
@app.get("/api/admin/bookings", response_model=BookingListResponse)
async def get_all_bookings(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    if not _is_admin(current_user):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    bookings = db.query(Booking).order_by(Booking.created_at.desc()).all()
    
    # Add flight info
    for booking in bookings:
        booking.flight = db.query(Flight).filter(Flight.id == booking.flight_id).first()
    
    return BookingListResponse(total=len(bookings), items=bookings)


# ==========================================================================
# EXTERNAL DATA (SERVER-SIDE PROXIES)
# ==========================================================================

@app.get("/api/live-flights")
async def get_live_flights():
    """Proxy OpenSky live flights via the backend to avoid browser CORS blocks."""
    try:
        r = requests.get("https://opensky-network.org/api/states/all", timeout=10)
        r.raise_for_status()
        data = r.json()
        states = data.get("states") or []

        # OpenSky state vector indices (relevant):
        # 1=callsign, 2=origin_country, 5=longitude, 6=latitude, 7=baro_altitude, 9=velocity
        tunisair = []
        for s in states:
            callsign = (s[1] or "").strip()
            if not callsign:
                continue

            # Tunisair commonly uses "TAR" ICAO callsign, and some feeds show "TS" prefixed.
            if not (callsign.startswith("TAR") or callsign.startswith("TS")):
                continue

            lon = s[5]
            lat = s[6]
            if lon is None or lat is None:
                continue

            tunisair.append({
                "callsign": callsign,
                "origin_country": s[2],
                "longitude": lon,
                "latitude": lat,
                "altitude": s[7],
                "velocity": s[9],
                "last_contact": s[4]
            })

        return {"total": len(tunisair), "items": tunisair}
    except Exception as e:
        return {"total": 0, "items": [], "error": str(e)}


# ==========================================================================
# STATIC FRONTEND
# ==========================================================================

# Serve index.html + assets from the same origin as the API to avoid
# file:// "null" origin issues (which commonly cause "Failed to fetch").
BASE_DIR = os.path.dirname(__file__)

# Mount AFTER /api routes are defined so it doesn't shadow them.
app.mount("/", StaticFiles(directory=BASE_DIR, html=True), name="static")

# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
async def startup_event():
    init_db()
    db = SessionLocal()
    try:
        seed_admin_user(db)
        load_aircraft_from_csv(db)
        load_crew_from_csv(db)
        load_flights_from_csv(db)
        load_crew_assignments_from_csv(db)
        load_maintenance_from_csv(db)
        # Demo data can be expensive to generate (bcrypt + lots of inserts).
        # Keep the server fast by default; enable explicitly when you want it.
        # مثال: TUNISAIR_SEED_DEMO=1 python main.py
        if os.getenv("TUNISAIR_SEED_DEMO", "0") == "1":
            seed_demo_data(db)
    finally:
        db.close()
    
    print("\n" + "="*70)
    print("✓ Tunisair Operations API Ready!")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔐 Admin Login: username='admin', password='admin123'")
    print("🖥️  Dashboard: Open dashboard.html in your browser")
    print("="*70 + "\n")

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import webbrowser
    import threading
    import time
    import os
    import sys
    import subprocess
    
    def open_browser():
        time.sleep(1.5)  # Wait for server to start
        # Open the frontend from the same origin as the API.
        # This avoids file:// "null" origin CORS issues.
        webbrowser.open('http://localhost:8000/')

    def ensure_requirements():
        """Install requirements on first run (simple and optional)."""
        marker = os.path.join(os.path.dirname(__file__), '.requirements_installed')
        if os.path.exists(marker):
            return
        req = os.path.join(os.path.dirname(__file__), 'requirements.txt')
        if not os.path.exists(req):
            return
        try:
            print("✓ Checking / installing requirements (first run)...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req])
            with open(marker, 'w', encoding='utf-8') as f:
                f.write(str(utcnow()))
        except Exception as e:
            print(f"⚠ Requirements install warning: {e}")
    
    # Install requirements on first run (guarded by a marker file).
    # If you prefer manual installs, you can skip this:
    #   TUNISAIR_SKIP_PIP=1 python main.py
    if os.getenv("TUNISAIR_SKIP_PIP", "0") != "1":
        ensure_requirements()

    # Opening a browser window is optional (useful on desktop, noisy in servers/CI).
    if os.getenv("TUNISAIR_OPEN_BROWSER", "0") == "1":
        threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)
