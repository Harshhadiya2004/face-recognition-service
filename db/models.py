import uuid
from .database import Base
from sqlalchemy import Column, String, Float, Date, TIMESTAMP, Enum
from sqlalchemy.dialects.postgresql import UUID
import enum


class AttendanceType(str, enum.Enum):
    gate = "gate"
    lecture = "lecture"

class StudentAttendance(Base):
    __tablename__ = "attendance_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(String)
    student_id = Column(String)
    attendance_type = Column(Enum(AttendanceType, name="attendancetype"))
    event_time = Column(TIMESTAMP)
    date = Column(Date)
    confidence_score = Column(Float)
    camera_id = Column(String)
    created_at = Column(TIMESTAMP)