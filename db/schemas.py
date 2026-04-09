from pydantic import BaseModel
from datetime import datetime, date
from enum import Enum


class AttendanceType(str, Enum):
    gate = "gate"
    lecture = "lecture"


class AttendanceCreate(BaseModel):
    organization_id: str
    student_id: str
    attendance_type: AttendanceType
    event_time: datetime
    date: date
    confidence_score: float
    camera_id: str


class AttendanceResponse(BaseModel):
    id: str
    organization_id: str
    student_id: str
    attendance_type: AttendanceType
    event_time: datetime
    date: date
    confidence_score: float
    camera_id: str
    created_at: datetime

    class Config:
        orm_mode = True