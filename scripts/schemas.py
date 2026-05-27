"""
Pydantic models for JSON schema validation.
All data structures used in the pipeline are defined here.
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"


import hashlib
import re
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# --- Enums ---

class VideoStatus(str, Enum):
    PENDING = "pending"
    PROCESSED = "processed"
    ERRORED = "errored"


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class FlagReason(str, Enum):
    LOW_CONFIDENCE = "possible_locale_mention_low_confidence"
    MISSING_NAME = "locale_name_not_identified"
    MISSING_ADDRESS = "address_not_identified"
    MISSING_RATING = "rating_not_identified"
    GEOCODING_FAILED = "geocoding_failed"
    OSM_NOT_FOUND = "osm_not_found"
    AMBIGUOUS_LOCALE = "ambiguous_locale_reference"
    RATING_TITLE_MISMATCH = "rating_title_transcript_mismatch"


# --- Helper functions ---

def generate_locale_id(name: str, lat: float, lon: float) -> str:
    """
    Generate a deterministic locale ID from name + rounded coordinates.
    Uses SHA256 hash truncated to 12 hex chars for stability.
    """
    normalized_name = re.sub(r'\s+', '_', name.strip().lower())
    rounded_lat = round(lat, 4)
    rounded_lon = round(lon, 4)
    raw = f"{normalized_name}_{rounded_lat}_{rounded_lon}"
    hash_hex = hashlib.sha256(raw.encode('utf-8')).hexdigest()[:12]
    return f"locale_{hash_hex}"


def generate_visit_id(video_id: str, timestamp_seconds: int) -> str:
    """
    Generate a deterministic visit ID from video_id + timestamp in seconds.
    """
    return f"visit_{video_id}_{timestamp_seconds}"


def timestamp_to_seconds(ts: str) -> int:
    """Convert MM:SS or HH:MM:SS to seconds."""
    parts = ts.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    raise ValueError(f"Invalid timestamp format: {ts}")


# --- Models ---

class Channel(BaseModel):
    channel_id: str = Field(..., description="Deterministic ID derived from channel URL")
    name: str = Field(..., description="Channel display name")
    url: str = Field(..., description="YouTube channel URL")
    description: str = Field(default="", description="Channel description")
    rubriche: list[str] = Field(default_factory=list, description="Show/rubrica names inferred from video titles")


class Video(BaseModel):
    video_id: str = Field(..., description="YouTube video ID")
    channel_id: str = Field(..., description="Reference to channels.json")
    title: str = Field(..., description="Video title (original language)")
    url: str = Field(..., description="YouTube video URL")
    publish_date: str = Field(..., description="Video publish date (YYYY-MM-DD)")
    processed_date: str = Field(default="", description="Date when pipeline processed this video (YYYY-MM-DD, empty if pending)")
    status: VideoStatus = Field(..., description="Processing status")

    @field_validator("publish_date")
    @classmethod
    def validate_publish_date(cls, v: str) -> str:
        if v == "":
            return v  # empty is valid, resolved later during processing
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Date must be YYYY-MM-DD or empty, got: {v}")
        return v

    @field_validator("processed_date")
    @classmethod
    def validate_processed_date(cls, v: str) -> str:
        if v == "":
            return v  # empty is valid for pending videos
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Date must be YYYY-MM-DD or empty, got: {v}")
        return v


class Locale(BaseModel):
    locale_id: str = Field(..., description="Deterministic hash ID from name + coordinates")
    name: str = Field(..., description="Primary locale name")
    aliases: list[str] = Field(default_factory=list, description="Alternative names for deduplication")
    address: str = Field(default="", description="Street address")
    city: str = Field(default="", description="City name")
    lat: float = Field(..., ge=-90, le=90, description="Latitude (4 decimal places)")
    lon: float = Field(..., ge=-180, le=180, description="Longitude (4 decimal places)")
    category: list[str] = Field(default_factory=list, description="Business categories (e.g., forno, ristorante)")
    google_maps_url: str = Field(default="", description="Google Maps search URL for this locale")


class Visit(BaseModel):
    visit_id: str = Field(..., description="Deterministic ID: visit_{video_id}_{timestamp_seconds}")
    locale_id: str = Field(..., description="Reference to locales.json")
    video_id: str = Field(..., description="Reference to videos.json")
    channel_id: str = Field(..., description="Reference to channels.json")
    timestamp_start: str = Field(..., description="Start timestamp (MM:SS or HH:MM:SS)")
    timestamp_end: str = Field(..., description="End timestamp (MM:SS or HH:MM:SS)")
    youtube_url: str = Field(..., description="Direct YouTube URL with timestamp parameter")
    rating: Optional[str] = Field(default=None, description="Overall venue rating as stated (e.g., '8', '8--', '6++', '10')")
    sentiment: Sentiment = Field(..., description="Sentiment: positive, neutral, negative")
    rubrica: str = Field(default="", description="Show/rubrica name")
    notes: str = Field(default="", description="What was eaten, partial item ratings, observations")
    llm_confidence: float = Field(..., ge=0, le=1, description="LLM extraction confidence (0-1)")
    extraction_date: str = Field(..., description="Date of extraction (YYYY-MM-DD)")
    date: str = Field(..., description="Approximate visit date (video publish date, YYYY-MM-DD)")

    @field_validator("timestamp_start", "timestamp_end")
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        pattern = r"^(\d{1,2}:)?\d{1,2}:\d{2}$"
        if not re.match(pattern, v):
            raise ValueError(f"Timestamp must be MM:SS or HH:MM:SS, got: {v}")
        return v

    @field_validator("extraction_date", "date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Date must be YYYY-MM-DD format, got: {v}")
        return v

    @field_validator("rating", mode="before")
    @classmethod
    def coerce_rating_to_str(cls, v: object) -> Optional[str]:
        if v is None or v == "":
            return None
        if isinstance(v, (int, float)):
            return str(v)
        return str(v).strip() if isinstance(v, str) else str(v)

    @field_validator("rating")
    @classmethod
    def validate_rating_range(cls, v: Optional[str]) -> Optional[str]:
        """Allow blogger-style ratings (e.g. '8--', '6++'); reject absurd numeric core > 10."""
        if v is None:
            return None
        m = re.match(r"^(\d+(?:\.\d+)?)", v.replace(",", ".").strip())
        if not m:
            return v
        num = float(m.group(1))
        if num > 10 or num < 0:
            raise ValueError(f"Rating numeric part must be 0–10, got: {v}")
        return v


class FlaggedSegment(BaseModel):
    video_id: str = Field(..., description="Reference to videos.json")
    channel_id: str = Field(..., description="Reference to channels.json")
    timestamp_start: str = Field(..., description="Start timestamp")
    timestamp_end: str = Field(..., description="End timestamp")
    youtube_url: str = Field(..., description="Direct YouTube URL with timestamp")
    reason: FlagReason = Field(..., description="Reason for flagging")
    extracted_text: str = Field(default="", description="Transcription segment text")
    llm_confidence: float = Field(..., ge=0, le=1, description="LLM confidence score")
    reviewed_by_human: bool = Field(default=False, description="Whether a human has reviewed this")
    reviewed_date: Optional[str] = Field(default=None, description="Date of human review (YYYY-MM-DD)")
    locale_name: Optional[str] = Field(default=None, description="Locale name if identified during review")
    rating: Optional[str] = Field(default=None, description="Rating if identified during review (e.g., '8', '8--', '6++')")
    city: Optional[str] = Field(default=None, description="City if identified during review")


class CorrectionOverrides(BaseModel):
    name: Optional[str] = Field(default=None)
    city: Optional[str] = Field(default=None)
    rating: Optional[float] = Field(default=None)
    sentiment: Optional[Sentiment] = Field(default=None)


class Correction(BaseModel):
    locale_id: str = Field(..., description="Reference to locales.json locale_id")
    type: str = Field(..., description="Correction type: 'hide' or 'edit'")
    reason: Optional[str] = Field(default=None, description="Human-readable reason")
    overrides: Optional[CorrectionOverrides] = Field(default=None, description="Field overrides for 'edit' type")

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        if v not in ("hide", "edit"):
            raise ValueError(f"type must be 'hide' or 'edit', got: {v}")
        return v


class SkippedVideo(BaseModel):
    video_id: str = Field(..., description="YouTube video ID")
    channel_id: str = Field(default="", description="Reference to channels.json")
    title: str = Field(..., description="Original video title")
    url: str = Field(..., description="YouTube video URL")
    reason: str = Field(..., description="Reason for skipping (e.g. 'Recipe keyword in title', 'Non-food video')")
    skipped_date: str = Field(..., description="Date when video was skipped (YYYY-MM-DD)")

    @field_validator("skipped_date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Date must be YYYY-MM-DD format, got: {v}")
        return v


class ProcessedVideo(BaseModel):
    video_id: str = Field(..., description="YouTube video ID")
    channel_id: str = Field(..., description="Channel that owns the video")
    processed_date: str = Field(..., description="Date of processing (YYYY-MM-DD)")
    status: VideoStatus = Field(..., description="Processing result status")
    visits_extracted: int = Field(default=0, ge=0, description="Number of visits extracted from this video")
    flagged_segments: int = Field(default=0, ge=0, description="Number of flagged segments from this video")

    @field_validator("processed_date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Date must be YYYY-MM-DD format, got: {v}")
        return v


# --- Validation helpers ---

def validate_channels(data: list[dict]) -> list[Channel]:
    return [Channel(**item) for item in data]

def validate_videos(data: list[dict]) -> list[Video]:
    return [Video(**item) for item in data]

def validate_locales(data: list[dict]) -> list[Locale]:
    return [Locale(**item) for item in data]

def validate_visits(data: list[dict]) -> list[Visit]:
    return [Visit(**item) for item in data]

def validate_flagged_segments(data: list[dict]) -> list[FlaggedSegment]:
    return [FlaggedSegment(**item) for item in data]

def validate_processed_videos(data: list[dict]) -> list[ProcessedVideo]:
    return [ProcessedVideo(**item) for item in data]


def validate_skipped_videos(data: list[dict]) -> list[SkippedVideo]:
    return [SkippedVideo(**item) for item in data]


def validate_corrections(data: list[dict]) -> list[Correction]:
    return [Correction(**item) for item in data]
