from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

class ReportType(Enum):
    EXECUTIVE = "executive"
    OPERATIONAL = "operational"
    TECHNICAL = "technical"
    MANAGEMENT = "management"
    AUDIT = "audit"
    DASHBOARD = "dashboard"
    JSON = "json"
    MARKDOWN = "markdown"

@dataclass
class ReportSection:
    title: str
    content: str
    order: int
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExecutiveSummary:
    overview: str
    key_findings: List[str]
    top_recommendations: List[str]
    top_risks: List[str]
    confidence_summary: str
    operational_outlook: str
    conclusion: str

@dataclass
class ReportAppendix:
    metadata: Dict[str, Any]
    warnings: List[str]
    errors: List[str]
    components: List[str]
    generation_time: str

@dataclass
class ReportMetadata:
    generated_at: str
    report_type: str
    version: str = "1.0"
    components: List[str] = field(default_factory=list)
    author: str = "ForecastAI"
    forecast_horizon: Optional[int] = None
    execution_duration: Optional[float] = None
    simulation_id: Optional[str] = None

@dataclass
class ReportResult:
    success: bool
    title: str
    executive_summary: Optional[ExecutiveSummary] = None
    sections: List[ReportSection] = field(default_factory=list)
    appendix: Optional[ReportAppendix] = None
    metadata: Optional[ReportMetadata] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
