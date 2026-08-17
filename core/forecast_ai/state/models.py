from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class OperationalState:
    """Operational state carried between forecast days."""
    quality: float
    competency: float
    transfer: float
    release: float
    attendance: float
    operations_health: Optional[float] = None
    nps: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for predictor consumption."""
        result = {
            "quality": self.quality,
            "competency": self.competency,
            "transfer": self.transfer,
            "release": self.release,
            "attendance": self.attendance,
        }
        if self.operations_health is not None:
            result["operations_health"] = self.operations_health
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OperationalState":
        """Create OperationalState from a dictionary."""
        return cls(
            quality=float(data.get("quality", 0.0)),
            competency=float(data.get("competency", 0.0)),
            transfer=float(data.get("transfer", 0.0)),
            release=float(data.get("release", 0.0)),
            attendance=float(data.get("attendance", 0.0)),
            operations_health=data.get("operations_health"),
            nps=data.get("nps"),
            metadata=data.get("metadata")
        )
