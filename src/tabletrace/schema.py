"""Pydantic models for extracted values and quarantined cells."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PropertyKind(StrEnum):
    vickers_hardness = "vickers_hardness"
    yield_strength = "yield_strength"
    ultimate_strength = "ultimate_strength"  # UTS in tension, UCS in compression
    elongation = "elongation"  # elongation, fracture strain, strain to failure


CANONICAL_UNIT: dict[PropertyKind, str] = {
    PropertyKind.vickers_hardness: "HV",
    PropertyKind.yield_strength: "MPa",
    PropertyKind.ultimate_strength: "MPa",
    PropertyKind.elongation: "%",
}

# Plausible bounds in canonical units. Strengths above 7 GPa or hardness above
# 3000 HV do not occur in metallic alloys and almost always mean a parsing slip
# (a dropped decimal point, a merged cell, the wrong column).
PHYSICAL_RANGE: dict[PropertyKind, tuple[float, float]] = {
    PropertyKind.vickers_hardness: (5.0, 3000.0),
    PropertyKind.yield_strength: (1.0, 7000.0),
    PropertyKind.ultimate_strength: (1.0, 7000.0),
    PropertyKind.elongation: (0.0, 100.0),
}
# Hot ductility routinely exceeds 100% (NASA TM X-3216 reports 160% at 650 C),
# so the elongation ceiling is relaxed for elevated-temperature tests.
ELEVATED_TEMPERATURE_C = 200.0
HOT_ELONGATION_MAX = 2000.0

Source = Literal["text_layer", "ocr"]
Qualifier = Literal["=", "<", ">", "~", "range"]


class ExtractedValue(BaseModel):
    """One property value, traceable to the exact table cell it came from."""

    model_config = ConfigDict(use_enum_values=False, frozen=False)

    paper_id: str
    page: int = Field(ge=1)
    table_index: int  # printed table number, or -1 when no caption could be linked
    material: str  # row label as printed, e.g. "CoCrFeNi-Al0.3"
    property: PropertyKind
    value: float
    uncertainty: float | None = None
    unit: str  # canonical unit for the property
    raw_text: str  # exactly what the cell said
    source: Source
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    # provenance and context beyond the core schema
    qualifier: Qualifier = "="
    raw_unit: str | None = None  # unit as printed, before conversion
    header: str = ""  # header path the property was mapped from
    test_temperature_c: float | None = None
    loading: Literal["tension", "compression"] | None = None
    condition: str | None = None
    row: int | None = None
    col: int | None = None
    bbox: tuple[float, float, float, float] | None = None  # cell box in PDF points
    route: str = ""  # which extraction route produced the value
    features: dict[str, float] = Field(default_factory=dict)  # inputs to the confidence score

    @field_validator("material")
    @classmethod
    def _material_present(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("missing material: the row has no label")
        return v

    @field_validator("uncertainty")
    @classmethod
    def _uncertainty_non_negative(cls, v: float | None) -> float | None:
        if v is not None and v < 0:
            raise ValueError("bad uncertainty: must be non-negative")
        return v

    @model_validator(mode="after")
    def _unit_and_range(self) -> ExtractedValue:
        expected = CANONICAL_UNIT[self.property]
        if self.unit != expected:
            raise ValueError(
                f"bad unit: {self.property.value} must be stored in {expected}, got {self.unit!r}"
            )
        lo, hi = PHYSICAL_RANGE[self.property]
        if (
            self.property is PropertyKind.elongation
            and self.test_temperature_c is not None
            and self.test_temperature_c >= ELEVATED_TEMPERATURE_C
        ):
            hi = HOT_ELONGATION_MAX
        if not (lo <= self.value <= hi):
            raise ValueError(
                f"value outside physical range: {self.value:g} {self.unit} not in [{lo:g}, {hi:g}] "
                f"for {self.property.value}"
            )
        return self


class QuarantinedCell(BaseModel):
    """A cell that looked like a property value but failed parsing or validation."""

    paper_id: str
    page: int
    table_index: int
    material: str
    property: str
    raw_text: str
    raw_unit: str | None = None
    header: str = ""
    source: str
    reason: str  # bad unit | non-numeric value | value outside physical range | missing material
    detail: str = ""
    row: int | None = None
    col: int | None = None


def quarantine_reason(message: str) -> str:
    """Map a validation message to one of the four quarantine categories."""
    for reason in ("bad unit", "non-numeric value", "value outside physical range", "missing material"):
        if reason in message:
            return reason
    return "other"
