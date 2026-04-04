"""
DataAudit — biological plausibility checks for AgroEdge training datasets.

Runs after synthetic data generation (and again when real data arrives)
to verify the dataset satisfies agronomic constraints before any model
training begins.

If any check fails, the dataset is rejected with a clear error message.
The generator or data source must be fixed before training is allowed.

Checks performed:
  1. Class balance           — positive rate within 35–65%
  2. Tank safety             — no irrigation scheduled when tank is critically low
  3. Moisture correlation    — low moisture must correlate with irrigation=1
  4. VPD correlation         — high VPD must correlate (weakly) with irrigation=1
  5. Stage duration ordering — heading/flowering must have higher mean duration than ripening
  6. DAT–stage consistency   — days_after_transplanting must fall in the stage's BBCH range
  7. No impossible values    — all numeric fields within physical bounds
  8. Stage coverage          — every stage has ≥100 records and both label classes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from data_pipeline.schemas import (
    GROWTH_STAGE_DAT_THRESHOLDS,
    GROWTH_STAGE_ENCODING,
    GrowthStage,
)

# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class AuditResult:
    passed: bool
    check_name: str
    message: str
    value: Optional[float] = None
    threshold: Optional[float] = None


@dataclass
class DataAuditReport:
    results: list[AuditResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failed_checks(self) -> list[AuditResult]:
        return [r for r in self.results if not r.passed]

    def summary(self) -> str:
        total  = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        lines  = [
            f"Data Audit: {passed}/{total} checks passed",
            "─" * 56,
        ]
        for r in self.results:
            status = "✓ PASS" if r.passed else "✗ FAIL"
            lines.append(f"  [{status}]  {r.check_name}")
            lines.append(f"            {r.message}")
        lines.append("─" * 56)
        lines.append("OVERALL: PASSED" if self.passed else "OVERALL: FAILED — fix issues before training")
        return "\n".join(lines)


# ── Audit engine ──────────────────────────────────────────────────────────────

class DataAudit:
    """
    Biological plausibility audit for AgroEdge training datasets.

    Accepts a pandas DataFrame where each row is one TrainingRecord.
    All column names must match TrainingRecord field names exactly.
    """

    # Acceptable positive (irrigation_needed=1) rate range
    MIN_POSITIVE_RATE = 0.35
    MAX_POSITIVE_RATE = 0.65

    # Moisture–irrigation Pearson correlation must be negative (inverse relationship)
    MOISTURE_MAX_CORRELATION = -0.25

    # VPD–irrigation Pearson correlation must be positive (water stress → irrigation)
    VPD_MIN_CORRELATION = 0.05

    # Tank safety: physically impossible to irrigate without water
    MIN_TANK_LEVEL_FOR_IRRIGATION = 10.0

    # Each stage must have at least this many records
    MIN_RECORDS_PER_STAGE = 100

    # Field / ThingSpeak pulls: smaller datasets, omit synthetic-only statistics
    MIN_ROWS_FIELD_AUDIT = 5

    def run(self, df: pd.DataFrame, *, profile: str = "synthetic") -> DataAuditReport:
        """
        Run checks on the dataset DataFrame.

        Args:
            profile: ``synthetic`` — full agronomic distribution checks (default).
                ``field`` — strict safety + bounds only, for real cloud pulls with few rows.

        Returns:
            DataAuditReport. Training should only proceed if report.passed is True.
        """
        report = DataAuditReport()
        if profile == "field":
            report.results.append(self._check_field_rowcount(df))
            report.results.append(self._check_tank_safety(df))
            report.results.append(self._check_no_impossible_values(df))
            return report
        if profile != "synthetic":
            raise ValueError(f"Unknown audit profile: {profile!r}")

        report.results.append(self._check_class_balance(df))
        report.results.append(self._check_tank_safety(df))
        report.results.append(self._check_moisture_correlation(df))
        report.results.append(self._check_vpd_correlation(df))
        report.results.append(self._check_stage_duration_ordering(df))
        report.results.append(self._check_dat_stage_consistency(df))
        report.results.append(self._check_no_impossible_values(df))
        report.results.append(self._check_stage_coverage(df))
        return report

    def _check_field_rowcount(self, df: pd.DataFrame) -> AuditResult:
        """Ensure enough rows to bother training (field datasets can be tiny)."""
        n = len(df)
        passed = n >= self.MIN_ROWS_FIELD_AUDIT
        return AuditResult(
            passed=passed,
            check_name="Minimum Row Count (field)",
            message=(
                f"{n} rows "
                f"(minimum {self.MIN_ROWS_FIELD_AUDIT} for retraining experiments)"
            ),
            value=float(n),
            threshold=float(self.MIN_ROWS_FIELD_AUDIT),
        )

    # ── Individual checks ─────────────────────────────────────────────────────

    def _check_class_balance(self, df: pd.DataFrame) -> AuditResult:
        """Positive rate must fall between 35% and 65%."""
        positive_rate = float(df["irrigation_needed"].mean())
        passed = self.MIN_POSITIVE_RATE <= positive_rate <= self.MAX_POSITIVE_RATE
        return AuditResult(
            passed=passed,
            check_name="Class Balance",
            message=(
                f"Positive rate = {positive_rate:.3f} "
                f"(acceptable range: {self.MIN_POSITIVE_RATE}–{self.MAX_POSITIVE_RATE})"
            ),
            value=positive_rate,
        )

    def _check_tank_safety(self, df: pd.DataFrame) -> AuditResult:
        """
        No record should have irrigation_needed=1 when tank level is critically low.
        This is a physical impossibility — zero tolerance.
        """
        violations = df[
            (df["water_tank_level_percent"] < self.MIN_TANK_LEVEL_FOR_IRRIGATION)
            & (df["irrigation_needed"] == 1)
        ]
        passed = len(violations) == 0
        return AuditResult(
            passed=passed,
            check_name="Tank Safety Constraint",
            message=(
                f"{len(violations)} records with tank < {self.MIN_TANK_LEVEL_FOR_IRRIGATION}% "
                f"and irrigation_needed=1 (must be 0)"
            ),
            value=float(len(violations)),
            threshold=0.0,
        )

    def _check_moisture_correlation(self, df: pd.DataFrame) -> AuditResult:
        """
        Pearson correlation between soil moisture and irrigation_needed must be
        meaningfully negative. Low moisture → irrigate (inverse relationship).
        """
        corr = float(df["soil_moisture_percent"].corr(df["irrigation_needed"]))
        passed = corr < self.MOISTURE_MAX_CORRELATION
        return AuditResult(
            passed=passed,
            check_name="Moisture–Irrigation Correlation",
            message=(
                f"Pearson r = {corr:.4f} "
                f"(must be < {self.MOISTURE_MAX_CORRELATION} to confirm inverse relationship)"
            ),
            value=corr,
            threshold=self.MOISTURE_MAX_CORRELATION,
        )

    def _check_vpd_correlation(self, df: pd.DataFrame) -> AuditResult:
        """
        VPD and irrigation_needed should have a positive (though weak) correlation.
        High atmospheric dryness increases plant water stress and irrigation urgency.
        """
        corr = float(df["vpd_kpa"].corr(df["irrigation_needed"]))
        passed = corr > self.VPD_MIN_CORRELATION
        return AuditResult(
            passed=passed,
            check_name="VPD–Irrigation Correlation",
            message=(
                f"Pearson r = {corr:.4f} "
                f"(must be > {self.VPD_MIN_CORRELATION})"
            ),
            value=corr,
            threshold=self.VPD_MIN_CORRELATION,
        )

    def _check_stage_duration_ordering(self, df: pd.DataFrame) -> AuditResult:
        """
        Among irrigated records, mean duration at heading/flowering must exceed
        mean duration at ripening/maturity.
        This validates FAO water requirement ordering: heading > ripening.
        """
        heading_enc  = GROWTH_STAGE_ENCODING[GrowthStage.HEADING_FLOWERING]
        ripening_enc = GROWTH_STAGE_ENCODING[GrowthStage.RIPENING_MATURITY]
        irrigated    = df[df["irrigation_needed"] == 1]

        heading_dur  = irrigated[
            irrigated["growth_stage_encoded"] == heading_enc
        ]["irrigation_duration_minutes"].mean()

        ripening_dur = irrigated[
            irrigated["growth_stage_encoded"] == ripening_enc
        ]["irrigation_duration_minutes"].mean()

        if pd.isna(heading_dur) or pd.isna(ripening_dur):
            return AuditResult(
                passed=False,
                check_name="Stage Duration Ordering",
                message="Insufficient irrigated records to compare stage means.",
            )

        passed = heading_dur > ripening_dur
        return AuditResult(
            passed=passed,
            check_name="Stage Duration Ordering (FAO)",
            message=(
                f"heading_flowering mean = {heading_dur:.1f} min, "
                f"ripening_maturity mean = {ripening_dur:.1f} min "
                f"(heading must exceed ripening)"
            ),
            value=float(heading_dur - ripening_dur),
        )

    def _check_dat_stage_consistency(self, df: pd.DataFrame) -> AuditResult:
        """
        Every record's days_after_transplanting must fall within the BBCH DAT
        range for its encoded growth stage.
        """
        violations = 0
        stage_issues: list[str] = []

        for start, end, stage in GROWTH_STAGE_DAT_THRESHOLDS:
            enc      = GROWTH_STAGE_ENCODING[stage]
            stage_df = df[df["growth_stage_encoded"] == enc]
            bad      = stage_df[
                (stage_df["days_after_transplanting"] < start) |
                (stage_df["days_after_transplanting"] >= end)
            ]
            if len(bad) > 0:
                violations += len(bad)
                stage_issues.append(f"{stage.value}: {len(bad)} violations")

        passed  = violations == 0
        message = (
            "All DAT values consistent with their growth stage"
            if passed
            else f"{violations} violations — {'; '.join(stage_issues)}"
        )
        return AuditResult(
            passed=passed,
            check_name="DAT–Stage Consistency",
            message=message,
            value=float(violations),
            threshold=0.0,
        )

    def _check_no_impossible_values(self, df: pd.DataFrame) -> AuditResult:
        """
        Check that all numeric fields contain physically possible values.
        Catches systematic generator bugs early.
        """
        checks: dict[str, pd.Series] = {
            "soil_moisture_percent [0,100]":
                (df["soil_moisture_percent"] < 0) | (df["soil_moisture_percent"] > 100),
            "air_humidity_percent [0,100]":
                (df["air_humidity_percent"] < 0) | (df["air_humidity_percent"] > 100),
            "vpd_kpa [>=0]":
                df["vpd_kpa"] < 0,
            "growth_stage_encoded [0,7]":
                (df["growth_stage_encoded"] < 0) | (df["growth_stage_encoded"] > 7),
            "irrigation_duration_minutes [>=0]":
                df["irrigation_duration_minutes"] < 0,
            "days_since_last_irrigation [>=0]":
                df["days_since_last_irrigation"] < 0,
            "hour_of_day [0,23]":
                (df["hour_of_day"] < 0) | (df["hour_of_day"] > 23),
            "month [1,12]":
                (df["month"] < 1) | (df["month"] > 12),
        }

        total_violations = 0
        field_issues: list[str] = []
        for field_desc, mask in checks.items():
            count = int(mask.sum())
            if count > 0:
                total_violations += count
                field_issues.append(f"{field_desc}: {count}")

        passed  = total_violations == 0
        message = (
            "All fields within physical bounds"
            if passed
            else f"{total_violations} violations — {'; '.join(field_issues)}"
        )
        return AuditResult(
            passed=passed,
            check_name="No Impossible Values",
            message=message,
            value=float(total_violations),
            threshold=0.0,
        )

    def _check_stage_coverage(self, df: pd.DataFrame) -> AuditResult:
        """
        Every growth stage must have at least MIN_RECORDS_PER_STAGE records
        and must contain both label classes (0 and 1).
        Catches sampling imbalances that would leave the model blind to a stage.
        """
        issues: list[str] = []

        for stage, enc in GROWTH_STAGE_ENCODING.items():
            stage_df = df[df["growth_stage_encoded"] == enc]

            if len(stage_df) < self.MIN_RECORDS_PER_STAGE:
                issues.append(
                    f"{stage.value}: only {len(stage_df)} records "
                    f"(min {self.MIN_RECORDS_PER_STAGE})"
                )
            elif stage_df["irrigation_needed"].nunique() < 2:
                issues.append(
                    f"{stage.value}: only one class present "
                    f"(both 0 and 1 required)"
                )

        passed  = len(issues) == 0
        message = (
            f"All {len(GROWTH_STAGE_ENCODING)} stages have adequate coverage"
            if passed
            else f"Coverage gaps: {'; '.join(issues)}"
        )
        return AuditResult(passed=passed, check_name="Per-Stage Coverage", message=message)
