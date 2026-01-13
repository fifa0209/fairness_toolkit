"""
Report Generator for Fairness Monitoring

Creates comprehensive reports for fairness monitoring results,
including executive summaries, technical details, and visualizations.

Author: FairML Consulting
Date: January 2026
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from shared.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ReportSection:
    """A section of the report."""
    
    title: str
    content: str
    level: int = 2  # Heading level (1-6)
    include_in_toc: bool = True


class FairnessMonitoringReport:
    """
    Generator for fairness monitoring reports.
    
    Creates comprehensive Markdown reports with:
    - Executive summary
    - Metric trends
    - Alert history
    - Recommendations
    """
    
    def __init__(self, output_dir: Path = Path("reports")):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory for saving reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.sections: List[ReportSection] = []
        
        logger.info(f"Initialized FairnessMonitoringReport (dir={output_dir})")
    
    def generate_monitoring_report(
        self,
        summary_stats: Dict[str, Dict[str, float]],
        alerts: List[Dict[str, Any]],
        time_series: Optional[pd.DataFrame] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate comprehensive monitoring report.
        
        Args:
            summary_stats: Summary statistics for each metric
            alerts: List of alerts triggered
            time_series: Time series data for metrics
            metadata: Additional metadata
        
        Returns:
            Path to generated report file
        """
        logger.info("Generating monitoring report")
        
        # Clear previous sections
        self.sections = []
        
        # Add sections
        self._add_header(metadata)
        self._add_executive_summary(summary_stats, alerts)
        self._add_metric_overview(summary_stats)
        self._add_alert_summary(alerts)
        
        if time_series is not None:
            self._add_trend_analysis(time_series)
        
        self._add_recommendations(summary_stats, alerts)
        self._add_technical_details(summary_stats, metadata)
        
        # Generate report
        report_content = self._compile_report()
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fairness_monitoring_report_{timestamp}.md"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Report saved to {filepath}")
        
        return str(filepath)
    
    def _add_header(self, metadata: Optional[Dict[str, Any]]):
        """Add report header."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        content = f"""# Fairness Monitoring Report

**Generated:** {timestamp}
"""
        
        if metadata:
            content += "\n**Report Details:**\n"
            for key, value in metadata.items():
                content += f"- **{key}:** {value}\n"
        
        content += "\n---\n"
        
        self.sections.append(ReportSection(
            title="Header",
            content=content,
            level=1,
            include_in_toc=False
        ))
    
    def _add_executive_summary(
        self,
        summary_stats: Dict[str, Dict[str, float]],
        alerts: List[Dict[str, Any]]
    ):
        """Add executive summary section."""
        # Count critical issues
        n_metrics = len(summary_stats)
        n_alerts = len(alerts)
        
        critical_alerts = [
            a for a in alerts
            if a.get('severity', '').upper() in ['CRITICAL', 'HIGH']
        ]
        n_critical = len(critical_alerts)
        
        # Overall status
        if n_critical > 0:
            status = "🔴 **CRITICAL**"
            status_desc = "Immediate attention required"
        elif n_alerts > 0:
            status = "🟡 **WARNING**"
            status_desc = "Issues detected"
        else:
            status = "🟢 **HEALTHY**"
            status_desc = "All metrics within acceptable ranges"
        
        content = f"""## Executive Summary

### Overall Status: {status}

{status_desc}

### Key Findings

- **Monitored Metrics:** {n_metrics}
- **Total Alerts:** {n_alerts}
- **Critical/High Severity:** {n_critical}

"""
        
        if critical_alerts:
            content += "### Critical Issues\n\n"
            for alert in critical_alerts[:3]:  # Show top 3
                metric = alert.get('metric_name', 'Unknown')
                value = alert.get('current_value', 0)
                threshold = alert.get('threshold', 0)
                content += f"- **{metric}:** {value:.3f} (threshold: {threshold:.3f})\n"
            
            if len(critical_alerts) > 3:
                content += f"\n*...and {len(critical_alerts) - 3} more*\n"
        
        content += "\n---\n"
        
        self.sections.append(ReportSection(
            title="Executive Summary",
            content=content,
            level=2
        ))
    
    def _add_metric_overview(self, summary_stats: Dict[str, Dict[str, float]]):
        """Add metric overview section."""
        content = """## Metric Overview

Current status of all monitored fairness metrics.

"""
        
        # Create table
        content += "| Metric | Current | Mean | Std Dev | Min | Max | Status |\n"
        content += "|--------|---------|------|---------|-----|-----|--------|\n"
        
        for metric_name, stats in summary_stats.items():
            current = stats.get('current', 0)
            mean = stats.get('mean', 0)
            std = stats.get('std', 0)
            min_val = stats.get('min', 0)
            max_val = stats.get('max', 0)
            
            # Determine status
            threshold = stats.get('threshold', 0.10)
            if current > threshold * 1.5:
                status = "🔴 Critical"
            elif current > threshold:
                status = "🟡 Warning"
            else:
                status = "🟢 Good"
            
            content += (
                f"| {metric_name} | {current:.3f} | {mean:.3f} | {std:.3f} | "
                f"{min_val:.3f} | {max_val:.3f} | {status} |\n"
            )
        
        content += "\n---\n"
        
        self.sections.append(ReportSection(
            title="Metric Overview",
            content=content,
            level=2
        ))
    
    def _add_alert_summary(self, alerts: List[Dict[str, Any]]):
        """Add alert summary section."""
        if not alerts:
            content = """## Alert Summary

No alerts triggered during this monitoring period. ✅

---
"""
            self.sections.append(ReportSection(
                title="Alert Summary",
                content=content,
                level=2
            ))
            return
        
        content = f"""## Alert Summary

**Total Alerts:** {len(alerts)}

"""
        
        # Group by severity
        severity_counts = {}
        for alert in alerts:
            sev = alert.get('severity', 'UNKNOWN').upper()
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        content += "### Alerts by Severity\n\n"
        for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO']:
            count = severity_counts.get(severity, 0)
            if count > 0:
                emoji = {
                    'CRITICAL': '🔴',
                    'HIGH': '🟠',
                    'MEDIUM': '🟡',
                    'LOW': '🔵',
                    'INFO': 'ℹ️'
                }
                content += f"- {emoji.get(severity, '')} **{severity}:** {count}\n"
        
        # Recent alerts
        content += "\n### Recent Alerts\n\n"
        
        # Sort by priority/timestamp
        sorted_alerts = sorted(
            alerts,
            key=lambda a: (
                -self._severity_to_priority(a.get('severity', 'LOW')),
                a.get('timestamp', '')
            ),
            reverse=True
        )
        
        for i, alert in enumerate(sorted_alerts[:5], 1):
            severity = alert.get('severity', 'UNKNOWN')
            metric = alert.get('metric_name', 'Unknown')
            message = alert.get('message', 'No message')
            timestamp = alert.get('timestamp', 'Unknown time')
            
            content += f"#### {i}. [{severity}] {metric}\n\n"
            content += f"**Time:** {timestamp}\n\n"
            content += f"**Message:** {message}\n\n"
            
            # Recommendations
            actions = alert.get('recommended_actions', [])
            if actions:
                content += "**Recommended Actions:**\n"
                for action in actions[:3]:
                    content += f"- {action}\n"
                content += "\n"
        
        if len(alerts) > 5:
            content += f"\n*Showing 5 of {len(alerts)} alerts. See detailed logs for complete history.*\n"
        
        content += "\n---\n"
        
        self.sections.append(ReportSection(
            title="Alert Summary",
            content=content,
            level=2
        ))
    
    def _add_trend_analysis(self, time_series: pd.DataFrame):
        """Add trend analysis section."""
        content = """## Trend Analysis

Analysis of fairness metrics over time.

"""
        
        if time_series.empty:
            content += "*No time series data available.*\n\n"
        else:
            # Analyze each metric
            metrics = [col for col in time_series.columns if col != 'timestamp']
            
            for metric in metrics:
                if metric not in time_series.columns:
                    continue
                
                values = time_series[metric].dropna()
                
                if len(values) == 0:
                    continue
                
                # Compute trend statistics
                mean_val = values.mean()
                std_val = values.std()
                
                # Simple trend detection (last 20% vs first 20%)
                n = len(values)
                if n >= 10:
                    recent = values.iloc[-max(1, n//5):].mean()
                    historical = values.iloc[:max(1, n//5)].mean()
                    
                    if recent > historical * 1.1:
                        trend = "📈 **Increasing** (worsening)"
                    elif recent < historical * 0.9:
                        trend = "📉 **Decreasing** (improving)"
                    else:
                        trend = "➡️ **Stable**"
                else:
                    trend = "➡️ **Insufficient data**"
                
                content += f"### {metric}\n\n"
                content += f"- **Mean:** {mean_val:.3f}\n"
                content += f"- **Std Dev:** {std_val:.3f}\n"
                content += f"- **Trend:** {trend}\n"
                content += f"- **Observations:** {n}\n\n"
        
        content += "---\n"
        
        self.sections.append(ReportSection(
            title="Trend Analysis",
            content=content,
            level=2
        ))
    
    def _add_recommendations(
        self,
        summary_stats: Dict[str, Dict[str, float]],
        alerts: List[Dict[str, Any]]
    ):
        """Add recommendations section."""
        content = """## Recommendations

### Immediate Actions

"""
        
        # Critical alerts
        critical_alerts = [
            a for a in alerts
            if a.get('severity', '').upper() in ['CRITICAL', 'HIGH']
        ]
        
        if critical_alerts:
            content += "Critical issues require immediate attention:\n\n"
            for i, alert in enumerate(critical_alerts[:3], 1):
                metric = alert.get('metric_name', 'Unknown')
                content += f"{i}. **{metric}:** "
                
                actions = alert.get('recommended_actions', [])
                if actions:
                    content += actions[0] + "\n"
                else:
                    content += "Investigate and remediate\n"
            
            content += "\n"
        else:
            content += "*No immediate actions required.*\n\n"
        
        # General recommendations
        content += "### General Recommendations\n\n"
        
        # Check for consistent issues
        problem_metrics = [
            name for name, stats in summary_stats.items()
            if stats.get('mean', 0) > stats.get('threshold', 0.10)
        ]
        
        if problem_metrics:
            content += "**Consistently Problematic Metrics:**\n\n"
            for metric in problem_metrics:
                content += f"- **{metric}:** Consider implementing bias mitigation strategies\n"
            content += "\n"
        
        # Best practices
        content += """**Ongoing Best Practices:**

1. **Regular Monitoring:** Continue daily monitoring of all fairness metrics
2. **Data Quality:** Ensure protected attribute data is accurate and complete
3. **Model Retraining:** Consider retraining if drift is detected
4. **Stakeholder Communication:** Share findings with relevant stakeholders
5. **Documentation:** Maintain detailed records of interventions and outcomes

"""
        
        content += "---\n"
        
        self.sections.append(ReportSection(
            title="Recommendations",
            content=content,
            level=2
        ))
    
    def _add_technical_details(
        self,
        summary_stats: Dict[str, Dict[str, float]],
        metadata: Optional[Dict[str, Any]]
    ):
        """Add technical details section."""
        content = """## Technical Details

### Monitoring Configuration

"""
        
        if metadata:
            for key, value in metadata.items():
                content += f"- **{key}:** {value}\n"
        
        content += "\n### Metric Definitions\n\n"
        
        # Add metric definitions
        definitions = {
            'demographic_parity': (
                "Difference in positive prediction rates between groups. "
                "Lower values indicate better fairness."
            ),
            'equalized_odds': (
                "Maximum difference in TPR and FPR between groups. "
                "Measures both error rate parity."
            ),
            'equal_opportunity': (
                "Difference in TPR between groups. "
                "Focuses on equal access to positive outcomes."
            ),
        }
        
        for metric_name in summary_stats.keys():
            if metric_name in definitions:
                content += f"**{metric_name}:** {definitions[metric_name]}\n\n"
        
        # Statistical notes
        content += """### Statistical Notes

- All metrics computed with 95% confidence intervals
- Thresholds based on established fairness guidelines
- Bootstrap resampling (1000 iterations) for uncertainty quantification
- Minimum group size requirement: n ≥ 30

"""
        
        content += "---\n"
        
        self.sections.append(ReportSection(
            title="Technical Details",
            content=content,
            level=2
        ))
    
    def _compile_report(self) -> str:
        """Compile all sections into final report."""
        # Generate table of contents
        toc = "## Table of Contents\n\n"
        
        section_numbers = {}
        current_number = [0, 0, 0, 0, 0]  # Support up to 5 levels
        
        for section in self.sections:
            if not section.include_in_toc or section.level == 1:
                continue
            
            # Update section numbering
            current_number[section.level - 1] += 1
            for i in range(section.level, 5):
                current_number[i] = 0
            
            # Create number string
            number = '.'.join(
                str(n) for n in current_number[:section.level] if n > 0
            )
            
            section_numbers[section.title] = number
            
            # Add to TOC
            indent = "  " * (section.level - 2)
            toc += f"{indent}{number}. [{section.title}](#{self._make_anchor(section.title)})\n"
        
        toc += "\n---\n\n"
        
        # Compile full report
        report = ""
        
        for section in self.sections:
            if section.title == "Header":
                report += section.content
            else:
                report += section.content
        
        # Insert TOC after header
        parts = report.split("---\n", 1)
        if len(parts) == 2:
            report = parts[0] + "---\n\n" + toc + parts[1]
        
        # Add footer
        report += self._generate_footer()
        
        return report
    
    def _generate_footer(self) -> str:
        """Generate report footer."""
        return f"""
---

## Report Information

**Generated by:** Fairness Pipeline Development Toolkit  
**Version:** 1.0.0  
**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

For questions or concerns about this report, contact your ML operations team.

---

*This is an automated report. The metrics and recommendations are based on statistical analysis and should be reviewed by domain experts before taking action.*
"""
    
    def _make_anchor(self, title: str) -> str:
        """Create Markdown anchor from title."""
        return title.lower().replace(' ', '-').replace(':', '')
    
    def _severity_to_priority(self, severity: str) -> int:
        """Convert severity to priority score."""
        priority_map = {
            'CRITICAL': 5,
            'HIGH': 4,
            'MEDIUM': 3,
            'LOW': 2,
            'INFO': 1,
        }
        return priority_map.get(severity.upper(), 0)


def generate_monitoring_report(
    summary_stats: Dict[str, Dict[str, float]],
    alerts: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    time_series: Optional[pd.DataFrame] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Convenience function to generate monitoring report.
    
    Args:
        summary_stats: Summary statistics for metrics
        alerts: List of alerts
        output_path: Output file path (optional)
        time_series: Time series data (optional)
        metadata: Additional metadata (optional)
    
    Returns:
        Path to generated report
    """
    if output_path:
        output_dir = Path(output_path).parent
    else:
        output_dir = Path("reports")
    
    generator = FairnessMonitoringReport(output_dir=output_dir)
    
    report_path = generator.generate_monitoring_report(
        summary_stats=summary_stats,
        alerts=alerts,
        time_series=time_series,
        metadata=metadata
    )
    
    return report_path


class ABTestReport:
    """Generator for A/B test analysis reports."""
    
    def __init__(self, output_dir: Path = Path("reports")):
        """
        Initialize A/B test report generator.
        
        Args:
            output_dir: Directory for saving reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized ABTestReport (dir={output_dir})")
    
    def generate_ab_test_report(
        self,
        test_results: Dict[str, Any],
        experiment_metadata: Dict[str, Any]
    ) -> str:
        """
        Generate A/B test analysis report.
        
        Args:
            test_results: Results from FairnessABTestAnalyzer
            experiment_metadata: Metadata about the experiment
        
        Returns:
            Path to generated report
        """
        logger.info("Generating A/B test report")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        content = f"""# A/B Test Analysis Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## Experiment Details

"""
        
        # Add metadata
        for key, value in experiment_metadata.items():
            content += f"- **{key}:** {value}\n"
        
        content += "\n---\n\n"
        
        # Overall results
        content += "## Overall Results\n\n"
        
        overall = test_results.get('overall', {})
        for metric_name, result in overall.items():
            result_dict = result.to_dict() if hasattr(result, 'to_dict') else result
            
            content += f"### {metric_name}\n\n"
            content += f"- **Control:** {result_dict['control_value']:.4f}\n"
            content += f"- **Treatment:** {result_dict['treatment_value']:.4f}\n"
            content += f"- **Difference:** {result_dict['absolute_difference']:.4f}\n"
            content += f"- **P-value:** {result_dict['p_value']:.4f}\n"
            content += f"- **Significant:** {'Yes ✓' if result_dict['is_significant'] else 'No ✗'}\n"
            content += f"- **Interpretation:** {result_dict['interpretation']}\n\n"
        
        # Multi-objective analysis
        if 'multi_objective' in test_results:
            multi_obj = test_results['multi_objective']
            
            content += "## Multi-Objective Analysis\n\n"
            content += f"**Outcome:** {multi_obj['outcome']}\n\n"
            content += f"**Recommendation:** {multi_obj['recommendation']}\n\n"
        
        # Heterogeneous effects
        if 'heterogeneous' in test_results:
            content += "## Heterogeneous Treatment Effects\n\n"
            
            for metric_name, subgroup_results in test_results['heterogeneous'].items():
                content += f"### {metric_name}\n\n"
                
                content += "| Subgroup | Control | Treatment | Effect | P-value | Significant |\n"
                content += "|----------|---------|-----------|--------|---------|-------------|\n"
                
                for result in subgroup_results:
                    result_dict = result.to_dict() if hasattr(result, 'to_dict') else result
                    
                    content += (
                        f"| {result_dict['subgroup']} | "
                        f"{result_dict['control_value']:.3f} | "
                        f"{result_dict['treatment_value']:.3f} | "
                        f"{result_dict['treatment_effect']:.3f} | "
                        f"{result_dict['p_value']:.3f} | "
                        f"{'Yes' if result_dict['is_significant'] else 'No'} |\n"
                    )
                
                content += "\n"
        
        # Save report
        filename = f"ab_test_report_{timestamp}.md"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"A/B test report saved to {filepath}")
        
        return str(filepath)