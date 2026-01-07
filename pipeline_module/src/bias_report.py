"""
Bias Report Generator - Create structured reports from bias detection results.

Generates JSON and Markdown reports for documentation and CI/CD integration.
"""

import json
from typing import Dict, List
from datetime import datetime
from pathlib import Path

from shared.schemas import BiasDetectionResult
from shared.logging import get_logger

logger = get_logger(__name__)


class BiasReportGenerator:
    """
    Generate structured bias detection reports.
    
    Example:
        >>> reporter = BiasReportGenerator()
        >>> reporter.add_result('representation', repr_result)
        >>> reporter.add_result('proxy', proxy_result)
        >>> reporter.save_json('reports/bias_report.json')
        >>> reporter.save_markdown('reports/bias_report.md')
    """
    
    def __init__(self):
        self.results: Dict[str, BiasDetectionResult] = {}
        self.metadata = {
            'generated_at': datetime.now().isoformat(),
            'version': '0.1.0'
        }
    
    def add_result(self, name: str, result: BiasDetectionResult) -> None:
        """Add a bias detection result to the report."""
        self.results[name] = result
        logger.info(f"Added result: {name} (detected={result.detected})")
    
    def to_dict(self) -> dict:
        """Convert report to dictionary."""
        return {
            'metadata': self.metadata,
            'summary': self.get_summary(),
            'results': {
                name: result.to_dict()
                for name, result in self.results.items()
            }
        }
    
    def get_summary(self) -> dict:
        """Get summary statistics."""
        total = len(self.results)
        detected = sum(1 for r in self.results.values() if r.detected)
        
        severity_counts = {
            'high': sum(1 for r in self.results.values() if r.severity == 'high'),
            'medium': sum(1 for r in self.results.values() if r.severity == 'medium'),
            'low': sum(1 for r in self.results.values() if r.severity == 'low'),
        }
        
        return {
            'total_checks': total,
            'bias_detected': detected,
            'bias_free': total - detected,
            'severity_counts': severity_counts,
        }
    
    def save_json(self, filepath: str) -> None:
        """Save report as JSON."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Saved JSON report to {filepath}")
    
    def save_markdown(self, filepath: str) -> None:
        """Save report as Markdown."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        md = self._generate_markdown()
        
        with open(filepath, 'w') as f:
            f.write(md)
        
        logger.info(f"Saved Markdown report to {filepath}")
    
    def _generate_markdown(self) -> str:
        """Generate Markdown report content."""
        lines = [
            "# Bias Detection Report",
            f"\n**Generated:** {self.metadata['generated_at']}",
            f"\n## Summary\n",
        ]
        
        summary = self.get_summary()
        lines.append(f"- **Total Checks:** {summary['total_checks']}")
        lines.append(f"- **Bias Detected:** {summary['bias_detected']}")
        lines.append(f"- **Bias Free:** {summary['bias_free']}")
        lines.append(f"\n### Severity Breakdown\n")
        lines.append(f"- High: {summary['severity_counts']['high']}")
        lines.append(f"- Medium: {summary['severity_counts']['medium']}")
        lines.append(f"- Low: {summary['severity_counts']['low']}")
        
        lines.append(f"\n## Detailed Results\n")
        
        for name, result in self.results.items():
            status = "🔴 DETECTED" if result.detected else "🟢 NOT DETECTED"
            lines.append(f"\n### {name.replace('_', ' ').title()}\n")
            lines.append(f"**Status:** {status}")
            lines.append(f"**Severity:** {result.severity}")
            
            if result.affected_groups:
                lines.append(f"**Affected Groups:** {', '.join(result.affected_groups)}")
            
            if result.recommendations:
                lines.append(f"\n**Recommendations:**")
                for rec in result.recommendations:
                    lines.append(f"- {rec}")
        
        return '\n'.join(lines)
    
    def print_summary(self) -> None:
        """Print summary to console."""
        summary = self.get_summary()
        
        print("\n" + "=" * 60)
        print("BIAS DETECTION SUMMARY")
        print("=" * 60)
        print(f"Total Checks: {summary['total_checks']}")
        print(f"Bias Detected: {summary['bias_detected']}")
        print(f"Bias Free: {summary['bias_free']}")
        print("\nSeverity Breakdown:")
        print(f"  High: {summary['severity_counts']['high']}")
        print(f"  Medium: {summary['severity_counts']['medium']}")
        print(f"  Low: {summary['severity_counts']['low']}")
        print("=" * 60)
        
        for name, result in self.results.items():
            status = "🔴" if result.detected else "🟢"
            print(f"{status} {name}: {result.severity}")