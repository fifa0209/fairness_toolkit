"""
Tests for Dashboard and Report Generator

Tests visualization generation and report creation for fairness monitoring.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

from monitoring_module.src.dashboard import (
    FairnessMonitoringDashboard,
    generate_monitoring_report,
)
from monitoring_module.src.report_generator import (
    FairnessMonitoringReport,
    ABTestReport,
    ReportSection,
    generate_monitoring_report as gen_report_func,
)


# Skip tests if Plotly not available
pytest.importorskip("plotly", reason="Plotly required for dashboard tests")


@pytest.fixture
def sample_time_series():
    """Create sample time series data."""
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    
    df = pd.DataFrame({
        'timestamp': dates,
        'demographic_parity': np.random.uniform(0.05, 0.15, 30),
        'equalized_odds': np.random.uniform(0.04, 0.12, 30),
    })
    
    return df


@pytest.fixture
def sample_group_metrics():
    """Create sample per-group metrics."""
    return {
        'group_0': {
            'demographic_parity': 0.08,
            'equalized_odds': 0.06,
        },
        'group_1': {
            'demographic_parity': 0.12,
            'equalized_odds': 0.10,
        }
    }


@pytest.fixture
def sample_alerts():
    """Create sample alerts."""
    return [
        {
            'timestamp': datetime.now() - timedelta(hours=2),
            'severity': 'HIGH',
            'metric_name': 'demographic_parity',
            'alert_type': 'threshold_violation',
            'message': 'Threshold exceeded',
            'current_value': 0.15,
            'threshold': 0.10,
            'recommended_actions': ['Review model', 'Check data'],
        },
        {
            'timestamp': datetime.now() - timedelta(hours=1),
            'severity': 'LOW',
            'metric_name': 'equalized_odds',
            'alert_type': 'drift_detected',
            'message': 'Drift detected',
            'current_value': 0.11,
            'threshold': 0.10,
            'recommended_actions': ['Monitor closely'],
        }
    ]


class TestFairnessMonitoringDashboard:
    """Tests for FairnessMonitoringDashboard."""
    
    def test_initialization(self):
        """Test dashboard initialization."""
        dashboard = FairnessMonitoringDashboard()
        
        # Should initialize without error
        assert dashboard is not None
    
    def test_plot_metrics_over_time(self, sample_time_series):
        """Test plotting metrics over time."""
        dashboard = FairnessMonitoringDashboard()
        
        fig = dashboard.plot_metrics_over_time(
            sample_time_series,
            metrics=['demographic_parity', 'equalized_odds'],
            threshold=0.10
        )
        
        # Check figure was created
        assert fig is not None
        assert len(fig.data) >= 2  # At least 2 traces (one per metric)
        
        # Check title
        assert 'Fairness Metrics' in fig.layout.title.text
    
    def test_plot_metrics_over_time_single_metric(self, sample_time_series):
        """Test plotting single metric."""
        dashboard = FairnessMonitoringDashboard()
        
        fig = dashboard.plot_metrics_over_time(
            sample_time_series,
            metrics=['demographic_parity'],
            threshold=0.10
        )
        
        assert fig is not None
        # Should have metric trace plus threshold line
    
    def test_plot_metrics_over_time_custom_title(self, sample_time_series):
        """Test plotting with custom title."""
        dashboard = FairnessMonitoringDashboard()
        
        fig = dashboard.plot_metrics_over_time(
            sample_time_series,
            title="Custom Dashboard Title"
        )
        
        assert "Custom Dashboard Title" in fig.layout.title.text
    
    def test_plot_group_comparison(self, sample_group_metrics):
        """Test plotting group comparison."""
        dashboard = FairnessMonitoringDashboard()
        
        fig = dashboard.plot_group_comparison(sample_group_metrics)
        
        assert fig is not None
        assert len(fig.data) >= 2  # Bars for each metric
        
        # Check that groups are in x-axis
        x_values = []
        for trace in fig.data:
            x_values.extend(list(trace.x))
        
        assert 'group_0' in x_values
        assert 'group_1' in x_values
    
    def test_plot_alert_timeline_with_alerts(self, sample_alerts):
        """Test plotting alert timeline."""
        dashboard = FairnessMonitoringDashboard()
        
        fig = dashboard.plot_alert_timeline(sample_alerts)
        
        assert fig is not None
        # Should have traces for each severity level present
    
    def test_plot_alert_timeline_empty(self):
        """Test plotting alert timeline with no alerts."""
        dashboard = FairnessMonitoringDashboard()
        
        fig = dashboard.plot_alert_timeline([])
        
        assert fig is not None
        # Should show "No alerts" message
    
    # def test_create_dashboard(self, sample_time_series, sample_group_metrics, sample_alerts):
    #     """Test creating complete dashboard."""
    #     dashboard = FairnessMonitoringDashboard()
        
    #     with tempfile.TemporaryDirectory() as tmpdir:
    #         output_path = Path(tmpdir) / 'test_dashboard.html'
            
    #         dashboard.create_dashboard(
    #             time_series=sample_time_series,
    #             group_metrics=sample_group_metrics,
    #             alerts=sample_alerts,
    #             output_path=str(output_path)
    #         )
            
    #         # Check file was created
    #         assert output_path.exists()
    #         assert output_path.stat().st_size > 0
            
    #         # Verify HTML content
    #         content = output_path.read_text()
    #         assert 'Fairness Monitoring Dashboard' in content
    def test_create_dashboard(self, sample_time_series, sample_group_metrics, sample_alerts):
        """Test creating complete dashboard."""
        dashboard = FairnessMonitoringDashboard()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test_dashboard.html'
            
            dashboard.create_dashboard(
                time_series=sample_time_series,
                group_metrics=sample_group_metrics,
                alerts=sample_alerts,
                output_path=str(output_path)
            )
            
            # Check file was created
            assert output_path.exists()
            assert output_path.stat().st_size > 0
            
            # Verify HTML content - FIX: Add encoding='utf-8'
            content = output_path.read_text(encoding='utf-8')
            assert 'Fairness Monitoring Dashboard' in content


# class TestDashboardHelperFunction:
#     """Tests for generate_monitoring_report helper function."""
    
#     def test_generate_monitoring_report(self, sample_alerts):
#         """Test report generation function."""
#         summary_stats = {
#             'demographic_parity': {
#                 'mean': 0.08,
#                 'std': 0.02,
#                 'min': 0.05,
#                 'max': 0.12,
#                 'current': 0.09,
#             }
#         }
        
#         with tempfile.TemporaryDirectory() as tmpdir:
#             output_path = Path(tmpdir) / 'report.md'
            
#             generate_monitoring_report(
#                 summary_stats=summary_stats,
#                 alerts=sample_alerts,
#                 output_path=str(output_path)
#             )
            
#             # Check file exists
#             assert output_path.exists()
            
#             # Check content
#             content = output_path.read_text()
#             assert 'Fairness Monitoring Report' in content

class TestFairnessMonitoringReport:
    """Tests for FairnessMonitoringReport."""
    
    @pytest.fixture
    def report_generator(self):
        """Create report generator."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = FairnessMonitoringReport(output_dir=Path(tmpdir))
            yield generator
    
    @pytest.fixture
    def summary_stats(self):
        """Create sample summary statistics."""
        return {
            'demographic_parity': {
                'mean': 0.08,
                'std': 0.02,
                'min': 0.05,
                'max': 0.12,
                'current': 0.09,
                'threshold': 0.10,
            },
            'equalized_odds': {
                'mean': 0.07,
                'std': 0.015,
                'min': 0.04,
                'max': 0.10,
                'current': 0.08,
                'threshold': 0.10,
            }
        }
    
    def test_initialization(self, report_generator):
        """Test report generator initialization."""
        assert report_generator.output_dir.exists()
        assert len(report_generator.sections) == 0
    
    def test_generate_monitoring_report(self, report_generator, summary_stats, sample_alerts):
        """Test generating complete monitoring report."""
        report_path = report_generator.generate_monitoring_report(
            summary_stats=summary_stats,
            alerts=sample_alerts
        )
        
        # Check file was created
        assert Path(report_path).exists()
        
        # Check content - FIX: Add encoding='utf-8'
        content = Path(report_path).read_text(encoding='utf-8')
        assert '# Fairness Monitoring Report' in content
        assert 'Executive Summary' in content
        assert 'demographic_parity' in content
    
    def test_generate_report_with_time_series(self, report_generator, summary_stats, 
                                              sample_alerts, sample_time_series):
        """Test report generation with time series."""
        report_path = report_generator.generate_monitoring_report(
            summary_stats=summary_stats,
            alerts=sample_alerts,
            time_series=sample_time_series
        )
        
        # FIX: Add encoding='utf-8'
        content = Path(report_path).read_text(encoding='utf-8')
        assert 'Trend Analysis' in content
    
    def test_generate_report_with_metadata(self, report_generator, summary_stats, sample_alerts):
        """Test report with metadata."""
        metadata = {
            'Model': 'LogisticRegression',
            'Version': '1.0.0',
            'Dataset': 'test_data',
        }
        
        report_path = report_generator.generate_monitoring_report(
            summary_stats=summary_stats,
            alerts=sample_alerts,
            metadata=metadata
        )
        
        # FIX: Add encoding='utf-8'
        content = Path(report_path).read_text(encoding='utf-8')
        assert 'Model' in content
        assert 'LogisticRegression' in content
    
    def test_add_executive_summary_healthy(self, report_generator):
        """Test executive summary for healthy status."""
        summary_stats = {
            'demographic_parity': {
                'mean': 0.05,
                'current': 0.05,
                'std': 0.01,
                'min': 0.04,
                'max': 0.06,
            }
        }
        
        report_generator._add_executive_summary(summary_stats, [])
        
        assert len(report_generator.sections) > 0
        content = report_generator.sections[-1].content
        # FIX: Update assertion to match new text-based status
        assert 'HEALTHY' in content or 'GOOD' in content
    
    def test_add_executive_summary_critical(self, report_generator, sample_alerts):
        """Test executive summary for critical status."""
        summary_stats = {
            'demographic_parity': {'mean': 0.15, 'current': 0.15, 'std': 0.02, 'min': 0.12, 'max': 0.18}
        }
        
        critical_alert = sample_alerts[0].copy()
        critical_alert['severity'] = 'CRITICAL'
        
        report_generator._add_executive_summary(summary_stats, [critical_alert])
        
        content = report_generator.sections[-1].content
        # FIX: Update assertion to match new text-based status
        assert 'CRITICAL' in content
    
    def test_add_metric_overview(self, report_generator, summary_stats):
        """Test metric overview section."""
        report_generator._add_metric_overview(summary_stats)
        
        assert len(report_generator.sections) > 0
        content = report_generator.sections[-1].content
        
        # Should have table
        assert '|' in content
        assert 'Current' in content
        assert 'Mean' in content
    
    def test_add_alert_summary_no_alerts(self, report_generator):
        """Test alert summary with no alerts."""
        report_generator._add_alert_summary([])
        
        content = report_generator.sections[-1].content
        assert 'No alerts' in content
    
    def test_add_alert_summary_with_alerts(self, report_generator, sample_alerts):
        """Test alert summary with alerts."""
        report_generator._add_alert_summary(sample_alerts)
        
        content = report_generator.sections[-1].content
        assert 'Total Alerts' in content
        assert str(len(sample_alerts)) in content
    
    def test_add_recommendations(self, report_generator, summary_stats, sample_alerts):
        """Test recommendations section."""
        report_generator._add_recommendations(summary_stats, sample_alerts)
        
        content = report_generator.sections[-1].content
        assert 'Recommendations' in content
        assert 'Actions' in content or 'action' in content.lower()
    
    def test_compile_report(self, report_generator, summary_stats, sample_alerts):
        """Test report compilation."""
        report_generator._add_header(None)
        report_generator._add_executive_summary(summary_stats, sample_alerts)
        report_generator._add_metric_overview(summary_stats)
        
        compiled = report_generator._compile_report()
        
        assert isinstance(compiled, str)
        assert len(compiled) > 0
        assert '# Fairness Monitoring Report' in compiled


class TestReportSection:
    """Tests for ReportSection dataclass."""
    
    def test_report_section_creation(self):
        """Test creating report section."""
        section = ReportSection(
            title="Test Section",
            content="Test content",
            level=2,
            include_in_toc=True
        )
        
        assert section.title == "Test Section"
        assert section.content == "Test content"
        assert section.level == 2
        assert section.include_in_toc is True


class TestFairnessMonitoringReport:
    """Tests for FairnessMonitoringReport."""
    
    @pytest.fixture
    def report_generator(self):
        """Create report generator."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = FairnessMonitoringReport(output_dir=Path(tmpdir))
            yield generator
    
    @pytest.fixture
    def summary_stats(self):
        """Create sample summary statistics."""
        return {
            'demographic_parity': {
                'mean': 0.08,
                'std': 0.02,
                'min': 0.05,
                'max': 0.12,
                'current': 0.09,
                'threshold': 0.10,
            },
            'equalized_odds': {
                'mean': 0.07,
                'std': 0.015,
                'min': 0.04,
                'max': 0.10,
                'current': 0.08,
                'threshold': 0.10,
            }
        }
    
    def test_initialization(self, report_generator):
        """Test report generator initialization."""
        assert report_generator.output_dir.exists()
        assert len(report_generator.sections) == 0
    
    def test_generate_monitoring_report(self, report_generator, summary_stats, sample_alerts):
        """Test generating complete monitoring report."""
        report_path = report_generator.generate_monitoring_report(
            summary_stats=summary_stats,
            alerts=sample_alerts
        )
        
        # Check file was created
        assert Path(report_path).exists()
        
        # Check content
        content = Path(report_path).read_text(encoding='utf-8')
        assert '# Fairness Monitoring Report' in content
        assert 'Executive Summary' in content
        assert 'demographic_parity' in content
    
    def test_generate_report_with_time_series(self, report_generator, summary_stats, 
                                              sample_alerts, sample_time_series):
        """Test report generation with time series."""
        report_path = report_generator.generate_monitoring_report(
            summary_stats=summary_stats,
            alerts=sample_alerts,
            time_series=sample_time_series
        )
        
        content = Path(report_path).read_text(encoding='utf-8')
        assert 'Trend Analysis' in content
    
    def test_generate_report_with_metadata(self, report_generator, summary_stats, sample_alerts):
        """Test report with metadata."""
        metadata = {
            'Model': 'LogisticRegression',
            'Version': '1.0.0',
            'Dataset': 'test_data',
        }
        
        report_path = report_generator.generate_monitoring_report(
            summary_stats=summary_stats,
            alerts=sample_alerts,
            metadata=metadata
        )
        
        content = Path(report_path).read_text(encoding='utf-8')
        assert 'Model' in content
        assert 'LogisticRegression' in content
    
    def test_add_executive_summary_healthy(self, report_generator):
        """Test executive summary for healthy status."""
        summary_stats = {
            'demographic_parity': {
                'mean': 0.05,
                'current': 0.05,
                'std': 0.01,
                'min': 0.04,
                'max': 0.06,
            }
        }
        
        report_generator._add_executive_summary(summary_stats, [])
        
        assert len(report_generator.sections) > 0
        content = report_generator.sections[-1].content
        assert 'HEALTHY' in content or '🟢' in content
    
    def test_add_executive_summary_critical(self, report_generator, sample_alerts):
        """Test executive summary for critical status."""
        summary_stats = {
            'demographic_parity': {'mean': 0.15, 'current': 0.15, 'std': 0.02, 'min': 0.12, 'max': 0.18}
        }
        
        critical_alert = sample_alerts[0].copy()
        critical_alert['severity'] = 'CRITICAL'
        
        report_generator._add_executive_summary(summary_stats, [critical_alert])
        
        content = report_generator.sections[-1].content
        assert 'CRITICAL' in content or '🔴' in content
    
    def test_add_metric_overview(self, report_generator, summary_stats):
        """Test metric overview section."""
        report_generator._add_metric_overview(summary_stats)
        
        assert len(report_generator.sections) > 0
        content = report_generator.sections[-1].content
        
        # Should have table
        assert '|' in content
        assert 'Current' in content
        assert 'Mean' in content
    
    def test_add_alert_summary_no_alerts(self, report_generator):
        """Test alert summary with no alerts."""
        report_generator._add_alert_summary([])
        
        content = report_generator.sections[-1].content
        assert 'No alerts' in content
    
    def test_add_alert_summary_with_alerts(self, report_generator, sample_alerts):
        """Test alert summary with alerts."""
        report_generator._add_alert_summary(sample_alerts)
        
        content = report_generator.sections[-1].content
        assert 'Total Alerts' in content
        assert str(len(sample_alerts)) in content
    
    def test_add_recommendations(self, report_generator, summary_stats, sample_alerts):
        """Test recommendations section."""
        report_generator._add_recommendations(summary_stats, sample_alerts)
        
        content = report_generator.sections[-1].content
        assert 'Recommendations' in content
        assert 'Actions' in content or 'action' in content.lower()
    
    def test_compile_report(self, report_generator, summary_stats, sample_alerts):
        """Test report compilation."""
        report_generator._add_header(None)
        report_generator._add_executive_summary(summary_stats, sample_alerts)
        report_generator._add_metric_overview(summary_stats)
        
        compiled = report_generator._compile_report()
        
        assert isinstance(compiled, str)
        assert len(compiled) > 0
        assert '# Fairness Monitoring Report' in compiled


class TestABTestReport:
    """Tests for ABTestReport."""
    
    @pytest.fixture
    def ab_report_generator(self):
        """Create AB test report generator."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ABTestReport(output_dir=Path(tmpdir))
            yield generator
    
    @pytest.fixture
    def test_results(self):
        """Create sample test results."""
        from monitoring_module.src.ab_testing import ABTestResult
        
        return {
            'overall': {
                'accuracy': ABTestResult(
                    metric_name='accuracy',
                    control_value=0.80,
                    treatment_value=0.82,
                    absolute_difference=0.02,
                    relative_difference=0.025,
                    confidence_interval=(0.01, 0.03),
                    p_value=0.03,
                    is_significant=True,
                    effect_size=0.15,
                    sample_sizes={'control': 500, 'treatment': 500},
                    interpretation='Significant improvement',
                ),
            },
            'multi_objective': {
                'outcome': 'Win-Win',
                'recommendation': 'STRONGLY RECOMMEND',
            }
        }
    
    def test_initialization(self, ab_report_generator):
        """Test AB test report initialization."""
        assert ab_report_generator.output_dir.exists()
    
    def test_generate_ab_test_report(self, ab_report_generator, test_results):
        """Test generating AB test report."""
        metadata = {
            'Experiment Name': 'Test Experiment',
            'Start Date': '2024-01-01',
            'Duration': '30 days',
        }
        
        report_path = ab_report_generator.generate_ab_test_report(
            test_results=test_results,
            experiment_metadata=metadata
        )
        
        # Check file exists
        assert Path(report_path).exists()
        
        # Check content
        content = Path(report_path).read_text(encoding='utf-8')
        assert 'A/B Test Analysis Report' in content
        assert 'Test Experiment' in content
        assert 'accuracy' in content.lower()
    
    def test_ab_test_report_with_heterogeneous_effects(self, ab_report_generator):
        """Test AB test report with heterogeneous effects."""
        from monitoring_module.src.ab_testing import ABTestResult, HeterogeneousEffectResult
        
        test_results = {
            'overall': {
                'demographic_parity': ABTestResult(
                    metric_name='demographic_parity',
                    control_value=0.15,
                    treatment_value=0.10,
                    absolute_difference=-0.05,
                    relative_difference=-0.33,
                    confidence_interval=(-0.08, -0.02),
                    p_value=0.01,
                    is_significant=True,
                    effect_size=-0.5,
                    sample_sizes={'control': 500, 'treatment': 500},
                    interpretation='Significant improvement',
                ),
            },
            'heterogeneous': {
                'demographic_parity': [
                    HeterogeneousEffectResult(
                        subgroup='age_group=young',
                        control_value=0.12,
                        treatment_value=0.08,
                        treatment_effect=-0.04,
                        confidence_interval=(-0.07, -0.01),
                        p_value=0.02,
                        is_significant=True,
                        sample_size=200,
                    ),
                ]
            },
            'multi_objective': {
                'outcome': 'Win-Win',
                'recommendation': 'RECOMMEND',
            }
        }
        
        metadata = {'Experiment': 'Subgroup Analysis'}
        
        report_path = ab_report_generator.generate_ab_test_report(
            test_results=test_results,
            experiment_metadata=metadata
        )

        content = Path(report_path).read_text(encoding='utf-8')

        assert 'Heterogeneous Treatment Effects' in content
        assert 'age_group=young' in content


class TestReportGeneratorHelpers:
    """Tests for helper functions."""
    
    def test_generate_monitoring_report_function(self, sample_alerts):
        """Test generate_monitoring_report convenience function."""
        summary_stats = {
            'demographic_parity': {
                'mean': 0.08,
                'std': 0.02,
                'min': 0.05,
                'max': 0.12,
                'current': 0.09,
            }
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'report.md'
            
            report_path = gen_report_func(
                summary_stats=summary_stats,
                alerts=sample_alerts,
                output_path=str(output_path)
            )
            
            assert Path(report_path).exists()


class TestIntegration:
    """Integration tests for dashboard and reporting."""
    
    def test_complete_monitoring_report_workflow(self, sample_time_series, 
                                                  sample_group_metrics, sample_alerts):
        """Test complete monitoring report generation workflow."""
        # 1. Generate summary statistics
        summary_stats = {
            'demographic_parity': {
                'mean': sample_time_series['demographic_parity'].mean(),
                'std': sample_time_series['demographic_parity'].std(),
                'min': sample_time_series['demographic_parity'].min(),
                'max': sample_time_series['demographic_parity'].max(),
                'current': sample_time_series['demographic_parity'].iloc[-1],
                'threshold': 0.10,
            }
        }
        
        # 2. Generate report
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = FairnessMonitoringReport(output_dir=Path(tmpdir))
            
            report_path = generator.generate_monitoring_report(
                summary_stats=summary_stats,
                alerts=sample_alerts,
                time_series=sample_time_series,
                metadata={'Model': 'TestModel'}
            )
            
            # 3. Verify report
            assert Path(report_path).exists()
            content = Path(report_path).read_text(encoding='utf-8')
            
            # Should have all major sections
            assert 'Executive Summary' in content
            assert 'Metric Overview' in content
            assert 'Alert Summary' in content
            assert 'Recommendations' in content
    
    def test_dashboard_and_report_together(self, sample_time_series, 
                                            sample_group_metrics, sample_alerts):
        """Test generating both dashboard and report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate dashboard
            dashboard = FairnessMonitoringDashboard()
            dashboard_path = Path(tmpdir) / 'dashboard.html'
            
            dashboard.create_dashboard(
                time_series=sample_time_series,
                group_metrics=sample_group_metrics,
                alerts=sample_alerts,
                output_path=str(dashboard_path)
            )
            
            # Generate report
            summary_stats = {
                'demographic_parity': {
                    'mean': 0.08,
                    'std': 0.02,
                    'min': 0.05,
                    'max': 0.12,
                    'current': 0.09,
                }
            }
            
            report_path = Path(tmpdir) / 'report.md'
            generate_monitoring_report(
                summary_stats=summary_stats,
                alerts=sample_alerts,
                output_path=str(report_path)
            )
            
            # Both should exist
            assert dashboard_path.exists()
            assert report_path.exists()
            
            # Both should have substantial content
            assert dashboard_path.stat().st_size > 1000
            assert report_path.stat().st_size > 500