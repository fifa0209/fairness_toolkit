"""
Dashboard - Visualization and reporting for fairness monitoring.

Generates Plotly visualizations and Markdown reports for monitoring.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

from shared.logging import get_logger

logger = get_logger(__name__)

# Try to import plotly
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available. Install with: pip install plotly")


class FairnessMonitoringDashboard:
    """
    Generate interactive dashboards for fairness monitoring.
    
    Creates Plotly visualizations of fairness metrics over time.
    
    Example:
        >>> dashboard = FairnessMonitoringDashboard()
        >>> 
        >>> # Add time series data
        >>> fig = dashboard.plot_metrics_over_time(
        ...     time_series_df,
        ...     metrics=['demographic_parity', 'equalized_odds']
        ... )
        >>> fig.write_html('dashboard.html')
    """
    
    def __init__(self):
        """Initialize dashboard generator."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly required. Install with: pip install plotly")
    
    def plot_metrics_over_time(
        self,
        time_series: pd.DataFrame,
        metrics: List[str] = None,
        threshold: float = 0.1,
        title: str = "Fairness Metrics Over Time",
    ) -> go.Figure:
        """
        Plot fairness metrics as time series.
        
        Args:
            time_series: DataFrame with 'timestamp' and metric columns
            metrics: List of metrics to plot (None = all)
            threshold: Fairness threshold line
            title: Plot title
            
        Returns:
            Plotly Figure object
        """
        if metrics is None:
            metrics = [col for col in time_series.columns if col != 'timestamp']
        
        fig = go.Figure()
        
        # Add trace for each metric
        for metric in metrics:
            if metric not in time_series.columns:
                continue
            
            fig.add_trace(go.Scatter(
                x=time_series['timestamp'],
                y=time_series[metric],
                mode='lines+markers',
                name=metric.replace('_', ' ').title(),
                line=dict(width=2),
                marker=dict(size=6),
            ))
        
        # Add threshold line
        fig.add_hline(
            y=threshold,
            line_dash='dash',
            line_color='red',
            annotation_text=f'Threshold ({threshold})',
            annotation_position='right',
        )
        
        # Layout
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Metric Value',
            hovermode='x unified',
            template='plotly_white',
            height=500,
        )
        
        return fig
    
    def plot_group_comparison(
        self,
        group_metrics: Dict[str, Dict[str, float]],
        title: str = "Per-Group Metrics",
    ) -> go.Figure:
        """
        Plot metrics comparison across groups.
        
        Args:
            group_metrics: Dict of {group_name: {metric: value}}
            title: Plot title
            
        Returns:
            Plotly Figure object
        """
        # Prepare data
        groups = list(group_metrics.keys())
        metrics = list(next(iter(group_metrics.values())).keys())
        
        fig = go.Figure()
        
        # Add bar for each metric
        for metric in metrics:
            values = [group_metrics[g].get(metric, 0) for g in groups]
            
            fig.add_trace(go.Bar(
                name=metric.replace('_', ' ').title(),
                x=groups,
                y=values,
                text=[f'{v:.3f}' for v in values],
                textposition='auto',
            ))
        
        # Layout
        fig.update_layout(
            title=title,
            xaxis_title='Group',
            yaxis_title='Metric Value',
            barmode='group',
            template='plotly_white',
            height=500,
        )
        
        return fig
    
    def plot_alert_timeline(
        self,
        alerts: List[Dict],
        title: str = "Alert Timeline",
    ) -> go.Figure:
        """
        Plot alert occurrences over time.
        
        Args:
            alerts: List of alert dictionaries
            title: Plot title
            
        Returns:
            Plotly Figure object
        """
        if not alerts:
            # Return empty figure
            fig = go.Figure()
            fig.add_annotation(
                text="No alerts recorded",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
            )
            return fig
        
        # Prepare data
        df = pd.DataFrame(alerts)
        
        # Map severity to colors
        severity_colors = {
            'LOW': 'yellow',
            'HIGH': 'orange',
            'CRITICAL': 'red',
        }
        
        df['color'] = df['severity'].map(severity_colors)
        
        # Create scatter plot
        fig = go.Figure()
        
        for severity in ['LOW', 'HIGH', 'CRITICAL']:
            mask = df['severity'] == severity
            if not mask.any():
                continue
            
            fig.add_trace(go.Scatter(
                x=df[mask]['timestamp'],
                y=df[mask]['metric_name'],
                mode='markers',
                name=severity,
                marker=dict(
                    size=12,
                    color=severity_colors[severity],
                    line=dict(width=1, color='black'),
                ),
                text=df[mask]['message'],
                hovertemplate='%{text}<extra></extra>',
            ))
        
        # Layout
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Metric',
            template='plotly_white',
            height=400,
        )
        
        return fig
    
    def create_dashboard(
        self,
        time_series: pd.DataFrame,
        group_metrics: Dict[str, Dict[str, float]],
        alerts: List[Dict],
        output_path: str = 'fairness_dashboard.html',
    ) -> None:
        """
        Create comprehensive HTML dashboard.
        
        Args:
            time_series: Time series data
            group_metrics: Per-group metrics
            alerts: Alert history
            output_path: Output HTML file path
        """
        from plotly.subplots import make_subplots
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                'Fairness Metrics Over Time',
                'Current Per-Group Metrics',
                'Alert Timeline'
            ),
            row_heights=[0.4, 0.3, 0.3],
            vertical_spacing=0.12,
        )
        
        # Add metrics over time
        metrics = [col for col in time_series.columns if col != 'timestamp']
        for metric in metrics:
            fig.add_trace(
                go.Scatter(
                    x=time_series['timestamp'],
                    y=time_series[metric],
                    mode='lines+markers',
                    name=metric.replace('_', ' ').title(),
                ),
                row=1, col=1
            )
        
        # Add threshold line
        fig.add_hline(
            y=0.1, line_dash='dash', line_color='red',
            row=1, col=1
        )
        
        # Add group comparison
        groups = list(group_metrics.keys())
        for metric in metrics:
            values = [group_metrics[g].get(metric, 0) for g in groups]
            fig.add_trace(
                go.Bar(
                    x=groups,
                    y=values,
                    name=metric.replace('_', ' ').title(),
                    showlegend=False,
                ),
                row=2, col=1
            )
        
        # Add alerts
        if alerts:
            df_alerts = pd.DataFrame(alerts)
            severity_colors = {'LOW': 'yellow', 'HIGH': 'orange', 'CRITICAL': 'red'}
            
            for severity in ['LOW', 'HIGH', 'CRITICAL']:
                mask = df_alerts['severity'] == severity
                if not mask.any():
                    continue
                
                fig.add_trace(
                    go.Scatter(
                        x=df_alerts[mask]['timestamp'],
                        y=df_alerts[mask]['metric_name'],
                        mode='markers',
                        name=severity,
                        marker=dict(size=10, color=severity_colors[severity]),
                        showlegend=False,
                    ),
                    row=3, col=1
                )
        
        # Update layout
        fig.update_layout(
            title_text="Fairness Monitoring Dashboard",
            template='plotly_white',
            height=1200,
        )
        
        # Save
        fig.write_html(output_path)
        logger.info(f"Dashboard saved to {output_path}")


def generate_monitoring_report(
    summary_stats: Dict,
    alerts: List[Dict],
    output_path: str = 'monitoring_report.md',
) -> None:
    """
    Generate Markdown monitoring report.
    
    Args:
        summary_stats: Summary statistics from tracker
        alerts: List of alerts
        output_path: Output file path
    """
    lines = [
        "# Fairness Monitoring Report",
        f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## Summary Statistics\n",
    ]
    
    # Summary table
    for metric, stats in summary_stats.items():
        lines.append(f"\n### {metric.replace('_', ' ').title()}\n")
        lines.append(f"- **Current:** {stats['current']:.4f}")
        lines.append(f"- **Mean:** {stats['mean']:.4f}")
        lines.append(f"- **Std:** {stats['std']:.4f}")
        lines.append(f"- **Range:** [{stats['min']:.4f}, {stats['max']:.4f}]")
    
    # Alerts section
    lines.append(f"\n## Alerts ({len(alerts)} total)\n")
    
    if alerts:
        for alert in alerts:
            severity_emoji = {'LOW': '🟡', 'HIGH': '🟠', 'CRITICAL': '🔴'}
            emoji = severity_emoji.get(alert['severity'], '⚪')
            
            lines.append(f"\n### {emoji} {alert['severity']} - {alert['alert_type']}")
            lines.append(f"- **Time:** {alert['timestamp']}")
            lines.append(f"- **Metric:** {alert['metric_name']}")
            lines.append(f"- **Message:** {alert['message']}")
    else:
        lines.append("No alerts recorded.")
    
    # Write file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"Monitoring report saved to {output_path}")