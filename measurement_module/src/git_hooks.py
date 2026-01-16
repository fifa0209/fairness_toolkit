# New file: measurement_module/scripts/git_hooks.py

#!/usr/bin/env python3
"""
Git hooks for fairness validation.

Install with:
    python -m measurement_module.scripts.git_hooks install
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional

def install_pre_commit_hook(repo_path: Optional[str] = None):
    """
    Install pre-commit hook for fairness checks.
    
    Args:
        repo_path: Path to git repository (None = current directory)
    """
    if repo_path is None:
        repo_path = os.getcwd()
    
    hooks_dir = Path(repo_path) / '.git' / 'hooks'
    
    if not hooks_dir.exists():
        print("❌ Not a git repository or hooks directory missing")
        return False
    
    hook_path = hooks_dir / 'pre-commit'
    
    hook_content = '''#!/usr/bin/env python3
"""Pre-commit hook: Run fairness checks on staged files."""

import sys
import subprocess

def check_fairness_tests():
    """Run fairness tests before commit."""
    print("🔍 Running fairness validation tests...")
    
    # Run pytest on fairness tests
    result = subprocess.run(
        ['pytest', 'tests/test_fairness.py', '-v'],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("❌ Fairness tests failed!")
        print(result.stdout)
        print(result.stderr)
        print("\\n💡 Fix fairness issues before committing.")
        return False
    
    print("✅ All fairness checks passed")
    return True

if __name__ == '__main__':
    if not check_fairness_tests():
        sys.exit(1)
    sys.exit(0)
'''
    
    # Write hook
    hook_path.write_text(hook_content)
    hook_path.chmod(0o755)  # Make executable
    
    print(f"✅ Pre-commit hook installed at: {hook_path}")
    return True


def generate_fairness_report_card(
    results: dict,
    output_path: str = "fairness_report_card.md"
):
    """
    Generate markdown report card for Pull Requests.
    
    Args:
        results: Dictionary of fairness metric results
        output_path: Where to save the report
    """
    lines = [
        "# 🛡️ Fairness Report Card",
        "",
        "## Summary",
        ""
    ]
    
    all_pass = all(r.is_fair for r in results.values())
    status_emoji = "✅" if all_pass else "❌"
    
    lines.append(f"{status_emoji} **Overall Status**: {'PASS' if all_pass else 'FAIL'}")
    lines.append("")
    
    lines.append("## Metrics")
    lines.append("")
    lines.append("| Metric | Value | Threshold | Status |")
    lines.append("|--------|-------|-----------|--------|")
    
    for metric_name, result in results.items():
        status = "✅ PASS" if result.is_fair else "❌ FAIL"
        lines.append(
            f"| {metric_name} | {result.value:.4f} | {result.threshold} | {status} |"
        )
    
    lines.append("")
    lines.append("## Group Metrics")
    lines.append("")
    
    for metric_name, result in results.items():
        lines.append(f"### {metric_name}")
        for group, value in result.group_metrics.items():
            lines.append(f"- **{group}**: {value:.4f}")
        lines.append("")
    
    # Write report
    report = "\n".join(lines)
    Path(output_path).write_text(report)
    
    print(f"📊 Report card generated: {output_path}")
    return report


def create_pr_comment_workflow():
    """
    Generate GitHub Actions workflow for PR comments.
    """
    workflow = '''name: Fairness Check

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  fairness-check:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest
    
    - name: Run fairness tests
      run: |
        pytest tests/test_fairness.py -v --tb=short
    
    - name: Generate fairness report
      if: always()
      run: |
        python -m measurement_module.scripts.git_hooks generate-report
    
    - name: Comment PR
      if: always()
      uses: actions/github-script@v6
      with:
        script: |
          const fs = require('fs');
          const report = fs.readFileSync('fairness_report_card.md', 'utf8');
          
          github.rest.issues.createComment({
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: report
          });
'''
    
    workflows_dir = Path('.github/workflows')
    workflows_dir.mkdir(parents=True, exist_ok=True)
    
    workflow_path = workflows_dir / 'fairness-check.yml'
    workflow_path.write_text(workflow)
    
    print(f"✅ GitHub Actions workflow created: {workflow_path}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'install':
            install_pre_commit_hook()
        elif command == 'generate-report':
            # This would be called by CI/CD with actual results
            print("Generate report (implement based on your test results)")
        elif command == 'create-workflow':
            create_pr_comment_workflow()
        else:
            print(f"Unknown command: {command}")
            print("Available: install, generate-report, create-workflow")
    else:
        print("Usage: python git_hooks.py [install|generate-report|create-workflow]")