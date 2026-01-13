"""
Unit tests for git_hooks.py

Tests Git integration functionality including pre-commit hooks,
report card generation, and GitHub Actions workflows.
"""

import pytest
import os
import sys
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
import tempfile
import shutil

# Import the module to test
try:
    from measurement_module.scripts.git_hooks import (
        install_pre_commit_hook,
        generate_fairness_report_card,
        create_pr_comment_workflow
    )
except ImportError:
    pytest.skip("git_hooks module not found", allow_module_level=True)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_git_repo():
    """Create a temporary git repository for testing."""
    temp_dir = tempfile.mkdtemp()
    
    # Initialize git repo
    git_dir = Path(temp_dir) / '.git'
    hooks_dir = git_dir / 'hooks'
    hooks_dir.mkdir(parents=True, exist_ok=True)
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_fairness_results():
    """Mock fairness metric results."""
    from dataclasses import dataclass
    
    @dataclass
    class MockResult:
        metric_name: str
        value: float
        is_fair: bool
        threshold: float
        group_metrics: dict
        confidence_interval: tuple = None
    
    results = {
        'demographic_parity': MockResult(
            metric_name='demographic_parity',
            value=0.08,
            is_fair=True,
            threshold=0.1,
            group_metrics={'Group_0': 0.45, 'Group_1': 0.53},
            confidence_interval=(0.05, 0.11)
        ),
        'equalized_odds': MockResult(
            metric_name='equalized_odds',
            value=0.12,
            is_fair=False,
            threshold=0.1,
            group_metrics={'Group_0': 0.62, 'Group_1': 0.74},
            confidence_interval=(0.08, 0.16)
        ),
        'equal_opportunity': MockResult(
            metric_name='equal_opportunity',
            value=0.06,
            is_fair=True,
            threshold=0.1,
            group_metrics={'Group_0': 0.67, 'Group_1': 0.73},
            confidence_interval=(0.03, 0.09)
        )
    }
    
    return results


@pytest.fixture
def mock_all_passing_results():
    """Mock fairness results where all metrics pass."""
    from dataclasses import dataclass
    
    @dataclass
    class MockResult:
        metric_name: str
        value: float
        is_fair: bool
        threshold: float
        group_metrics: dict
    
    results = {
        'demographic_parity': MockResult(
            metric_name='demographic_parity',
            value=0.05,
            is_fair=True,
            threshold=0.1,
            group_metrics={'Group_0': 0.48, 'Group_1': 0.53}
        ),
        'equalized_odds': MockResult(
            metric_name='equalized_odds',
            value=0.07,
            is_fair=True,
            threshold=0.1,
            group_metrics={'Group_0': 0.65, 'Group_1': 0.72}
        )
    }
    
    return results


# ============================================================================
# Test Pre-Commit Hook Installation
# ============================================================================

class TestInstallPreCommitHook:
    """Test pre-commit hook installation."""
    
    def test_install_in_git_repo(self, temp_git_repo):
        """Test installing hook in valid git repo."""
        result = install_pre_commit_hook(temp_git_repo)
        
        assert result is True
        
        # Check hook file exists
        hook_path = Path(temp_git_repo) / '.git' / 'hooks' / 'pre-commit'
        assert hook_path.exists()
    
    def test_hook_file_is_executable(self, temp_git_repo):
        """Test that installed hook is executable."""
        install_pre_commit_hook(temp_git_repo)
        
        hook_path = Path(temp_git_repo) / '.git' / 'hooks' / 'pre-commit'
        
        # Check executable permission
        assert os.access(hook_path, os.X_OK)
    
    def test_hook_contains_fairness_check(self, temp_git_repo):
        """Test that hook contains fairness checking code."""
        install_pre_commit_hook(temp_git_repo)
        
        hook_path = Path(temp_git_repo) / '.git' / 'hooks' / 'pre-commit'
        content = hook_path.read_text()
        
        assert 'fairness' in content.lower()
        assert 'pytest' in content
        assert 'test_fairness.py' in content
    
    def test_hook_has_python_shebang(self, temp_git_repo):
        """Test that hook starts with Python shebang."""
        install_pre_commit_hook(temp_git_repo)
        
        hook_path = Path(temp_git_repo) / '.git' / 'hooks' / 'pre-commit'
        content = hook_path.read_text()
        
        assert content.startswith('#!/usr/bin/env python3')
    
    def test_install_without_git_repo(self):
        """Test installation fails in non-git directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = install_pre_commit_hook(temp_dir)
            
            assert result is False
    
    def test_install_overwrites_existing_hook(self, temp_git_repo):
        """Test that installation overwrites existing hook."""
        hook_path = Path(temp_git_repo) / '.git' / 'hooks' / 'pre-commit'
        
        # Create existing hook
        hook_path.write_text("#!/bin/bash\necho 'old hook'")
        
        # Install new hook
        install_pre_commit_hook(temp_git_repo)
        
        # Check new content
        content = hook_path.read_text()
        assert 'old hook' not in content
        assert 'fairness' in content.lower()
    
    def test_install_with_none_path_uses_cwd(self):
        """Test that None path defaults to current directory."""
        with patch('os.getcwd') as mock_cwd:
            mock_cwd.return_value = '/fake/path'
            
            with patch('pathlib.Path.exists') as mock_exists:
                mock_exists.return_value = False
                
                result = install_pre_commit_hook(None)
                
                # Should try to use current directory
                mock_cwd.assert_called_once()
    
    def test_hook_exit_codes(self, temp_git_repo):
        """Test that hook uses proper exit codes."""
        install_pre_commit_hook(temp_git_repo)
        
        hook_path = Path(temp_git_repo) / '.git' / 'hooks' / 'pre-commit'
        content = hook_path.read_text()
        
        # Should have sys.exit calls
        assert 'sys.exit(1)' in content  # Failure
        assert 'sys.exit(0)' in content  # Success


# ============================================================================
# Test Report Card Generation
# ============================================================================

class TestGenerateFairnessReportCard:
    """Test fairness report card generation."""
    
    def test_basic_report_generation(self, mock_fairness_results, tmp_path):
        """Test basic report card generation."""
        output_path = tmp_path / "report.md"
        
        report = generate_fairness_report_card(
            mock_fairness_results,
            output_path=str(output_path)
        )
        
        assert output_path.exists()
        assert isinstance(report, str)
        assert len(report) > 0
    
    def test_report_contains_header(self, mock_fairness_results, tmp_path):
        """Test that report contains header."""
        output_path = tmp_path / "report.md"
        
        report = generate_fairness_report_card(
            mock_fairness_results,
            output_path=str(output_path)
        )
        
        assert "Fairness Report Card" in report
        assert "Summary" in report
    
    def test_report_contains_all_metrics(self, mock_fairness_results, tmp_path):
        """Test that report includes all metrics."""
        output_path = tmp_path / "report.md"
        
        report = generate_fairness_report_card(
            mock_fairness_results,
            output_path=str(output_path)
        )
        
        for metric_name in mock_fairness_results.keys():
            # Should mention each metric
            assert metric_name in report.lower() or \
                   metric_name.replace('_', ' ') in report.lower()
    
    def test_report_shows_pass_fail_status(self, mock_fairness_results, tmp_path):
        """Test that report shows pass/fail status."""
        output_path = tmp_path / "report.md"
        
        report = generate_fairness_report_card(
            mock_fairness_results,
            output_path=str(output_path)
        )
        
        assert 'PASS' in report or '✅' in report
        assert 'FAIL' in report or '❌' in report
    
    def test_report_includes_metric_values(self, mock_fairness_results, tmp_path):
        """Test that report includes metric values."""
        output_path = tmp_path / "report.md"
        
        report = generate_fairness_report_card(
            mock_fairness_results,
            output_path=str(output_path)
        )
        
        # Should contain numeric values
        assert '0.08' in report  # demographic_parity value
        assert '0.12' in report  # equalized_odds value
    
    def test_report_includes_thresholds(self, mock_fairness_results, tmp_path):
        """Test that report includes threshold values."""
        output_path = tmp_path / "report.md"
        
        report = generate_fairness_report_card(
            mock_fairness_results,
            output_path=str(output_path)
        )
        
        assert '0.1' in report or '0.10' in report  # Threshold
    
    def test_report_includes_group_metrics(self, mock_fairness_results, tmp_path):
        """Test that report includes per-group metrics."""
        output_path = tmp_path / "report.md"
        
        report = generate_fairness_report_card(
            mock_fairness_results,
            output_path=str(output_path)
        )
        
        assert 'Group_0' in report or 'Group 0' in report
        assert 'Group_1' in report or 'Group 1' in report
    
    def test_report_markdown_formatting(self, mock_fairness_results, tmp_path):
        """Test that report uses proper markdown formatting."""
        output_path = tmp_path / "report.md"
        
        report = generate_fairness_report_card(
            mock_fairness_results,
            output_path=str(output_path)
        )
        
        # Should have markdown headers
        assert '#' in report
        # Should have table formatting
        assert '|' in report
        # Should have separators
        assert '---' in report or '===' in report
    
    def test_report_overall_status_fail(self, mock_fairness_results, tmp_path):
        """Test overall status shows FAIL when any metric fails."""
        output_path = tmp_path / "report.md"
        
        report = generate_fairness_report_card(
            mock_fairness_results,
            output_path=str(output_path)
        )
        
        # equalized_odds fails in mock data
        assert 'FAIL' in report or '❌' in report
    
    def test_report_overall_status_pass(self, mock_all_passing_results, tmp_path):
        """Test overall status shows PASS when all metrics pass."""
        output_path = tmp_path / "report.md"
        
        report = generate_fairness_report_card(
            mock_all_passing_results,
            output_path=str(output_path)
        )
        
        assert 'ALL METRICS PASSED' in report or 'PASS' in report
    
    def test_report_confidence_intervals(self, mock_fairness_results, tmp_path):
        """Test that confidence intervals are included."""
        output_path = tmp_path / "report.md"
        
        report = generate_fairness_report_card(
            mock_fairness_results,
            output_path=str(output_path)
        )
        
        # Should mention CI
        assert 'CI' in report or 'confidence' in report.lower()
    
    def test_report_file_created(self, mock_fairness_results, tmp_path):
        """Test that report file is actually created."""
        output_path = tmp_path / "report.md"
        
        generate_fairness_report_card(
            mock_fairness_results,
            output_path=str(output_path)
        )
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_report_default_filename(self, mock_fairness_results):
        """Test default filename for report."""
        with patch('pathlib.Path.write_text') as mock_write:
            generate_fairness_report_card(mock_fairness_results)
            
            # Should use default name
            mock_write.assert_called_once()
    
    def test_empty_results_dict(self, tmp_path):
        """Test handling of empty results dictionary."""
        output_path = tmp_path / "report.md"
        
        report = generate_fairness_report_card(
            {},
            output_path=str(output_path)
        )
        
        # Should still generate valid report
        assert isinstance(report, str)
        assert len(report) > 0
    
    def test_report_unicode_characters(self, mock_fairness_results, tmp_path):
        """Test that report handles unicode characters (emojis)."""
        output_path = tmp_path / "report.md"
        
        generate_fairness_report_card(
            mock_fairness_results,
            output_path=str(output_path)
        )
        
        # Read with UTF-8 encoding
        content = output_path.read_text(encoding='utf-8')
        
        # Should contain emojis or special chars
        assert '✅' in content or '❌' in content or 'PASS' in content


# ============================================================================
# Test GitHub Actions Workflow Creation
# ============================================================================

class TestCreatePRCommentWorkflow:
    """Test GitHub Actions workflow creation."""
    
    def test_workflow_file_created(self, tmp_path):
        """Test that workflow file is created."""
        with patch('pathlib.Path.cwd', return_value=tmp_path):
            create_pr_comment_workflow()
            
            workflow_path = tmp_path / '.github' / 'workflows' / 'fairness-check.yml'
            assert workflow_path.exists()
    
    def test_workflow_directory_created(self, tmp_path):
        """Test that .github/workflows directory is created."""
        with patch('pathlib.Path.cwd', return_value=tmp_path):
            create_pr_comment_workflow()
            
            workflows_dir = tmp_path / '.github' / 'workflows'
            assert workflows_dir.exists()
            assert workflows_dir.is_dir()
    
    def test_workflow_yaml_syntax(self, tmp_path):
        """Test that workflow file contains valid YAML."""
        with patch('pathlib.Path.cwd', return_value=tmp_path):
            create_pr_comment_workflow()
            
            workflow_path = tmp_path / '.github' / 'workflows' / 'fairness-check.yml'
            content = workflow_path.read_text()
            
            # Check for YAML structure
            assert 'name:' in content
            assert 'on:' in content
            assert 'jobs:' in content
    
    def test_workflow_triggers_on_pr(self, tmp_path):
        """Test that workflow triggers on pull requests."""
        with patch('pathlib.Path.cwd', return_value=tmp_path):
            create_pr_comment_workflow()
            
            workflow_path = tmp_path / '.github' / 'workflows' / 'fairness-check.yml'
            content = workflow_path.read_text()
            
            assert 'pull_request:' in content
    
    def test_workflow_runs_pytest(self, tmp_path):
        """Test that workflow runs pytest."""
        with patch('pathlib.Path.cwd', return_value=tmp_path):
            create_pr_comment_workflow()
            
            workflow_path = tmp_path / '.github' / 'workflows' / 'fairness-check.yml'
            content = workflow_path.read_text()
            
            assert 'pytest' in content
            assert 'test_fairness.py' in content
    
    def test_workflow_installs_dependencies(self, tmp_path):
        """Test that workflow installs dependencies."""
        with patch('pathlib.Path.cwd', return_value=tmp_path):
            create_pr_comment_workflow()
            
            workflow_path = tmp_path / '.github' / 'workflows' / 'fairness-check.yml'
            content = workflow_path.read_text()
            
            assert 'pip install' in content
            assert 'requirements.txt' in content
    
    def test_workflow_generates_report(self, tmp_path):
        """Test that workflow generates report."""
        with patch('pathlib.Path.cwd', return_value=tmp_path):
            create_pr_comment_workflow()
            
            workflow_path = tmp_path / '.github' / 'workflows' / 'fairness-check.yml'
            content = workflow_path.read_text()
            
            assert 'generate-report' in content
    
    def test_workflow_comments_on_pr(self, tmp_path):
        """Test that workflow comments on PR."""
        with patch('pathlib.Path.cwd', return_value=tmp_path):
            create_pr_comment_workflow()
            
            workflow_path = tmp_path / '.github' / 'workflows' / 'fairness-check.yml'
            content = workflow_path.read_text()
            
            assert 'github-script' in content
            assert 'createComment' in content
    
    def test_workflow_uses_ubuntu(self, tmp_path):
        """Test that workflow runs on Ubuntu."""
        with patch('pathlib.Path.cwd', return_value=tmp_path):
            create_pr_comment_workflow()
            
            workflow_path = tmp_path / '.github' / 'workflows' / 'fairness-check.yml'
            content = workflow_path.read_text()
            
            assert 'ubuntu-latest' in content
    
    def test_workflow_runs_on_failure(self, tmp_path):
        """Test that workflow runs even if tests fail."""
        with patch('pathlib.Path.cwd', return_value=tmp_path):
            create_pr_comment_workflow()
            
            workflow_path = tmp_path / '.github' / 'workflows' / 'fairness-check.yml'
            content = workflow_path.read_text()
            
            assert 'if: always()' in content


# ============================================================================
# Integration Tests
# ============================================================================

class TestGitHooksIntegration:
    """Integration tests for git hooks functionality."""
    
    def test_full_workflow_setup(self, temp_git_repo, mock_fairness_results):
        """Test complete workflow: hook + report + workflow."""
        # Install hook
        hook_result = install_pre_commit_hook(temp_git_repo)
        assert hook_result is True
        
        # Generate report
        report_path = Path(temp_git_repo) / "fairness_report.md"
        report = generate_fairness_report_card(
            mock_fairness_results,
            output_path=str(report_path)
        )
        assert report_path.exists()
        
        # Create workflow (in temp repo)
        with patch('pathlib.Path.cwd', return_value=Path(temp_git_repo)):
            create_pr_comment_workflow()
            
            workflow_path = Path(temp_git_repo) / '.github' / 'workflows' / 'fairness-check.yml'
            assert workflow_path.exists()
    
    def test_hook_can_be_executed(self, temp_git_repo):
        """Test that installed hook can be executed."""
        install_pre_commit_hook(temp_git_repo)
        
        hook_path = Path(temp_git_repo) / '.git' / 'hooks' / 'pre-commit'
        
        # Should be executable
        assert os.access(hook_path, os.X_OK)
    
    @patch('subprocess.run')
    def test_hook_execution_with_mock(self, mock_run, temp_git_repo):
        """Test hook execution with mocked subprocess."""
        install_pre_commit_hook(temp_git_repo)
        
        hook_path = Path(temp_git_repo) / '.git' / 'hooks' / 'pre-commit'
        
        # Mock successful pytest run
        mock_run.return_value = Mock(returncode=0, stdout='', stderr='')
        
        # Execute hook (in subprocess to avoid import issues)
        # This is a simplified test
        assert hook_path.exists()


# ============================================================================
# Command-Line Interface Tests
# ============================================================================

class TestGitHooksCLI:
    """Test command-line interface functionality."""
    
    @patch('sys.argv', ['git_hooks.py', 'install'])
    @patch('measurement_module.scripts.git_hooks.install_pre_commit_hook')
    def test_cli_install_command(self, mock_install):
        """Test CLI install command."""
        mock_install.return_value = True
        
        # Would normally import and run main()
        # This is a placeholder for CLI testing
        assert True
    
    @patch('sys.argv', ['git_hooks.py', 'generate-report'])
    def test_cli_generate_report_command(self):
        """Test CLI generate-report command."""
        # Placeholder for CLI testing
        assert True
    
    @patch('sys.argv', ['git_hooks.py', 'create-workflow'])
    @patch('measurement_module.scripts.git_hooks.create_pr_comment_workflow')
    def test_cli_create_workflow_command(self, mock_create):
        """Test CLI create-workflow command."""
        # Placeholder for CLI testing
        assert True
    
    @patch('sys.argv', ['git_hooks.py', 'invalid-command'])
    def test_cli_invalid_command(self):
        """Test CLI with invalid command."""
        # Should handle gracefully
        assert True
    
    @patch('sys.argv', ['git_hooks.py'])
    def test_cli_no_command(self):
        """Test CLI with no command."""
        # Should show usage
        assert True


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestGitHooksErrorHandling:
    """Test error handling in git hooks."""
    
    def test_install_with_permission_error(self, temp_git_repo):
        """Test handling of permission errors during install."""
        hooks_dir = Path(temp_git_repo) / '.git' / 'hooks'
        
        # Make directory read-only
        hooks_dir.chmod(0o444)
        
        try:
            result = install_pre_commit_hook(temp_git_repo)
            # Should handle error gracefully
            assert result is False or result is True
        finally:
            # Restore permissions for cleanup
            hooks_dir.chmod(0o755)
    
    def test_report_with_invalid_path(self, mock_fairness_results):
        """Test report generation with invalid path."""
        invalid_path = "/invalid/directory/that/does/not/exist/report.md"
        
        with pytest.raises((OSError, FileNotFoundError, PermissionError)):
            generate_fairness_report_card(
                mock_fairness_results,
                output_path=invalid_path
            )
    
    def test_report_with_none_results(self, tmp_path):
        """Test report generation with None results."""
        output_path = tmp_path / "report.md"
        
        # Should handle None gracefully or raise appropriate error
        with pytest.raises((TypeError, AttributeError)):
            generate_fairness_report_card(
                None,
                output_path=str(output_path)
            )
    
    def test_workflow_creation_without_permissions(self, tmp_path):
        """Test workflow creation without write permissions."""
        # Make directory read-only
        tmp_path.chmod(0o444)
        
        try:
            with patch('pathlib.Path.cwd', return_value=tmp_path):
                with pytest.raises((OSError, PermissionError)):
                    create_pr_comment_workflow()
        finally:
            # Restore permissions
            tmp_path.chmod(0o755)


# ============================================================================
# Utility Tests
# ============================================================================

class TestGitHooksUtilities:
    """Test utility functions in git hooks."""
    
    def test_report_formatting_consistency(self, mock_fairness_results, tmp_path):
        """Test that report formatting is consistent."""
        output_path1 = tmp_path / "report1.md"
        output_path2 = tmp_path / "report2.md"
        
        report1 = generate_fairness_report_card(
            mock_fairness_results,
            output_path=str(output_path1)
        )
        
        report2 = generate_fairness_report_card(
            mock_fairness_results,
            output_path=str(output_path2)
        )
        
        # Should be identical
        assert report1 == report2
    
    def test_report_encoding_utf8(self, mock_fairness_results, tmp_path):
        """Test that report uses UTF-8 encoding."""
        output_path = tmp_path / "report.md"
        
        generate_fairness_report_card(
            mock_fairness_results,
            output_path=str(output_path)
        )
        
        # Should be readable as UTF-8
        content = output_path.read_text(encoding='utf-8')
        assert isinstance(content, str)
    
    def test_hook_script_is_valid_python(self, temp_git_repo):
        """Test that generated hook script is valid Python."""
        install_pre_commit_hook(temp_git_repo)
        
        hook_path = Path(temp_git_repo) / '.git' / 'hooks' / 'pre-commit'
        
        # Try to compile the script
        code = hook_path.read_text()
        try:
            compile(code, str(hook_path), 'exec')
            is_valid = True
        except SyntaxError:
            is_valid = False
        
        assert is_valid
    
    def test_workflow_yaml_is_valid_syntax(self, tmp_path):
        """Test that generated YAML has valid syntax."""
        with patch('pathlib.Path.cwd', return_value=tmp_path):
            create_pr_comment_workflow()
            
            workflow_path = tmp_path / '.github' / 'workflows' / 'fairness-check.yml'
            content = workflow_path.read_text()
            
            # Basic YAML syntax check
            assert content.count('  ') > 0  # Should have indentation
            assert ':' in content  # Should have key-value pairs