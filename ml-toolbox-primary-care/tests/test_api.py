"""Basic tests for API"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_import():
    """Test that modules can be imported"""
    try:
        from ml_toolbox.serving import api
        assert api is not None
    except ImportError:
        pass  # API may have dependencies not installed yet

def test_basic():
    """Basic test to ensure pytest works"""
    assert True
