#!/usr/bin/env python3
"""Video Captioning System - Project Summary and Validation Script."""

import os
import sys
from pathlib import Path
import subprocess
import importlib.util


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True


def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        "torch", "transformers", "streamlit", "opencv-python", 
        "numpy", "pandas", "tqdm", "omegaconf", "pytest"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    return len(missing_packages) == 0


def check_project_structure():
    """Check if project structure is correct."""
    required_dirs = [
        "src", "configs", "data", "tests", "demo", "assets", 
        "notebooks", ".github/workflows"
    ]
    
    required_files = [
        "README.md", "requirements.txt", "pyproject.toml", 
        "main.py", "setup.sh", ".gitignore", ".pre-commit-config.yaml"
    ]
    
    all_good = True
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/")
            all_good = False
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            all_good = False
    
    return all_good


def check_source_code():
    """Check if source code files exist and are importable."""
    source_files = [
        "src/__init__.py",
        "src/models/__init__.py", 
        "src/data/__init__.py",
        "src/eval/__init__.py",
        "src/training/__init__.py",
        "src/utils/__init__.py",
        "src/cli.py"
    ]
    
    all_good = True
    for file_path in source_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            all_good = False
    
    return all_good


def run_basic_tests():
    """Run basic tests to verify functionality."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print("✅ Tests passed")
            return True
        else:
            print("❌ Tests failed")
            print(result.stdout)
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("⚠️  Tests timed out")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False


def check_demo():
    """Check if demo can be imported."""
    try:
        # Try to import demo components
        sys.path.insert(0, str(Path("demo")))
        import streamlit_app
        print("✅ Demo imports successfully")
        return True
    except Exception as e:
        print(f"❌ Demo import failed: {e}")
        return False


def main():
    """Main validation function."""
    print("🎬 Video Captioning System - Project Validation")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Project Structure", check_project_structure),
        ("Source Code", check_source_code),
        ("Basic Tests", run_basic_tests),
        ("Demo", check_demo),
    ]
    
    results = []
    
    for check_name, check_func in checks:
        print(f"\n📋 {check_name}:")
        print("-" * 20)
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ Error in {check_name}: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! The project is ready to use.")
        print("\nNext steps:")
        print("1. Run: ./setup.sh")
        print("2. Launch demo: streamlit run demo/streamlit_app.py")
        print("3. Train model: python -m src.cli train --num-epochs 5")
    else:
        print(f"\n⚠️  {total - passed} checks failed. Please fix the issues above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
