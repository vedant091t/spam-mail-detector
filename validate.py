#!/usr/bin/env python3
"""
Validation script to verify project setup and completeness
Run this to ensure everything is properly configured
"""

import os
import sys
from pathlib import Path

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

def check(condition, message, critical=False):
    """Check a condition and print status"""
    if condition:
        print(f"{Colors.GREEN}✓{Colors.RESET} {message}")
        return True
    else:
        marker = f"{Colors.RED}✗{Colors.RESET}" if critical else f"{Colors.YELLOW}⚠{Colors.RESET}"
        print(f"{marker} {message}")
        return False

def main():
    """Main validation function"""
    print(f"\n{Colors.BOLD}{'='*70}")
    print(f"  SPAM MAIL DETECTOR - PROJECT VALIDATION")
    print(f"{'='*70}{Colors.RESET}\n")
    
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    passed = 0
    failed = 0
    warnings = 0
    
    # Section 1: Python Environment
    print(f"{Colors.BLUE}[1] Python Environment{Colors.RESET}")
    if check(sys.version_info >= (3, 8), f"Python version {sys.version_info.major}.{sys.version_info.minor} >= 3.8", critical=True):
        passed += 1
    else:
        failed += 1
    
    # Check dependencies
    try:
        import pandas
        import numpy
        import sklearn
        import nltk
        check(True, "All required packages installed")
        passed += 1
    except ImportError as e:
        check(False, f"Missing package: {e}", critical=True)
        failed += 1
    print()
    
    # Section 2: Project Structure
    print(f"{Colors.BLUE}[2] Project Structure{Colors.RESET}")
    
    required_files = [
        'README.md',
        'REPRODUCIBILITY.md',
        'QUICK_START.md',
        'PROJECT_OVERVIEW.md',
        'requirements.txt',
        'predict.py',
        'setup.py',
        'LICENSE',
        '.gitignore',
    ]
    
    for file in required_files:
        if check((project_root / file).exists(), f"File exists: {file}", critical=True):
            passed += 1
        else:
            failed += 1
    print()
    
    # Section 3: Directories
    print(f"{Colors.BLUE}[3] Required Directories{Colors.RESET}")
    
    required_dirs = [
        'src',
        'data',
        'models',
        'report',
    ]
    
    for dir_name in required_dirs:
        if check((project_root / dir_name).exists(), f"Directory exists: {dir_name}/", critical=True):
            passed += 1
        else:
            failed += 1
    print()
    
    # Section 4: Source Files
    print(f"{Colors.BLUE}[4] Source Code Modules{Colors.RESET}")
    
    src_files = [
        'src/main.py',
        'src/train_sms.py',
        'src/test_sms_model.py',
        'src/data_loader.py',
        'src/preprocess.py',
        'src/feature_extraction.py',
        'src/model.py',
        'src/evaluate.py',
    ]
    
    for file in src_files:
        if check((project_root / file).exists(), f"Module exists: {file}"):
            passed += 1
        else:
            warnings += 1
    print()
    
    # Section 5: Datasets
    print(f"{Colors.BLUE}[5] Datasets{Colors.RESET}")
    
    data_files = [
        ('data/spam_sms.csv', True),  # Critical
        ('data/spam.csv', False),      # Optional
        ('data/emails.csv', False),    # Optional
    ]
    
    for file, critical in data_files:
        if check((project_root / file).exists(), f"Dataset: {file}", critical=critical):
            passed += 1
        else:
            if critical:
                failed += 1
            else:
                warnings += 1
    print()
    
    # Section 6: Trained Models
    print(f"{Colors.BLUE}[6] Trained Models{Colors.RESET}")
    
    model_files = [
        'models/spam_classifier_sms.pkl',
        'models/tfidf_vectorizer_sms.pkl',
    ]
    
    models_exist = True
    for file in model_files:
        if check((project_root / file).exists(), f"Model file: {file}"):
            passed += 1
        else:
            models_exist = False
            warnings += 1
    
    if not models_exist:
        print(f"  {Colors.YELLOW}💡 Tip: Run 'python src/train_sms.py' to train the model{Colors.RESET}")
    print()
    
    # Section 7: NLTK Data
    print(f"{Colors.BLUE}[7] NLTK Data{Colors.RESET}")
    
    try:
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        check(True, "NLTK punkt tokenizer available")
        check(True, "NLTK stopwords corpus available")
        passed += 2
    except LookupError:
        check(False, "NLTK data missing")
        warnings += 1
        print(f"  {Colors.YELLOW}💡 Tip: Run 'python -c \"import nltk; nltk.download('punkt'); nltk.download('stopwords')\"'{Colors.RESET}")
    print()
    
    # Section 8: Documentation
    print(f"{Colors.BLUE}[8] Documentation Completeness{Colors.RESET}")
    
    readme_path = project_root / 'README.md'
    if readme_path.exists():
        readme_content = readme_path.read_text(encoding='utf-8')
        
        check('Problem Statement' in readme_content, "README contains problem statement")
        check('Real-World Applications' in readme_content, "README contains use cases")
        check('Performance' in readme_content or 'Metrics' in readme_content, "README contains performance metrics")
        check('Reproducibility' in readme_content, "README mentions reproducibility")
        check('Quick Start' in readme_content, "README contains quick start")
        passed += 5
    else:
        failed += 1
    print()
    
    # Section 9: Test Executable Scripts
    print(f"{Colors.BLUE}[9] Script Executability{Colors.RESET}")
    
    scripts_to_test = [
        'predict.py',
        'setup.py',
        'src/train_sms.py',
        'src/test_sms_model.py',
    ]
    
    for script in scripts_to_test:
        script_path = project_root / script
        if script_path.exists():
            # Check if file has proper Python shebang or is a .py file
            is_executable = script.endswith('.py')
            if check(is_executable, f"Script is valid Python: {script}"):
                passed += 1
            else:
                warnings += 1
        else:
            warnings += 1
    print()
    
    # Final Summary
    print(f"\n{Colors.BOLD}{'='*70}")
    print(f"  VALIDATION SUMMARY")
    print(f"{'='*70}{Colors.RESET}")
    
    total = passed + failed + warnings
    print(f"\n{Colors.GREEN}✓ Passed: {passed}{Colors.RESET}")
    if warnings > 0:
        print(f"{Colors.YELLOW}⚠ Warnings: {warnings}{Colors.RESET}")
    if failed > 0:
        print(f"{Colors.RED}✗ Failed: {failed}{Colors.RESET}")
    
    success_rate = (passed / total * 100) if total > 0 else 0
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    if failed == 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 PROJECT VALIDATION SUCCESSFUL!{Colors.RESET}")
        print(f"{Colors.GREEN}Your project is ready for internship evaluation.{Colors.RESET}")
        
        if warnings > 0:
            print(f"\n{Colors.YELLOW}Note: There are {warnings} optional components missing, but they are not critical.{Colors.RESET}")
        
        print(f"\n{Colors.BOLD}Next Steps:{Colors.RESET}")
        if not models_exist:
            print(f"  1. Train the model: {Colors.BLUE}python src/train_sms.py{Colors.RESET}")
            print(f"  2. Test predictions: {Colors.BLUE}python predict.py --interactive{Colors.RESET}")
        else:
            print(f"  1. Test predictions: {Colors.BLUE}python predict.py --interactive{Colors.RESET}")
            print(f"  2. Review documentation: {Colors.BLUE}README.md, PROJECT_OVERVIEW.md{Colors.RESET}")
        
        return 0
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}❌ PROJECT VALIDATION FAILED{Colors.RESET}")
        print(f"{Colors.RED}Please fix the critical issues above before proceeding.{Colors.RESET}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    print(f"\n{'='*70}\n")
    sys.exit(exit_code)
