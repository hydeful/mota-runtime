"""Original fixed-seed evaluation implementation, rebound to topology modules."""
import sys
import types
from pathlib import Path

SOURCE=Path(__file__).resolve().parent.parent/'learning/evaluation.py'
module=types.ModuleType(__package__+'._evaluation_core');module.__package__=__package__;module.__file__=str(SOURCE)
sys.modules[module.__name__]=module
exec(compile(SOURCE.read_text(),str(SOURCE),'exec'),module.__dict__)
evaluate=module.evaluate
BudgetStop=module.BudgetStop
