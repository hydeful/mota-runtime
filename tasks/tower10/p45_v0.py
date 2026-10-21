"""Unified P45 V0 entry: measured serial default, optional parallel actors.

The frozen p45 core remains unchanged so all earlier probe runs stay restorable.
"""
import sys
from tasks.tower10.p45.cli import main

if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='init' and not any(a=='--mode' or a.startswith('--mode=') for a in sys.argv[2:]):
        sys.argv.extend(['--mode','serial'])
    main()
