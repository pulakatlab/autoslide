"""
Make the color correction module executable as a script.

Usage:
    python -m autoslide.utils.color_correction process --help
    python -m autoslide.utils.color_correction restore --help
    python -m autoslide.utils.color_correction list-backups --help
"""

from .cli import main

if __name__ == '__main__':
    main()
