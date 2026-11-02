import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # sports/ (so `edgelab.rules...` and `lab...` import)

from edgelab.cli import main  # noqa: E402

sys.exit(main())
