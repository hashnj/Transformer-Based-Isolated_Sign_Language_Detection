import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sign_language.data.check_landmarks import main


if __name__ == "__main__":
    main()
