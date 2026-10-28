import sys
import yaml
from PyQt6.QtWidgets import QApplication
from ui.main_window import ThreatSightWindow

def main():
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    app = QApplication(sys.argv)
    window = ThreatSightWindow(config)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()