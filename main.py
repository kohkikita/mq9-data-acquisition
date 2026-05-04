# main.py
from daq_app.gui import App
from daq_app.utils import ensure_dir
from daq_app.config import RUNS_DIR

if __name__ == "__main__":
    ensure_dir(RUNS_DIR)
    app = App()
    app.mainloop()