"""
run_spottr_app.py
Unified Launcher for the Spottr Web Application & ML Face Verification Microservice.
Launches both services and opens the UI in your web browser.
"""
import os
import sys

# Add backend_spottr/backend to sys.path for app module resolution
_backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "backend_spottr", "backend"))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

import time
import webbrowser
import subprocess
import threading

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def start_face_service():
    """Runs the Face Verification microservice on port 8002."""
    print("[1/2] Starting Face Verification Microservice on http://localhost:8002...")
    face_script = os.path.join(os.path.dirname(__file__), "ml_services", "face", "app.py")
    env = os.environ.copy()
    env["PORT"] = "8002"
    env["PYTHONIOENCODING"] = "utf-8"
    subprocess.run([sys.executable, face_script], env=env)

def start_backend_service():
    """Runs the Spottr Backend & Frontend SPA on port 8000."""
    print("[2/2] Starting Spottr Core Backend & UI on http://localhost:8000...")
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "backend_spottr", "backend"))
    os.chdir(backend_dir)
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    
    import uvicorn
    from app.main import app
    uvicorn.run(app, host="0.0.0.0", port=8000)

def main():
    print("========================================================================")
    print("  SPOTTR // AI LOCATION DISCOVERY & BIOMETRIC VERIFICATION")
    print("========================================================================")
    
    # 1. Start Face microservice in a background thread
    face_thread = threading.Thread(target=start_face_service, daemon=True)
    face_thread.start()

    # 2. Wait 3 seconds for face service to boot, then open browser
    def open_browser():
        time.sleep(3)
        print("\nOpening Spottr Web Application in your default browser: http://localhost:8000")
        webbrowser.open("http://localhost:8000")

    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()

    # 3. Start Backend on main thread
    start_backend_service()

if __name__ == "__main__":
    main()
