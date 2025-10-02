# facial recognition

Minimal FastAPI service for face/liveness analysis — Windows-friendly, CPU-first.

## Quick Start (Windows, PowerShell)

```powershell
# 1) Clone (already done) and create a clean venv outside the repo
py -3.11 -m venv D:\venvs\facesvc
D:\venvs\facesvc\Scripts\pip install --upgrade pip
D:\venvs\facesvc\Scripts\pip install -r requirements.txt

# 2) (Optional) local env
Copy-Item .env.example .env -ErrorAction Ignore

# 3) Run
python -m uvicorn main:app --host localhost --port 8000
