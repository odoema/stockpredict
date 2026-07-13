"""
Local development entrypoint.

    python run.py

For production, use gunicorn against app:create_app(), e.g.:

    gunicorn "app:create_app()" --bind 0.0.0.0:$PORT
"""
import os

from app import create_app

app = create_app(os.environ.get("FLASK_ENV", "development"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=app.config["DEBUG"])
