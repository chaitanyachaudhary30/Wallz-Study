"""Flask application factory and production entry point (P08).

Initializes application blueprints, JSON error handlers, and serves the built
React frontend if present in `web/frontend/dist`.
"""
from __future__ import annotations

import argparse
import os
from flask import Flask, jsonify, send_from_directory

from web.backend.routes_games import games_bp
from web.backend.routes_data import data_bp


def create_app(test_config: dict | None = None) -> Flask:
    """Creates and configures the Flask web application."""
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    if test_config:
        app.config.update(test_config)

    # Register blueprints
    app.register_blueprint(games_bp)
    app.register_blueprint(data_bp)

    # Global JSON error handlers
    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({"error": getattr(e, "description", "Bad Request"), "code": 400}), 400

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": getattr(e, "description", "Not Found"), "code": 404}), 404

    @app.errorhandler(409)
    def conflict(e):
        return jsonify({"error": getattr(e, "description", "Conflict"), "code": 409}), 409

    @app.errorhandler(500)
    def server_error(e):
        return jsonify({"error": getattr(e, "description", "Internal Server Error"), "code": 500}), 500

    # Serve built React frontend if present
    frontend_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve_frontend(path):
        if path.startswith("api/"):
            return jsonify({"error": "Endpoint not found", "code": 404}), 404
        if os.path.exists(os.path.join(frontend_dist, path)) and path != "":
            return send_from_directory(frontend_dist, path)
        if os.path.exists(os.path.join(frontend_dist, "index.html")):
            return send_from_directory(frontend_dist, "index.html")
        return jsonify({
            "message": "Modified Wallz Web API Running",
            "frontend_status": "built dist not found (run npm build in web/frontend)",
            "api_endpoints": [
                "/api/config/defaults",
                "/api/agents",
                "/api/games",
            ],
        })

    return app


def main():
    parser = argparse.ArgumentParser(description="Run Modified Wallz Web Server.")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on (default 5000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default 127.0.0.1)")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")

    args = parser.parse_args()
    app = create_app()
    print(f"Starting Wallz Web Demo server on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
