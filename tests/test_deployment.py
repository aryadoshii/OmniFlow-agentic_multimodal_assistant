"""Tests for Phase 7's single-service deployment wiring: serving the built
frontend from the same FastAPI process as the API, in production only.

create_app() decides this once, at construction time, using the REAL
(process-wide, lru_cache'd) get_settings() -- not a per-request Depends()
override -- exactly like a real deployed container would (APP_ENV=production
is part of its actual startup environment). These tests patch
omniflow.main.get_settings and omniflow.main._FRONTEND_DIST directly to
exercise both branches without needing a real build or a real production
environment.
"""

from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from omniflow.config import Settings
from omniflow.main import create_app


def _settings(app_env: str) -> Settings:
    return Settings(APP_ENV=app_env, LOG_LEVEL="WARNING")


class TestDevelopmentAndTestServeJsonRoot:
    """Baseline: non-production environments must be completely unaffected
    by this feature, regardless of whether a built frontend happens to
    exist on disk (e.g. a stray local `npm run build`)."""

    def test_development_serves_json_root_even_if_frontend_dist_exists(self, tmp_path: Path) -> None:
        built_dist = tmp_path / "dist"
        (built_dist / "assets").mkdir(parents=True)
        (built_dist / "index.html").write_text("<html>fake spa</html>")

        with (
            patch("omniflow.main.get_settings", return_value=_settings("development")),
            patch("omniflow.main._FRONTEND_DIST", built_dist),
        ):
            app = create_app()
            with TestClient(app) as client:
                response = client.get("/")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")
        assert "OmniFlow" in response.json()["message"]

    def test_production_without_a_built_frontend_falls_back_to_json_root(self, tmp_path: Path) -> None:
        missing_dist = tmp_path / "dist"  # deliberately never created

        with (
            patch("omniflow.main.get_settings", return_value=_settings("production")),
            patch("omniflow.main._FRONTEND_DIST", missing_dist),
        ):
            app = create_app()
            with TestClient(app) as client:
                response = client.get("/")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")


class TestProductionServesBuiltFrontend:
    def test_production_with_a_built_frontend_serves_index_html_at_root(self, tmp_path: Path) -> None:
        built_dist = tmp_path / "dist"
        (built_dist / "assets").mkdir(parents=True)
        (built_dist / "index.html").write_text("<html><body>OmniFlow SPA</body></html>")
        (built_dist / "assets" / "index-abc123.js").write_text("console.log('app');")

        with (
            patch("omniflow.main.get_settings", return_value=_settings("production")),
            patch("omniflow.main._FRONTEND_DIST", built_dist),
        ):
            app = create_app()
            with TestClient(app) as client:
                index_response = client.get("/")
                asset_response = client.get("/assets/index-abc123.js")
                # The real API routes must still work exactly as before --
                # serving the frontend must never shadow them.
                health_response = client.get("/health")

        assert index_response.status_code == 200
        assert "OmniFlow SPA" in index_response.text
        assert index_response.headers["content-type"].startswith("text/html")

        assert asset_response.status_code == 200
        assert "console.log" in asset_response.text

        assert health_response.status_code == 200
        assert health_response.json()["status"] == "healthy"

    def test_production_json_root_route_is_not_also_registered(self, tmp_path: Path) -> None:
        """The JSON discovery response must not coexist with the SPA at
        the same path -- only one or the other is ever registered."""
        built_dist = tmp_path / "dist"
        (built_dist / "assets").mkdir(parents=True)
        (built_dist / "index.html").write_text("<html>spa</html>")

        with (
            patch("omniflow.main.get_settings", return_value=_settings("production")),
            patch("omniflow.main._FRONTEND_DIST", built_dist),
        ):
            app = create_app()
            with TestClient(app) as client:
                response = client.get("/")

        assert "docs_url" not in response.text
