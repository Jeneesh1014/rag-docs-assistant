def test_fastapi_app_importable():
    from app.api import app

    assert app.title == "Ask My Docs"
