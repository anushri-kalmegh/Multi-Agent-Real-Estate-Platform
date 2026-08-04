import sqlite3

import agents.app_repository as repository


def isolated_database(tmp_path, monkeypatch):
    database = tmp_path / "app.db"
    monkeypatch.setattr(repository, "APP_DATABASE_PATH", database)
    repository.initialize_app_database()
    return database


def test_registration_authentication_and_session(tmp_path, monkeypatch):
    isolated_database(tmp_path, monkeypatch)
    user = repository.create_user("person@example.com", "Test Person", "secure123")
    assert repository.authenticate("PERSON@example.com", "secure123")["id"] == user["id"]
    assert repository.authenticate("person@example.com", "wrong-password") is None
    token = repository.create_session(user["id"])
    assert repository.session_user(token)["email"] == "person@example.com"
    repository.revoke_session(token)
    assert repository.session_user(token) is None


def test_password_policy_and_duplicate_email(tmp_path, monkeypatch):
    isolated_database(tmp_path, monkeypatch)
    try:
        repository.create_user("person@example.com", "Person", "short")
        raise AssertionError("Weak password was accepted")
    except ValueError:
        pass
    repository.create_user("person@example.com", "Person", "secure123")
    try:
        repository.create_user("PERSON@example.com", "Other", "secure456")
        raise AssertionError("Duplicate email was accepted")
    except ValueError:
        pass


def test_persistent_user_features(tmp_path, monkeypatch):
    database = isolated_database(tmp_path, monkeypatch)
    user = repository.create_user("person@example.com", "Person", "secure123")
    repository.save_message(user["id"], "thread-1", "user", "Find a property")
    repository.save_search(user["id"], "Find a property", {"city": "Pune"})
    assert repository.add_favourite(
        user["id"], {"property_id": "prop-1", "locality": "Baner", "price": 8_000_000}
    )
    assert not repository.add_favourite(
        user["id"], {"property_id": "prop-1", "locality": "Baner", "price": 8_000_000}
    )
    report_id = repository.save_report(
        user["id"], "thread-1", "Find a property", {"status": "ok"}
    )
    assert repository.list_favourites(user["id"])[0]["property_id"] == "prop-1"
    assert repository.list_reports(user["id"])[0]["id"] == report_id
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM saved_searches").fetchone()[0] == 1
    assert repository.clear_conversation_history(user["id"], "thread-1") == 1
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] == 0
