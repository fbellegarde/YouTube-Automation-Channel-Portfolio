import pytest
from sqlalchemy import create_engine

def test_db_load():
    engine = create_engine('sqlite:///db/local.db')
    with engine.connect() as conn:
        result = conn.execute("SELECT COUNT(*) FROM shows").fetchone()[0]
    assert result > 0, "DB should have data"

# More tests...