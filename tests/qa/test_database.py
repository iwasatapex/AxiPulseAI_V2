from api.database.connection import (
    Base,
    engine
)


def test_database_tables():

    Base.metadata.create_all(
        bind=engine
    )

    tables = (
        Base.metadata.tables.keys()
    )

    assert (
        "decision_history"
        in tables
    )

    assert (
        "prediction_history"
        in tables
    )
