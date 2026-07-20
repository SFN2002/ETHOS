from services.db_repository import DBRepository


def test_connection():
    try:
        db = DBRepository()
        print("Connection successful!")

        db.add_citizen(id=1, name="Test Citizen", profession="Tester", religion="None")
        print("Data inserted successfully!")

        db.close()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_connection()
