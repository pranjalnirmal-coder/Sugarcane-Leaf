import tempfile
import unittest
from pathlib import Path

from src.database import authenticate, create_user, get_history, initialize_database, save_prediction


class DatabaseTests(unittest.TestCase):
    def test_account_and_prediction_history(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "test.sqlite3"
            initialize_database(path)
            user_id = create_user("FarmerOne", "secure-password", path)
            self.assertEqual(authenticate("farmerone", "secure-password", path), user_id)
            self.assertIsNone(authenticate("farmerone", "wrong-password", path))
            save_prediction(user_id, "leaf.jpeg", "Rust", 0.83, False, "model.keras", path)
            history = get_history(user_id, db_path=path)
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["class_name"], "Rust")


if __name__ == "__main__":
    unittest.main()
