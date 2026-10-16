import unittest

from src.advisory import ADVISORIES, get_advisory, validate_advisories


class AdvisoryTests(unittest.TestCase):
    def test_every_supported_class_has_advice(self):
        validate_advisories(["Healthy", "Mosaic", "RedRot", "Rust", "Yellow"])
        self.assertEqual(set(ADVISORIES), {"Healthy", "Mosaic", "RedRot", "Rust", "Yellow"})

    def test_unknown_class_is_not_silently_mapped(self):
        with self.assertRaises(KeyError):
            get_advisory("Unlisted Disease")

    def test_yellow_label_is_not_overstated(self):
        self.assertIn("does not identify a specific disease", get_advisory("Yellow")["description"])


if __name__ == "__main__":
    unittest.main()
