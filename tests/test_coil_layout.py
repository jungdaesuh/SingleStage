import unittest

from coil_layout import layout_for_nfp


class CoilLayoutTests(unittest.TestCase):
    def test_supported_periodicities_preserve_physical_counts(self):
        expected_roots = {2: (4, 8), 4: (2, 4), 8: (1, 2)}
        for nfp, expected in expected_roots.items():
            layout = layout_for_nfp(nfp)
            self.assertEqual((layout.ntf_roots, layout.wp_ntor), expected)
            self.assertEqual(layout.tf_physical_count, 16)
            self.assertEqual(layout.dipole_physical_count, 352)
            self.assertEqual(layout.total_physical_count, 368)
            layout.validate()

    def test_unsupported_periodicity_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "unsupported Nfp"):
            layout_for_nfp(3)


if __name__ == "__main__":
    unittest.main()
