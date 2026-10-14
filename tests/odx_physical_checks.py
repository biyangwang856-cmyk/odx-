"""Run with the pinned ODX image's Python/GDAL: python tests/odx_physical_checks.py."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.validation.odx_output import read_origin_offset, validate_cog
from osgeo import gdal, osr


class PhysicalOutputChecks(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory(prefix="odx-physical-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        gdal.UseExceptions()

    def raster(self, *, cog: bool = True, alpha: int = 255, bigtiff: bool = False) -> Path:
        source = gdal.GetDriverByName("MEM").Create("", 512, 512, 4, gdal.GDT_Byte)
        spatial = osr.SpatialReference()
        spatial.ImportFromEPSG(32650)
        source.SetSpatialRef(spatial)
        source.SetGeoTransform([215613, 0.05, 0, 2519070, 0, -0.05])
        for index, value in enumerate([20, 40, 60, alpha], 1):
            source.GetRasterBand(index).Fill(value)
            source.GetRasterBand(index).SetColorInterpretation(
                [gdal.GCI_RedBand, gdal.GCI_GreenBand, gdal.GCI_BlueBand, gdal.GCI_AlphaBand][index - 1]
            )
        path = self.root / "image.tif"
        options = ["COMPRESS=DEFLATE", "BLOCKSIZE=256", f"BIGTIFF={'YES' if bigtiff else 'NO'}"]
        target = gdal.GetDriverByName("COG" if cog else "GTiff").CreateCopy(
            str(path), source, options=options if cog else []
        )
        target.FlushCache()
        target = source = None
        return path

    def test_real_cog_and_bigtiff(self) -> None:
        for bigtiff in [False, True]:
            path = self.raster(bigtiff=bigtiff)
            result = validate_cog(path, "EPSG:32650")
            self.assertEqual(result["width"], 512)
            self.assertGreater(result["overviews"], 0)
            self.assertEqual(result["transform"], [215613, 0.05, 0, 2519070, 0, -0.05])

    def test_wrong_crs_and_empty_validity_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "CRS"):
            validate_cog(self.raster(), "EPSG:32649")
        with self.assertRaisesRegex(ValueError, "valid pixels"):
            validate_cog(self.raster(alpha=0), "EPSG:32650")

    def test_non_cog_and_corrupt_raster_fail(self) -> None:
        with self.assertRaises(ValueError):
            validate_cog(self.raster(cog=False), "EPSG:32650")
        path = self.root / "corrupt.tif"
        path.write_bytes(b"not a raster")
        with self.assertRaises((ValueError, RuntimeError)):
            validate_cog(path, "EPSG:32650")

    def test_offset_and_coordinate_mismatch(self) -> None:
        path = self.root / "coords.txt"
        path.write_text("WGS84 UTM 50N\n215613 2519070\n")
        self.assertEqual(read_origin_offset(path, "EPSG:32650"), [215613, 2519070, 0])
        with self.assertRaisesRegex(ValueError, "CRS"):
            read_origin_offset(path, "EPSG:32649")


if __name__ == "__main__":
    unittest.main()
