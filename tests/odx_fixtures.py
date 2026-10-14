"""Small physical mesh files for adapter component tests."""

from pathlib import Path

from PIL import Image


def make_output(root: Path) -> None:
    mesh = root / "odm_texturing"
    mesh.mkdir(parents=True)
    (mesh / "odm_textured_model_geo.obj").write_text(
        "mtllib odm_textured_model_geo.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\n"
        "vt 0 0\nvt 1 0\nvt 0 1\nusemtl material0\nf 1/1 2/2 3/3\n"
    )
    (mesh / "odm_textured_model_geo.mtl").write_text("newmtl material0\nmap_Kd material0000.jpg\n")
    Image.new("RGB", (8, 8), "red").save(mesh / "material0000.jpg")
    (root / "odm_orthophoto").mkdir()
    (root / "odm_orthophoto/odm_orthophoto.tif").write_bytes(b"fixture-raster")
    (root / "odm_georeferencing").mkdir()
    (root / "odm_georeferencing/coords.txt").write_text("WGS84 UTM 50N\n215613 2519070\n")
