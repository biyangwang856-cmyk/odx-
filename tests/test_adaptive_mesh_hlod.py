import tempfile
import unittest
from pathlib import Path

import numpy as np
from core.pipelines.adaptive_hlod import (
    extract_world_geometry,
    measured_proxy_error,
    node_matrix,
    sample_inset_vertex_colors,
    sample_texture,
    simplifier_error,
    srgb_to_linear,
    transformed_texcoords,
    write_colored_glb,
)


class IntermediateMeshHlodTest(unittest.TestCase):
    def test_node_matrix_applies_translation_scale_and_rotation(self):
        matrix = node_matrix(
            {
                "translation": [3, 4, 5],
                "scale": [2, 2, 2],
                "rotation": [0, 0, 0, 1],
            }
        )
        transformed = np.asarray([[1, 2, 3]]) @ matrix[:3, :3].T + matrix[:3, 3]
        np.testing.assert_array_equal(transformed, [[5, 8, 11]])

    def test_colored_glb_round_trips_world_geometry(self):
        positions = np.asarray(
            [
                [1000.0, 2000.0, 3000.0],
                [1001.0, 2000.0, 3000.0],
                [1000.0, 2001.0, 3000.0],
            ]
        )
        colors = np.asarray([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
        indices = np.asarray([0, 1, 2], dtype=np.uint32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "triangle.glb"
            radius = write_colored_glb(path, positions, colors, indices)
            round_trip_positions, round_trip_indices = extract_world_geometry(path)
        self.assertAlmostEqual(radius, np.sqrt(2) / 2)
        np.testing.assert_allclose(round_trip_positions, positions)
        np.testing.assert_array_equal(round_trip_indices, indices)

    def test_texture_transform_and_bilinear_sampling(self):
        image = np.asarray(
            [
                [[0, 0, 0], [100, 0, 0]],
                [[0, 100, 0], [100, 100, 0]],
            ],
            dtype=np.uint8,
        )
        texcoords = transformed_texcoords(
            np.asarray([[0.5, 0.5]]),
            {"extensions": {"KHR_texture_transform": {"offset": [0.25, 0.25], "scale": [0.5, 0.5]}}},
        )
        colors = sample_texture(image, texcoords, 33071, 33071)
        np.testing.assert_array_equal(colors, [[50, 50, 0]])

    def test_inset_sampling_avoids_triangle_corner_gutter(self):
        image = np.full((11, 11, 3), [20, 40, 60], dtype=np.uint8)
        image[0, :, :] = 255
        image[-1, :, :] = 255
        image[:, 0, :] = 255
        image[:, -1, :] = 255
        texcoords = np.asarray([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
        indices = np.asarray([0, 1, 2], dtype=np.uint32)
        colors = sample_inset_vertex_colors(
            image,
            texcoords,
            indices,
            0.3,
            33071,
            33071,
        )
        np.testing.assert_array_equal(colors, np.full((3, 3), [20, 40, 60]))

    def test_inset_sampling_averages_shared_vertex_faces(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        image[:, 1, 0] = 100
        texcoords = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
        indices = np.asarray([0, 1, 2, 0, 3, 1], dtype=np.uint32)
        colors = sample_inset_vertex_colors(image, texcoords, indices, 0, 33071, 33071)
        np.testing.assert_array_equal(colors[:, 0], [0, 100, 0, 100])

    def test_srgb_colors_are_converted_to_linear_vertex_values(self):
        linear = srgb_to_linear(np.asarray([[0, 128, 255]], dtype=np.uint8))
        np.testing.assert_allclose(linear, [[0, 0.2158605, 1]], atol=1e-7)

    def test_parses_achieved_simplifier_error(self):
        self.assertEqual(simplifier_error("debug: simplify: 20 -> 10, error: 0.0060."), 0.006)

    def test_identical_proxy_has_zero_measured_error(self):
        positions = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        colors = np.full((3, 3), 255, dtype=np.uint8)
        indices = np.asarray([0, 1, 2], dtype=np.uint32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "triangle.glb"
            write_colored_glb(path, positions, colors, indices)
            report = measured_proxy_error(positions, indices, path, 0.99)
        self.assertAlmostEqual(report["maximumM"], 0.0, places=6)
        self.assertAlmostEqual(report["candidateToSourceAreaRatio"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
