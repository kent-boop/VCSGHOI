from pathlib import Path

from PIL import Image

from tools.evidencehoi_image_search_v5.feature_views import (
    Box,
    ContextRole,
    compute_geometry,
    context_visible_pair_only,
    generate_views,
    intersection_area,
    outside_fraction,
    true_pair_occupancy_area,
    union_rect,
)


def test_target_occupancy_uses_true_box_union_not_enclosing_rect() -> None:
    human = Box(0, 0, 10, 10)
    obj = Box(20, 20, 30, 30)
    crop = Box(0, 0, 40, 40)
    geom = compute_geometry(human, obj, crop, 200)
    assert true_pair_occupancy_area(human, obj) == 200
    assert union_rect([human, obj]).area == 900
    assert geom.target_pair_occupancy == 200 / 1600


def test_pair_context_visibility_rejects_target_endpoint_context() -> None:
    human = Box(0, 0, 10, 20)
    obj = Box(20, 0, 30, 20)
    context = Box(21, 1, 29, 19)
    assert context_visible_pair_only(context, human, obj)
    external = Box(40, 0, 50, 10)
    assert not context_visible_pair_only(external, human, obj)


def test_outside_fraction_with_dilated_pair_region() -> None:
    pair_region = Box(0, 0, 10, 10)
    context = Box(5, 5, 15, 15)
    assert outside_fraction(context, pair_region) == 0.75
    assert outside_fraction(Box(20, 20, 30, 30), pair_region) == 1.0


def test_generated_views_share_canvas_and_mask_without_pasting(tmp_path: Path) -> None:
    image_path = tmp_path / "synthetic.jpg"
    img = Image.new("RGB", (100, 80), (10, 20, 30))
    img.save(image_path)
    human = Box(10, 10, 30, 60)
    obj = Box(35, 20, 55, 50)
    cpos = ContextRole("cpos", "cpos1", "knife", Box(70, 10, 85, 25), "[GT]/test", "cut", "h->knife", "test")
    cneg = ContextRole("cneg", "cneg1", "cup", Box(70, 45, 90, 65), "[GT]/test", "hold", "h2->cup", "test")
    crop = Box(5, 5, 95, 70)
    geom = compute_geometry(human, obj, crop, 180)
    paths = generate_views(image_path, tmp_path / "views", human, obj, cpos, cneg, geom)
    sizes = []
    for key in ["pair_only_exact_path", "union_only_exact_path", "full_evidence_crop_path", "full_minus_cpos_path", "role_overlay_path"]:
        with Image.open(Path("/wzj/zijian/OSSM-HOI-main") / paths[key]) if paths[key].startswith("outputs/") else Image.open(tmp_path / "views" / Path(paths[key]).name) as view:
            sizes.append(view.size)
    assert len(set(sizes)) == 1
    assert cpos.context_id != cneg.context_id
    assert intersection_area(cpos.box, union_rect([human, obj])) == 0
    assert intersection_area(cneg.box, union_rect([human, obj])) == 0
