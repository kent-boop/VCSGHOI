from src.engine.arg_parser import get_args_parser


def test_default_variant_and_blip_defaults_are_unchanged():
    args = get_args_parser().parse_args([])
    assert args.stip_model_variant == "default"
    assert args.model_variant is None
    assert args.blip_vit == "base"


def test_ossm_box_arguments_are_explicit_and_independent():
    args = get_args_parser().parse_args(
        [
            "--stip_model_variant",
            "ossm_box",
            "--ossm_blip_pretrained",
            "old_blip_l.pth",
            "--ossm_blip_vit",
            "large",
        ]
    )
    assert args.stip_model_variant == "ossm_box"
    assert args.ossm_blip_pretrained == "old_blip_l.pth"
    assert args.ossm_blip_vit == "large"
    assert args.blip_vit == "base"
