# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

def build(args):
    from .detr import build as build_detr
    return build_detr(args)


def build_model(args):
    return build(args)
