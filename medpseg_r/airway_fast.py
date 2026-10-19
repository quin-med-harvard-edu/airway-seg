"""Airway-only forward pass for TRAINING and VALIDATION of MEDPSeg-R.

The released network computes three output groups on every call: the polymorphic branch
(BiFPN + attention + `main` head with 4 deep-supervision heads), and the circulatory branch
(BiFPN + attention + `atm` and `vessel` heads, each with 4 deep-supervision heads).

With airway-only data, the loss uses only `atm`/`atm1..4`
(`PolySeg2DModule.atm_branch`; the other format indexes are empty, so their losses are never
evaluated). The airway outputs depend only on the backbone features and the circulatory
BiFPN/attention (`EfficientDetForSemanticSegmentation.build_return_dict` passes
`backbone_features` to `CirculatoryBranch`; nothing from the polymorphic branch enters it),
and the skipped modules contain no random operations. Skipping them therefore leaves the
airway outputs, the loss and every gradient unchanged; it only removes forward compute
whose results were discarded. `tests/test_airway_fast.py` checks this numerically.

The checkpoint keeps the complete released module set (skipped modules stay at their
initial weights), so inference uses the unchanged released pipeline and full forward.
"""
import types


def _circulatory_airway_only(cb, backbone_features):
    """`CirculatoryBranch.forward` (edet/efficientdet/model.py:514-553) without the vessel heads."""
    feat_map = cb.bifpn(backbone_features)
    if cb.attention:
        assert len(feat_map) == len(cb.attention_modules)
        feat_map = [m(x) for x, m in zip(feat_map, cb.attention_modules)]
    if cb.feature_adapters is not None:
        raise RuntimeError("feature adapters are not used by the released configuration")
    if cb.deep_supervision:
        new_feat_map = []
        for feat in feat_map:
            if cb.expand_bifpn is not None and cb.expand_bifpn != True and cb.expand_bifpn != "no":  # noqa: E712
                new_feat_map.append(cb.expand_conv(feat))
        feat_map = new_feat_map
        out = {"atm": cb.airway_head(feat_map[0], return_tensor=True)}
        for i in range(1, 5):
            out[f"atm{i}"] = cb.ds_airway_heads[i - 1](feat_map[i], return_tensor=True)
    else:
        feat_map = feat_map[0]
        if cb.expand_bifpn is not None and cb.expand_bifpn != True and cb.expand_bifpn != "no":  # noqa: E712
            feat_map = cb.expand_conv(feat_map)
        out = {"atm": cb.airway_head(feat_map, return_tensor=True)}
    return out


def _airway_only_forward(self, inputs):
    """`EfficientDetForSemanticSegmentation.forward` restricted to the airway outputs."""
    features = self.extract_backbone_features(inputs)
    return _circulatory_airway_only(self.circulatory_classifier, features)


def enable_airway_only_forward(poly_module):
    """Patch the given PolySeg2DModule instance (not the class, not the installed package)."""
    det = poly_module.model.model          # PolySeg2DModule.model -> MEDSeg; MEDSeg.model -> EfficientDet...
    assert det.circulatory_classifier is not None
    det.forward = types.MethodType(_airway_only_forward, det)
    return poly_module
