# -*- coding: utf-8 -*-
"""The codebook-derived output layout.

The property worth testing is the round trip: a rater's label encoded to a target and decoded
back must be the same label, for every field of every behaviour. That is what makes a model
output and a human label comparable, and it is checked against the real codebook rather than
against a fixture, because a fixture would only prove the code agrees with itself.
"""
from __future__ import annotations

import pytest

from praxis.annotation import CODEBOOK_V1
from praxis.annotation.codebook import ScaleType
from praxis.behaviour.heads import (
    HeadKind,
    behaviour_heads,
    decode_prediction,
    encode_labels,
    output_width,
)
from praxis.vocabulary import BEHAVIOUR_IDS


def every_field():
    for behaviour in BEHAVIOUR_IDS:
        for head in behaviour_heads(CODEBOOK_V1, behaviour):
            yield behaviour, head


@pytest.mark.parametrize("behaviour", BEHAVIOUR_IDS)
def test_every_codebook_field_gets_exactly_one_head(behaviour: str) -> None:
    """Detection refuses a prediction that omits a field, so a missing head is a crash at
    serialisation rather than a smaller model."""
    spec = CODEBOOK_V1.behaviour(behaviour)
    heads = behaviour_heads(CODEBOOK_V1, behaviour)

    assert [head.field for head in heads] == [field.name for field in spec.fields], (
        "order is the codebook's and is load-bearing: it fixes which slice of the output "
        "tensor belongs to which field")


def test_binary_fields_get_one_logit_not_two() -> None:
    presence = next(h for b, h in every_field() if h.field == "b1_present")
    assert presence.kind is HeadKind.BINARY
    assert presence.width == 1


def test_categorical_fields_get_one_logit_per_level() -> None:
    dominant = next(h for b, h in every_field() if h.field == "b2_dominant")
    assert dominant.kind is HeadKind.CATEGORICAL
    assert dominant.width == 4
    assert dominant.levels == ("class", "board", "materials", "away")


def test_ordinal_interval_and_ratio_all_become_scalars() -> None:
    kinds = {head.field: head.kind for _, head in every_field()
             if head.scale in (ScaleType.ORDINAL, ScaleType.INTERVAL, ScaleType.RATIO)}
    assert set(kinds.values()) == {HeadKind.SCALAR}
    assert {"b1_count", "b1_amplitude", "b2_facing_proportion", "b5_duration"} <= set(kinds)


@pytest.mark.parametrize("behaviour", BEHAVIOUR_IDS)
def test_output_width_is_the_sum_of_the_heads(behaviour: str) -> None:
    heads = behaviour_heads(CODEBOOK_V1, behaviour)
    assert output_width(heads) == sum(head.width for head in heads)


class TestRoundTrip:
    """Encode then decode returns the label. Checked on every level of every field."""

    @pytest.mark.parametrize("behaviour", BEHAVIOUR_IDS)
    def test_every_declared_level_survives_the_round_trip(self, behaviour: str) -> None:
        for head in behaviour_heads(CODEBOOK_V1, behaviour):
            if head.levels is None:
                continue
            for level in head.levels:
                encoded = head.encode(level)
                raw = ([1.0 if i == int(encoded) else 0.0 for i in range(head.width)]
                       if head.kind is HeadKind.CATEGORICAL else encoded)
                assert head.decode(raw) == level, f"{head.field} lost {level!r}"

    def test_a_count_survives_every_value_in_its_range(self) -> None:
        count = next(h for _, h in every_field() if h.field == "b1_count")
        for value in range(0, 21):
            assert count.decode(count.encode(value)) == value

    def test_a_proportion_survives_to_floating_point_precision(self) -> None:
        proportion = next(h for _, h in every_field() if h.field == "b2_facing_proportion")
        for value in (0.0, 0.25, 0.5, 0.75, 1.0, 0.333):
            assert proportion.decode(proportion.encode(value)) == pytest.approx(value)

    @pytest.mark.parametrize("behaviour", BEHAVIOUR_IDS)
    def test_a_whole_label_set_round_trips_into_a_valid_prediction(self,
                                                                   behaviour: str) -> None:
        """The end-to-end claim: what comes out is something the codebook accepts."""
        heads = behaviour_heads(CODEBOOK_V1, behaviour)
        labels = {}
        for head in heads:
            if head.kind is HeadKind.CATEGORICAL or head.levels is not None:
                labels[head.field] = head.levels[0]
            else:
                labels[head.field] = head.minimum

        targets = encode_labels(heads, labels)
        outputs = {
            head.field: ([1.0 if i == int(targets[head.field]) else 0.0
                          for i in range(head.width)]
                         if head.kind is HeadKind.CATEGORICAL else targets[head.field])
            for head in heads
        }
        decoded = decode_prediction(heads, outputs)

        assert decoded == labels
        # The real gate: the same validator a rater's label goes through.
        CODEBOOK_V1.validate_labels(behaviour, decoded)


class TestNormalisation:
    """Scalar targets land in [0, 1] so no field dominates the loss by its units."""

    def test_every_scalar_target_is_in_the_unit_interval(self) -> None:
        for _, head in every_field():
            if head.kind is not HeadKind.SCALAR:
                continue
            values = head.levels if head.levels else (head.minimum, head.maximum)
            for value in values:
                assert 0.0 <= head.encode(value) <= 1.0, head.field

    def test_a_count_and_a_proportion_reach_the_loss_on_the_same_scale(self) -> None:
        """b1_count spans 0 to 20 and b2_facing_proportion spans 0 to 1. Unnormalised, the
        count would weigh roughly four hundred times more in a summed squared error."""
        count = next(h for _, h in every_field() if h.field == "b1_count")
        proportion = next(h for _, h in every_field() if h.field == "b2_facing_proportion")

        assert count.encode(20) == proportion.encode(1.0) == 1.0
        assert count.encode(0) == proportion.encode(0.0) == 0.0

    def test_ordinal_levels_are_encoded_by_position_not_face_value(self) -> None:
        """b5_duration runs 0 to 3 and b1_amplitude runs 1 to 3. Encoding the number itself
        would put their first bands at different points on the scale."""
        amplitude = next(h for _, h in every_field() if h.field == "b1_amplitude")
        duration = next(h for _, h in every_field() if h.field == "b5_duration")

        assert amplitude.encode(1) == duration.encode(0) == 0.0
        assert amplitude.encode(3) == duration.encode(3) == 1.0


class TestClipping:
    """A linear head emits whatever it emits. The codebook domain is not optional."""

    def test_an_out_of_range_count_is_clipped_not_passed_on(self) -> None:
        count = next(h for _, h in every_field() if h.field == "b1_count")
        assert count.decode(-0.4) == 0
        assert count.decode(1.9) == 20

    def test_an_out_of_range_ordinal_is_clipped_to_a_declared_level(self) -> None:
        amplitude = next(h for _, h in every_field() if h.field == "b1_amplitude")
        assert amplitude.decode(-2.0) == 1
        assert amplitude.decode(9.0) == 3

    def test_a_clipped_prediction_is_still_one_the_codebook_accepts(self) -> None:
        heads = behaviour_heads(CODEBOOK_V1, "B1")
        outputs = {"b1_present": 5.0, "b1_count": -3.0,
                   "b1_amplitude": 7.0, "b1_nonscorable": -1.0}
        CODEBOOK_V1.validate_labels("B1", decode_prediction(heads, outputs))


class TestNonScorableFlag:
    """The one head the non-scorable mask must not remove."""

    def test_the_nonscorable_field_is_identified_on_every_behaviour(self) -> None:
        for behaviour in BEHAVIOUR_IDS:
            flags = [h for h in behaviour_heads(CODEBOOK_V1, behaviour)
                     if h.is_nonscorable_flag]
            assert len(flags) == 1, f"{behaviour} needs exactly one non-scorable flag"
            assert flags[0].kind is HeadKind.BINARY

    def test_no_other_field_is_mistaken_for_it(self) -> None:
        for _, head in every_field():
            assert head.is_nonscorable_flag == head.field.endswith("_nonscorable")


def test_encoding_refuses_an_incomplete_label_set() -> None:
    """A missing field would train its head on whatever a default happened to be."""
    heads = behaviour_heads(CODEBOOK_V1, "B1")
    with pytest.raises(KeyError, match="b1_amplitude"):
        encode_labels(heads, {"b1_present": True, "b1_count": 2, "b1_nonscorable": False})


def test_encoding_refuses_a_level_the_codebook_does_not_define() -> None:
    dominant = next(h for _, h in every_field() if h.field == "b2_dominant")
    with pytest.raises(ValueError, match="is not one of"):
        dominant.encode("staffroom")
