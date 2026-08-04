from __future__ import annotations

import pytest

from swarm_inference.coordinator.event_stream import BoundedRequestEventStream
from swarm_inference.exceptions import BackpressureError
from swarm_inference.protocol.messages import (
    StreamEventType,
    SubmitStreamEvent,
    parse_message,
    serialize_message,
)


@pytest.mark.asyncio
async def test_request_events_are_strictly_ordered_from_zero_and_round_trip() -> None:
    stream = BoundedRequestEventStream(request_id="request", capacity=4)
    stream.publish(StreamEventType.REQUEST_ACCEPTED)
    stream.publish(
        StreamEventType.TOKEN_GENERATED,
        token_position=0,
        token_id=17,
        decoded_text_fragment="hello",
    )
    stream.finish(
        StreamEventType.REQUEST_COMPLETED,
        final_token_ids=[17],
        timing_metrics={"end_to_end_s": 0.1},
    )

    events = [event async for event in stream]

    assert [event.sequence_number for event in events] == [0, 1, 2]
    assert [event.event_type for event in events] == [
        StreamEventType.REQUEST_ACCEPTED,
        StreamEventType.TOKEN_GENERATED,
        StreamEventType.REQUEST_COMPLETED,
    ]
    assert [event.monotonic_timestamp_ns for event in events] == sorted(
        event.monotonic_timestamp_ns for event in events
    )
    assert events[-1].final_token_ids == [17]
    assert parse_message(serialize_message(events[1]), SubmitStreamEvent) == events[1]


@pytest.mark.asyncio
async def test_bounded_event_backpressure_fails_without_unbounded_growth() -> None:
    stream = BoundedRequestEventStream(request_id="request", capacity=1)
    stream.publish(StreamEventType.REQUEST_ACCEPTED)

    with pytest.raises(BackpressureError, match="bounded client event queue"):
        stream.publish(StreamEventType.TOPOLOGY_SELECTED, topology_id="topology")

    assert stream.qsize == 1
    events = [event async for event in stream]
    assert [event.sequence_number for event in events] == [0, 1]
    assert events[-1].event_type == StreamEventType.REQUEST_FAILED
    assert "bounded client event queue" in events[-1].status_detail
    assert stream.backpressure_failures == 1
    assert stream.closed


def test_token_stream_event_requires_position_and_token_id() -> None:
    with pytest.raises(ValueError, match="TOKEN_GENERATED"):
        SubmitStreamEvent(
            event_type=StreamEventType.TOKEN_GENERATED,
            request_id="request",
            sequence_number=0,
            monotonic_timestamp_ns=1,
        )
