# 020 Frontend Run event stream

## Goal

Connect the existing authenticated Run SSE endpoint to the browser research
desk so a submitted Run changes state in real time and exposes the current
orchestration stage without giving the client authority over Run ownership or
workspace identity.

## Scope

- add an authenticated Fetch-based SSE reader because browser `EventSource`
  cannot send the required Bearer header;
- update the research desk when durable Run events arrive;
- refresh the server-owned Run inspection after the terminal event;
- show a compact live-stage indicator in the recent research panel;
- abort streams when the workspace view is unmounted or changes.

## Out of scope

- evidence rendering and citation inspection;
- reconnect persistence beyond the endpoint's `since` cursor;
- changing the backend worker or SSE contract.

## Acceptance criteria

- a newly submitted Run is monitored through the API event stream;
- queued/running/terminal states come from the server inspection, not client
  guesses;
- the stream carries the existing bearer token and cannot cross workspace
  boundaries;
- frontend production build and repository quality gate pass.

## Status

Implemented locally. Evidence presentation is the next frontend slice.
