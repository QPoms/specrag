# 016 Official 3GPP source adapter

## Objective

Bind the dormant specification-version synchronization flow to the official
3GPP FTP archive without enabling background synchronization or deleting
historical versions.

## Scope

- Discover the newest revision for locally configured specifications from
  `https://www.3gpp.org/ftp/Specs/archive/{series}/{specification}/`.
- Decode the official revision filename into the Release/version identity used
  by the corpus registry.
- Download one ZIP only when `corpus sync --official --apply` is requested.
- Safely extract a DOC/DOCX source, convert it to canonical UTF-8 Markdown,
  and pass the resulting document to the existing build-before-activate flow.
- Retain the ZIP, Word source, Markdown, registry record, and old versions.
- Keep the adapter in `infrastructure`; keep synchronization decisions in
  `retrieval`; expose it only through an explicit CLI flag.

## Non-goals

- No scheduler, automatic polling, third-party mirror, or destructive cleanup.
- No chunk-level diffing or special embedding behavior.
- No requirement to install Pandoc or Python Word-processing packages.

## Verification

Use mocked official directory/download responses to verify revision decoding,
safe extraction, DOCX conversion, historical retention, and CLI source
selection. Run `python scripts/check.py`.
