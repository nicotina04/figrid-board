# Changelog

All notable changes to CB2Vec are documented in this file.

## 0.1.0 - 2026-07-26

- Extract the game-independent codebook model and integer access kernel from
  FIGRID.
- Add floating-point and quantized linear/FM scoring.
- Add exact flat and class-base-plus-residual embedding representations.
- Add the canonical `CB2VEC01` artifact and legacy `NORUCBF1` reader.
- Add the preallocated reversible token journal used by make/undo search.
