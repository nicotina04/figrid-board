@echo off
rem CB-D1 candidate: same release binary, directional delta enabled.
set "NORU_CODEBOOK_DIRECTIONAL_DELTA=on"
"%~dp0..\..\target\release\pbrain-figrid.exe"
