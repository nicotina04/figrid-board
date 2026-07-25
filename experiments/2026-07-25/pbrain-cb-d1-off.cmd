@echo off
rem CB-D1 control: same release binary, directional delta disabled.
set "NORU_CODEBOOK_DIRECTIONAL_DELTA=off"
"%~dp0..\..\target\release\pbrain-figrid.exe"
