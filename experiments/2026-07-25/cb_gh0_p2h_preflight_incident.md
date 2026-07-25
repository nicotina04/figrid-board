# CB-GH0 P2-H preflight incident

Observed: 2026-07-26 03:36 KST

The first invocation of the committed P2-H harness stopped during source
preflight, before scheduling setup, clock calibration, transition timing, or
search timing. No authoritative JSON report was created.

The canonical manifest path returned by Windows used the verbatim
`\\?\C:\...` form. The harness converted backslashes to slashes but did not
remove the verbatim prefix, so its process-local Git
`safe.directory=//?/C:/...` override did not match the repository path. Git
therefore rejected `rev-parse HEAD` as dubious ownership under the sandbox
account.

The correction normalizes Windows verbatim drive paths to `C:/...` and
verbatim UNC paths to `//server/share/...` before passing the process-local
Git override. A focused unit test invokes `git rev-parse HEAD` through the
same canonical-manifest path. Because the failure occurred before timing and
created no report, the frozen protocol permits a corrected build to be
invoked again.
