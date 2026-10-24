"""pytest wrapper around the smoke test (python -m soundgen.smoke)."""


def test_smoke():
    from soundgen.smoke import main
    main(mm=None)        # the MMAudio part runs only where MMAudio and its weights are available
