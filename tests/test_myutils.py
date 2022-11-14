from myutils import run_command


def test_run_command():
    cmd = 'echo "Hello world!"'
    out, err = run_command(cmd)

    assert out == "Hello world!"
    assert err == ""
