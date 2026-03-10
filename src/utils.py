import pathlib
import shutil

def clean_dir(cachedir):
    """Recursively delete the contents of *cachedir*.

    The directory itself is left in place; if it doesn't exist this is a no-op.
    Previously we recursed and called ``rmdir`` which could fail with
    "directory not empty" when there were leftover files (e.g. hidden files
    or race conditions).  Use ``shutil.rmtree`` on subdirectories instead and
    catch errors to avoid the exception during downloads.
    """

    path_ = pathlib.Path(cachedir)
    if not path_.exists():
        return

    # iterate over immediate children and remove them safely
    for child in path_.iterdir():
        try:
            if child.is_dir():
                # rmtree handles nested contents and symlinks gracefully
                shutil.rmtree(child, ignore_errors=True)
            else:
                # files and symlinks
                child.unlink()
        except Exception:
            # if for some reason removal fails, ignore and continue
            # (log message could be added here)
            pass
