import importlib.util
from pathlib import Path
import urllib.request

spec=importlib.util.spec_from_file_location('publish_release',Path(__file__).parents[1]/'scripts/publish_release.py')
release=importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def test_release_uses_exact_successful_main_push():
    base={'head_sha':'abc','event':'push','head_branch':'main','conclusion':'success'}
    assert release.select_run([dict(base,event='pull_request')],'abc') is None
    assert release.select_run([dict(base,head_branch='feature')],'abc') is None
    assert release.select_run([dict(base,conclusion='failure')],'abc') is None
    assert release.select_run([base],'other') is None
    assert release.select_run([base],'abc')==base


def test_redirect_drops_token_on_other_host():
    request=urllib.request.Request('https://api.github.com/archive',headers={'Authorization':'Bearer synthetic'})
    redirected=release.SafeRedirect().redirect_request(request,None,302,'',{},'https://example.com/file.zip')
    assert not redirected.has_header('Authorization')


def test_release_version_is_read_without_importing_native_code():
    from meshrepair import __version__
    assert release.version_from_file(Path(__file__).parents[1]/'meshrepair/__init__.py')==__version__
