"""Publish the tested main-branch artifact for an existing version tag; no rebuild."""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if urllib.parse.urlsplit(req.full_url).netloc != urllib.parse.urlsplit(newurl).netloc:
            redirected.remove_header('Authorization')
        return redirected


def version_from_file(path):
    for node in ast.parse(Path(path).read_text(encoding='utf-8')).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '__version__' for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('В исходниках отсутствует версия.')


def select_run(runs, sha):
    return next((r for r in runs if r['head_sha'] == sha and r['event'] == 'push'
                 and r['head_branch'] == 'main' and r.get('conclusion') == 'success'), None)


def main():
    repository = os.environ['GITHUB_REPOSITORY']
    sha = os.environ.get('RELEASE_COMMIT') or os.environ['GITHUB_SHA']
    version = version_from_file('meshrepair/__init__.py')
    automatic = os.environ.get('RELEASE_NEW_VERSION') == 'true'
    tag = 'v'+version if automatic else os.environ['GITHUB_REF_NAME']
    if not re.fullmatch(r'v\d+\.\d+\.\d+', tag) or tag != 'v'+version:
        raise ValueError('Тег должен совпадать с версией приложения: v'+version)
    api = 'https://api.github.com/repos/'+repository
    opener = urllib.request.build_opener(SafeRedirect())

    def request(url, data=None, method=None, binary=False, content_type=None, accept=None):
        headers = {'Authorization': 'Bearer '+os.environ['GH_TOKEN'],
                   'Accept': accept or ('application/octet-stream' if binary else 'application/vnd.github+json'), 'X-GitHub-Api-Version': '2022-11-28'}
        if data is not None and not isinstance(data, bytes):
            data = json.dumps(data).encode('utf-8')
            content_type = 'application/json'
        if content_type:
            headers['Content-Type'] = content_type
        with opener.open(urllib.request.Request(url, data=data, headers=headers, method=method), timeout=120) as response:
            body = response.read()
        return body if binary else json.loads(body)

    # Only successful push builds of the exact tagged main commit may publish.
    # Fork PR artifacts never qualify, even if their names match.
    deadline = time.monotonic()+1200
    run = None
    while time.monotonic() < deadline:
        data = request(api+'/actions/workflows/check.yml/runs?event=push&branch=main&head_sha='+sha+'&per_page=100')
        run = select_run(data['workflow_runs'], sha)
        if run:
            break
        matching = [r for r in data['workflow_runs'] if r['head_sha'] == sha]
        if matching and all(r['status'] == 'completed' for r in matching):
            raise RuntimeError('Для этого коммита нет успешной сборки main. Релиз не опубликован.')
        print('Ожидание успешной сборки main для тега '+tag, flush=True)
        time.sleep(15)
    if run is None:
        raise RuntimeError('Не найдена проверенная сборка main для этого тега.')
    create_tag = False
    if automatic:
        try:
            reference = request(api+'/git/ref/tags/'+tag)
            if reference['object']['sha'] != sha:
                print('Версия '+tag+' уже выпущена; новый релиз не нужен.')
                return
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
            create_tag = True
    name = 'MeshRepair-'+version+'-windows-x64'
    artifacts = request(api+'/actions/runs/'+str(run['id'])+'/artifacts')['artifacts']
    artifact = next((a for a in artifacts if a['name'] == name and not a['expired']), None)
    if artifact is None:
        raise RuntimeError('Архив EXE отсутствует или истёк; повторите сборку коммита main.')
    # Actions' archive endpoint requires the JSON media type for its redirect,
    # although the signed download target returns ZIP bytes. Release *assets*
    # instead require application/octet-stream.
    payload = request(artifact['archive_download_url'], binary=True, accept='application/vnd.github+json')
    digest = hashlib.sha256(payload).hexdigest()
    if artifact.get('digest') and artifact['digest'] != 'sha256:'+digest:
        raise RuntimeError('Контрольная сумма архива не совпадает.')
    Path(name+'.zip').write_bytes(payload)
    if create_tag:
        request(api+'/git/refs', {'ref': 'refs/tags/'+tag, 'sha': sha})
    try:
        release = request(api+'/releases/tags/'+urllib.parse.quote(tag, safe=''))
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        release = request(api+'/releases', {'tag_name': tag, 'name': 'MeshRepair '+version,
                          'body': Path('RELEASE_NOTES.md').read_text(encoding='utf-8'), 'draft': True})
    upload = release['upload_url'].split('{', 1)[0]
    assets = {a['name']: a for a in release['assets']}
    for filename, content, mime in [(name+'.zip', payload, 'application/zip'),
            ('SHA256SUMS.txt', (digest+'  '+name+'.zip\n').encode(), 'text/plain')]:
        if filename in assets:
            existing_digest = assets[filename].get('digest')
            if not existing_digest:
                existing_digest = 'sha256:'+hashlib.sha256(request(assets[filename]['url'], binary=True)).hexdigest()
            if existing_digest != 'sha256:'+hashlib.sha256(content).hexdigest():
                raise RuntimeError('Существующий файл релиза отличается: '+filename)
            continue
        request(upload+'?name='+urllib.parse.quote(filename), content, method='POST', content_type=mime)
    if release['draft']:
        release = request(api+'/releases/'+str(release['id']), {'draft': False}, method='PATCH')
    print('Опубликован '+release['html_url'])


if __name__ == '__main__':
    main()
