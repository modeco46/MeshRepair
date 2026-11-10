import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import pytest
from meshlib import mrmeshpy as mm
from meshrepair import storage
from meshrepair.engine import run, Settings


@pytest.fixture
def repaired(tmp_path):
    inputs=tmp_path/'input';inputs.mkdir()
    source=inputs/'source.stl';mm.saveMesh(mm.makeCube(),source)
    job=storage.create_job(tmp_path/'temp')
    report=run(source,settings=Settings(),work_dir=job)
    yield report
    storage.remove_job(job)


@pytest.mark.parametrize('selected', [dict(stl=True),dict(json=True),dict(deviation=True,log=True),
                                    dict(stl=True,json=True,deviation=True,log=True)])
def test_only_selected_files_saved_atomically(repaired,tmp_path,selected):
    target=tmp_path/'saved.stl'
    saved,sizes=storage.save_artifacts(repaired,target,selected,'Журнал ремонта')
    assert set(sizes)=={str(p) for p in storage.save_paths(target,selected).values()}
    assert not list(tmp_path.glob('*.part'))
    assert Path(repaired['output']).is_file()
    assert list(Path(repaired['source']).parent.iterdir())==[Path(repaired['source'])]
    if selected.get('json'):
        r=json.loads(Path(str(target)+'.json').read_text(encoding='utf-8'))
        assert r['output']==str(target) and r['saved_at']
    # Can save repeatedly; original native artifacts are never moved away.
    storage.save_artifacts(repaired,tmp_path/'again.stl',selected)


def test_overwrite_requires_explicit_permission_and_source_is_protected(repaired,tmp_path):
    target=tmp_path/'saved.stl';target.write_bytes(b'previous')
    with pytest.raises(FileExistsError): storage.save_artifacts(repaired,target,{'stl':True})
    assert target.read_bytes()==b'previous'
    storage.save_artifacts(repaired,target,{'stl':True},overwrite=True)
    assert target.read_bytes()==Path(repaired['output']).read_bytes()
    with pytest.raises(ValueError,match='исходник'):
        storage.save_artifacts(repaired,repaired['source'],{'stl':True},overwrite=True)
    with pytest.raises(ValueError):
        storage.save_artifacts(repaired,repaired['output'],{'stl':True},overwrite=True)


def test_failed_atomic_replace_removes_part_and_preserves_target(repaired,tmp_path,monkeypatch):
    target=tmp_path/'saved.stl';target.write_bytes(b'previous')
    def fail(*args):raise OSError('disk error')
    monkeypatch.setattr(storage.os,'replace',fail)
    with pytest.raises(OSError):storage.save_artifacts(repaired,target,{'stl':True},overwrite=True)
    assert target.read_bytes()==b'previous' and not Path(str(target)+'.part').exists()


def test_existing_part_file_is_not_deleted(repaired,tmp_path):
    target=tmp_path/'saved.stl';part=Path(str(target)+'.part');part.write_bytes(b'another writer')
    with pytest.raises(FileExistsError):storage.save_artifacts(repaired,target,{'stl':True})
    assert part.read_bytes()==b'another writer'


def test_cleanup_only_old_dead_jobs(tmp_path):
    child=subprocess.Popen([sys.executable,'-c','pass']);child.wait()
    now=time.time()
    def job(name,pid,old):
        path=tmp_path/name;path.mkdir();(path/'owner.pid').write_text(str(pid))
        t=now-90000 if old else now;os.utime(path,(t,t));return path
    old=job('meshrepair-job-old',child.pid,True)
    live=job('meshrepair-job-live',os.getpid(),True)
    young=job('meshrepair-job-young',child.pid,False)
    unknown=job('meshrepair-job-unknown',-1,True)
    assert storage.cleanup_stale(tmp_path,now=now)==[str(old)]
    assert all(p.exists() for p in (live,young,unknown))
    assert storage.pid_alive(os.getpid()) and not storage.pid_alive(child.pid)


def test_legacy_cleanup_and_space_check(tmp_path,monkeypatch):
    old=tmp_path/'.meshrepair-old';old.mkdir();os.utime(old,(0,0))
    recent=tmp_path/'.meshrepair-recent';recent.mkdir()
    storage.cleanup_stale(tmp_path,legacy=True)
    assert not old.exists() and recent.exists()
    monkeypatch.setattr(storage.shutil,'disk_usage',lambda p:SimpleNamespace(free=99))
    with pytest.raises(OSError,match='Недостаточно места'):storage.create_job(tmp_path,triangles=1)
    assert not list(tmp_path.glob('meshrepair-job-*'))


def test_atexit_removes_owned_directory(tmp_path):
    code='from meshrepair.storage import create_job; import sys; print(create_job(sys.argv[1]),flush=True)'
    result=subprocess.run([sys.executable,'-c',code,str(tmp_path)],capture_output=True,text=True,check=True)
    assert not Path(result.stdout.strip()).exists()


def test_cli_output_retains_old_publication_contract(tmp_path):
    source=tmp_path/'source.stl';mm.saveMesh(mm.makeCube(),source)
    target=tmp_path/'result.stl'
    result=subprocess.run([sys.executable,'-m','meshrepair.worker',str(source),'--output',str(target),'--no-cuda',
                           '--temp-directory',str(tmp_path/'temp')],capture_output=True,text=True,encoding='utf-8',timeout=30)
    assert result.returncode==0,result.stderr+result.stdout
    assert all(Path(str(target)+suffix).is_file() for suffix in ('','.json','.deviation.npz'))
    assert not list((tmp_path/'temp').iterdir())
