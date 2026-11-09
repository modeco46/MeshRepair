"""Manual representative benchmark; not a performance promise for damaged models."""
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from meshlib import mrmeshpy as mm, mrmeshnumpy as mn
from meshrepair.engine import Settings, Progress
from meshrepair.orientation import weld, orient_consistently
from meshrepair.reduction import propose_reduction, verify_surface_bound

if __name__ == '__main__':
    p=mm.SphereParams();p.numMeshVertices=360002;p.radius=10
    mesh=mm.makeSphere(p)
    v,f=mn.getNumpyVerts(mesh),mn.getNumpyFaces(mesh.topology).copy()
    f[::17]=f[::17][:,[0,2,1]]
    t=time.perf_counter();v,f,_=weld(v,f);print('weld_seconds',time.perf_counter()-t,flush=True)
    t=time.perf_counter();f,r=orient_consistently(v,f);print('orientation_seconds',time.perf_counter()-t,flush=True)
    t=time.perf_counter();candidate,_=propose_reduction(mesh,mm,Settings(.02,False,1,True),Progress(lambda _:None));print('decimation_seconds',time.perf_counter()-t,'triangles',mesh.topology.numValidFaces(),candidate.topology.numValidFaces(),flush=True)
    t=time.perf_counter();report=verify_surface_bound(mesh,candidate,.02);print('verification_seconds',time.perf_counter()-t,'report',report,flush=True)
