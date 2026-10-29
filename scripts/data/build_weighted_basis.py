#!/usr/bin/env python3
"""Build weighted input-PCA or output-POD bases from training shards."""

from __future__ import annotations
import argparse, glob, time
from pathlib import Path
import h5py, numpy as np, torch

def args():
 p=argparse.ArgumentParser(); p.add_argument('--shards',nargs='+',required=True); p.add_argument('--field',choices=('u','p'),required=True); p.add_argument('--samples',type=int,default=10000); p.add_argument('--rank',type=int,default=400); p.add_argument('--oversample',type=int,default=32); p.add_argument('--niter',type=int,default=4); p.add_argument('--seed',type=int,default=20260827); p.add_argument('--device',default='cuda'); p.add_argument('--output',type=Path,required=True); return p.parse_args()
def paths(ps):
 out=[]
 for s in ps:
  m=sorted(glob.glob(s)); out += [Path(x).resolve() for x in m] if m else ([Path(s).resolve()] if Path(s).is_file() else [])
 if not out: raise FileNotFoundError('No shards found')
 return list(dict.fromkeys(out))
def get(h,names):
 for n in names:
  if n in h:return h[n]
 raise KeyError(names)
def ds(h,f): return get(h,('u','input','inputs') if f=='u' else ('p','output','outputs'))
def grid(p):
 with h5py.File(p,'r') as h:return np.asarray(get(h,('x_grid','x'))[:],float).ravel(),np.asarray(get(h,('y_grid','y'))[:],float).ravel()
def sqrtw(x,y):
 dx=np.diff(x);dy=np.diff(y); wx=np.full(x.size,dx[0]);wy=np.full(y.size,dy[0]);wx[[0,-1]]*=.5;wy[[0,-1]]*=.5;return np.sqrt((wx[:,None]*wy[None,:]).ravel())
def main():
 a=args(); ps=paths(a.shards); x,y=grid(ps[0]); shape=(len(x),len(y)); d=np.prod(shape); counts=[]
 for p in ps:
  with h5py.File(p,'r') as h:
   xx=np.asarray(get(h,('x_grid','x'))[:],float).ravel(); yy=np.asarray(get(h,('y_grid','y'))[:],float).ravel(); D=ds(h,a.field)
   if not(np.allclose(xx,x) and np.allclose(yy,y)): raise ValueError(f'grid mismatch {p}')
   counts.append(int(D.shape[0]))
 if not 2<=a.samples<=sum(counts): raise ValueError('bad sample count')
 if not 1<=a.rank<=min(a.samples,d): raise ValueError('bad rank')
 X=np.empty((a.samples,d),np.float32); n=0
 for p,c in zip(ps,counts):
  if n>=a.samples: break
  q=min(c,a.samples-n)
  with h5py.File(p,'r') as h:b=np.asarray(ds(h,a.field)[:q],np.float32).reshape(q,d)
  X[n:n+q]=b;n+=q;print(f'loaded {a.field}: {n}/{a.samples}',flush=True)
 sw=sqrtw(x,y); mean=X.astype(np.float64).mean(0); X-=mean.astype(np.float32);X*=sw.astype(np.float32)
 dev=torch.device(a.device)
 if dev.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
 torch.manual_seed(a.seed); A=torch.from_numpy(X).to(dev); q=min(a.rank+a.oversample,min(A.shape)); t=time.time()
 print(f'weighted {a.field} PCA/POD: A={tuple(A.shape)} rank={a.rank} q={q} niter={a.niter} device={dev}',flush=True)
 U,S,V=torch.pca_lowrank(A,q=q,center=False,niter=a.niter); del U
 V=V[:,:a.rank].contiguous(); S=S[:a.rank].contiguous(); C=A@V
 cm=C.mean(0); cs=C.std(0,unbiased=True).clamp_min(1e-12); total=float(torch.sum(A.double().square()).item()); evr=(S.double().square()/max(total,1e-300)).cpu().numpy(); cum=np.cumsum(evr)
 orth=float(torch.max(torch.abs(V.T@V-torch.eye(a.rank,device=dev))).item())
 o=a.output.expanduser(); o=o if o.is_absolute() else Path.cwd()/o; o=o.resolve();o.parent.mkdir(parents=True,exist_ok=True)
 np.savez(o,mean_field=mean.astype(np.float32),sqrt_weights=sw.astype(np.float32),weighted_modes=V.cpu().numpy().astype(np.float32),coefficient_mean=cm.cpu().numpy().astype(np.float32),coefficient_std=cs.cpu().numpy().astype(np.float32),singular_values=S.cpu().numpy().astype(np.float64),explained_variance_ratio=evr,cumulative_explained_variance_ratio=cum,total_centered_weighted_energy=np.array([total]),x_grid=x,y_grid=y,field=np.array([a.field]),n_samples=np.array([a.samples]),max_rank=np.array([a.rank]),randomized_q=np.array([q]),randomized_niter=np.array([a.niter]),randomized_seed=np.array([a.seed]),weighted_mode_orthogonality_linf=np.array([orth]),source_shards=np.array([str(p) for p in ps]))
 print('saved',o);print('orthogonality Linf',f'{orth:.3e}');
 for r in (10,25,50,100,150,200,300,400):
  if r<=a.rank:print(f'captured centered {a.field} energy first {r:3d}: {cum[r-1]:.8f}')
 print('seconds',f'{time.time()-t:.1f}')
if __name__=='__main__':main()
