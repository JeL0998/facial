"""
facesvc (localhost-only) — Detect + Liveness + Identify (+Enroll)
Security: HMAC headers required; bound to 127.0.0.1 only.
Models:
  - Detector: YOLOv8-face (.pt). Auto-download if missing.
  - Embedding: Facenet (VGGFace2 512-D) by default; ArcFace TorchScript optional.
  - Anti-spoof: TorchScript optional; else heuristic fallback (variance+CLAHE).
ENV knobs:
  FACESVC_SECRET, Y8_FACE_MODEL, ARC_PTH, FAS_MODEL_PATH,
  LIVE_T(0.80), COS_T(0.60), MARGIN_T(0.05), YOLO_IMG(640), TORCH_T(4),
  FAS_CLAHE(1), FAS_MEDIAN(3)
"""
from fastapi import FastAPI, Request, HTTPException, Depends
from pydantic import BaseModel
import os, time, base64, hashlib, hmac, secrets, csv, glob, urllib.request
import numpy as np, cv2, torch
from torchvision import transforms
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1

# ---------- Config ----------
APP="facesvc"; ROOT=os.path.dirname(__file__)
LOG_DIR=os.path.join(ROOT,"data","logs"); GALLERY=os.path.join(ROOT,"gallery")
os.makedirs(LOG_DIR, exist_ok=True); os.makedirs(GALLERY, exist_ok=True)
HMAC_SECRET=(os.getenv("FACESVC_SECRET") or "e20d14ea2eb40e58a3ab37646ef520d8f4d02b6f47ad9977af0f1966d970f02d").encode()
LIVE_T=float(os.getenv("LIVE_T","0.80")); COS_T=float(os.getenv("COS_T","0.60"))
MARGIN_T=float(os.getenv("MARGIN_T","0.05")); YOLO_IMG=int(os.getenv("YOLO_IMG","640"))
torch.set_num_threads(max(1,int(os.getenv("TORCH_T","4"))))
FAS_CLAHE=int(os.getenv("FAS_CLAHE","1")); FAS_MEDIAN=int(os.getenv("FAS_MEDIAN","3"))

# ---------- App ----------
app=FastAPI(title=APP)
_seen={}  # nonce->ts (simple anti-replay)

def _hmac_ok(req:Request, body:bytes):
    ts=req.headers.get("X-Timestamp"); nonce=req.headers.get("X-Nonce")
    body_sha=req.headers.get("X-Body-SHA256"); sig=req.headers.get("X-Signature")
    if not (ts and nonce and body_sha and sig): raise HTTPException(401,"missing auth")
    now=int(time.time()); ts_i=int(ts)
    if abs(now-ts_i)>60: raise HTTPException(401,"stale")
    if nonce in _seen and now-_seen[nonce]<300: raise HTTPException(401,"replay")
    _seen[nonce]=now
    if hashlib.sha256(body).hexdigest()!=body_sha: raise HTTPException(401,"body hash")
    msg=f"{req.method}\n{req.url.path}\n{ts}\n{nonce}\n{body_sha}".encode()
    exp=base64.b64encode(hmac.new(HMAC_SECRET,msg,hashlib.sha256).digest()).decode()
    if not secrets.compare_digest(exp,sig): raise HTTPException(401,"bad sig")
async def require_hmac(req:Request): _hmac_ok(req, await req.body()); return True

# ---------- Weights resolve (auto-download YOLOv8-face) ----------
def _resolve_face(p:str)->str:
    if p and os.path.isfile(p): return p
    dst=os.path.join(ROOT,"weights","yolov8n-face.pt")
    if os.path.isfile(dst): return dst
    os.makedirs(os.path.dirname(dst),exist_ok=True)
    url="https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.pt"
    urllib.request.urlretrieve(url,dst); return dst

# ---------- Models (load once) ----------
yolo=YOLO(_resolve_face(os.getenv("Y8_FACE_MODEL","")))
device="cpu"
fas=None; fp=os.getenv("FAS_MODEL_PATH","")
if fp and os.path.exists(fp): fas=torch.jit.load(fp,map_location=device).eval()

# Embeddings: ArcFace TorchScript optional; else Facenet
emb=None; emb_name="facenet-vggface2-512"; size=160
norm=transforms.Normalize(mean=[0.5]*3,std=[0.5]*3)
to112=transforms.Compose([transforms.ToTensor(),transforms.Resize((112,112)),norm])
to160=transforms.Compose([transforms.ToTensor(),transforms.Resize((160,160)),norm])
ap=os.getenv("ARC_PTH","")
if ap and os.path.exists(ap): emb=torch.jit.load(ap,map_location=device).eval(); emb_name="arcface-iresnet100-512"; size=112
else: emb=InceptionResnetV1(pretrained="vggface2").eval()

# ---------- Helpers ----------
def _b64img(b64:str):
    if "," in b64: b64=b64.split(",",1)[1]
    arr=np.frombuffer(base64.b64decode(b64,validate=False),np.uint8)
    img=cv2.imdecode(arr,cv2.IMREAD_COLOR); 
    if img is None: raise HTTPException(400,"bad_base64"); 
    return img
def _log(row):
    ym=time.strftime("%Y-%m"); p=os.path.join(LOG_DIR,ym); os.makedirs(p,exist_ok=True)
    f=os.path.join(p,"events.csv"); new=not os.path.exists(f)
    with open(f,"a",newline="") as fp:
        w=csv.writer(fp); 
        if new: w.writerow(["CapturedAt","DeviceId","Faces","Liveness","Info"])
        w.writerow(row)
def _prep_gray(g):
    if FAS_MEDIAN and FAS_MEDIAN%2==1: g=cv2.medianBlur(g,FAS_MEDIAN)
    if FAS_CLAHE: g=cv2.createCLAHE(2.0,(8,8)).apply(g)
    return g
def _live(gray_roi):
    if gray_roi.size==0: return 0.0
    if fas is not None:
        rgb=cv2.cvtColor(cv2.cvtColor(gray_roi,cv2.COLOR_GRAY2BGR),cv2.COLOR_BGR2RGB)
        t=to112(rgb).unsqueeze(0)
        with torch.inference_mode(): p=float(torch.sigmoid(fas(t)).cpu().numpy().squeeze())
        return max(0.0,min(1.0,p))
    lap=float(cv2.Laplacian(_prep_gray(gray_roi),cv2.CV_64F).var())
    return max(0.0,min(1.0,lap/300.0))
def _crop_to_tensor(img,box,target):
    x,y,w,h=box; x2,y2=x+w,y+h
    x,y=max(0,x),max(0,y); x2,y2=min(img.shape[1],x2),min(img.shape[0],y2)
    crop=cv2.cvtColor(img[y:y2,x:x2],cv2.COLOR_BGR2RGB)
    tfm=to112 if target==112 else to160
    return tfm(crop).unsqueeze(0)

# ---------- Gallery (file vectors only) ----------
G={},{}  # (vecs,means)
def _l2n(v): n=np.linalg.norm(v,axis=-1,keepdims=True)+1e-8; return v/n
def _load_gallery():
    V, M = {}, {}
    for person in [d for d in os.listdir(GALLERY) if os.path.isdir(os.path.join(GALLERY,d))]:
        vecs=[]
        for f in glob.glob(os.path.join(GALLERY,person,"*.npy")):
            try: v=np.load(f).astype("float32")
            except: continue
            if v.ndim==1 and v.shape[0]==512: vecs.append(v)
        if vecs:
            vv=_l2n(np.stack(vecs,0)); V[person]=vv; M[person]=_l2n(vv.mean(0,keepdims=True))[0]
    return V,M
def _save_embed(pid,vec):
    d=os.path.join(GALLERY,pid); os.makedirs(d,exist_ok=True)
    np.save(os.path.join(d,f"{time.strftime('%Y%m%d_%H%M%S')}.npy"),vec.astype("float32"))
    G[0].setdefault(pid,np.empty((0,512),dtype="float32"))
    G[0][pid]=_l2n(np.vstack([G[0][pid],vec]))
    G[1][pid]=_l2n(G[0][pid].mean(0,keepdims=True))[0]
G=_load_gallery()

# ---------- Schemas ----------
class ImgIn(BaseModel): image:str; device_id:str|None=None
class EnrollIn(BaseModel): person_id:str; image:str; device_id:str|None=None

# ---------- Endpoints ----------
@app.get("/") 
def root(): return {"status":"ok","data":{"message":"facesvc up. See /docs"},"errors":[]}

@app.post("/detect_liveness")
def detect_liveness(x:ImgIn, _:bool=Depends(require_hmac)):
    img=_b64img(x.image); h,w=img.shape[:2]; gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    r=yolo.predict(img,verbose=False,imgsz=YOLO_IMG)
    out=[]
    for b in r[0].boxes:
        x1,y1,x2,y2=map(int,b.xyxy[0].tolist()); box=[x1,y1,x2-x1,y2-y1]
        live=_live(gray[y1:y2,x1:x2])
        out.append({"box":box,"det":float(b.conf[0]),"live_score":live,"live_label":"live" if live>=LIVE_T else "spoof"})
    _log([time.strftime("%Y-%m-%d %H:%M:%S"),x.device_id or "",len(out),f"{np.mean([f['live_score'] for f in out]) if out else 0:.3f}",""])
    return {"status":"ok","data":{"faces":out},"errors":[]}

@app.post("/embed")
@torch.inference_mode()
def embed(x:ImgIn, _:bool=Depends(require_hmac)):
    img=_b64img(x.image); gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    r=yolo.predict(img,verbose=False,imgsz=YOLO_IMG)
    if len(r[0].boxes)==0: return {"status":"error","data":None,"errors":["no_face"]}
    # choose largest live face
    picks=[]
    for b in r[0].boxes:
        x1,y1,x2,y2=map(int,b.xyxy[0].tolist()); live=_live(gray[y1:y2,x1:x2])
        picks.append(((x2-x1)*(y2-y1),live,[x1,y1,x2-x1,y2-y1]))
    area,live,box=max(picks,key=lambda t:t[0])
    if live<LIVE_T: return {"status":"error","data":None,"errors":["not_live"]}
    v=emb(_crop_to_tensor(img,box,size)).squeeze(0).cpu().numpy().astype("float32")
    v=v/(np.linalg.norm(v)+1e-8)
    return {"status":"ok","data":{"vector":v.tolist(),"model_ver":emb_name},"errors":[]}

@app.post("/enroll")
@torch.inference_mode()
def enroll(x:EnrollIn, _:bool=Depends(require_hmac)):
    if not x.person_id: return {"status":"error","data":None,"errors":["person_id_required"]}
    r=embed(ImgIn(image=x.image,device_id=x.device_id), True)  # reuse gate
    if r["status"]!="ok": return r
    vec=np.array(r["data"]["vector"],dtype="float32"); _save_embed(x.person_id,vec)
    return {"status":"ok","data":{"person_id":x.person_id},"errors":[]}

@app.post("/identify")
@torch.inference_mode()
def identify(x:ImgIn, _:bool=Depends(require_hmac)):
    if not G[1]: return {"status":"error","data":None,"errors":["gallery_empty"]}
    r=embed(x, True); 
    if r["status"]!="ok": return r
    v=np.array(r["data"]["vector"],dtype="float32")
    names=list(G[1].keys()); M=np.stack([G[1][n] for n in names],0)  # Kx512
    scores=(M@v).astype("float32"); i=int(np.argmax(scores)); best=float(scores[i])
    margin=best - (float(np.partition(scores,-2)[-2]) if scores.size>=2 else best)
    match=(best>=COS_T) and (margin>=MARGIN_T)
    return {"status":"ok","data":{"match":match,"person_id":(names[i] if match else "unknown"),
            "score":best,"margin":margin}, "errors":[]}
