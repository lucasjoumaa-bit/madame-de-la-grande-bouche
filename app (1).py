"""
Madame de la Grande Bouche — Streamlit App
Deploy on Streamlit Cloud: share.streamlit.io
"""
import subprocess, sys
subprocess.run([sys.executable,'-m','pip','install','torch','torchvision','--index-url','https://download.pytorch.org/whl/cpu'], capture_output=True)
subprocess.run([sys.executable,'-m','pip','uninstall','-y','opencv-python'], capture_output=True)
subprocess.run([sys.executable,'-m','pip','install','opencv-python-headless'], capture_output=True)
import os, io, json, uuid, base64, datetime, requests
import numpy as np
import torch
import torch.nn as nn
import streamlit as st

from PIL         import Image
from ultralytics import YOLO
from torchvision import transforms as T
from groq        import Groq

# ── page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title = 'Madame de la Grande Bouche',
    page_icon  = '👗',
    layout     = 'wide',
)

# ── custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main-header {
        font-family: 'Playfair Display', serif;
        font-size: 3rem; font-weight: 700;
        background: linear-gradient(135deg, #6B2D8B, #C06BA8);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 0.2rem;
    }
    .sub-header { font-size: 1rem; color: #888; text-align: center; margin-bottom: 2rem; letter-spacing: 0.05em; }
    .weather-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem 1.5rem; border-radius: 14px; color: white; margin-bottom: 1rem;
    }
    .weather-card h4 { margin: 0 0 0.4rem 0; font-size: 1rem; opacity: 0.9; }
    .weather-card .temp { font-size: 2rem; font-weight: 700; margin: 0; }
    .weather-card p { margin: 0.2rem 0 0 0; opacity: 0.85; font-size: 0.9rem; }
    .rec-card {
        background: #faf7ff; border-left: 4px solid #6B2D8B;
        padding: 1.2rem 1.5rem; border-radius: 0 14px 14px 0;
        margin-bottom: 1rem; white-space: pre-wrap;
    }
    .item-label { font-size: 0.75rem; text-align: center; color: #555; margin-top: 0.2rem; }
    .stButton > button {
        background: linear-gradient(135deg, #6B2D8B, #9B59B6);
        color: white !important; border: none !important;
        border-radius: 10px; padding: 0.6rem 1.5rem;
        font-weight: 600; font-size: 0.95rem; width: 100%;
    }
    .sidebar-title { font-family: 'Playfair Display', serif; font-size: 1.3rem; color: #6B2D8B; margin-bottom: 0.5rem; }
    .badge { display: inline-block; padding: 0.2rem 0.7rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
    .badge-green { background: #e8f5e9; color: #2e7d32; }
</style>
""", unsafe_allow_html=True)


# ── Drive file IDs ────────────────────────────────────────────────────────
DRIVE_FILES = {
    'fyp/runs/fashionpedia_yolo26s/weights/best.pt': '16TxxwOUcT6sN0yMaC59MSD2-qghXS3WO',
    'fyp/attribute_classifier/best_model.pt'        : '17UPUnoEFBc_HWslmK0w0CN9HJP0Uftgu',
    'fyp/attribute_classifier/attribute_map.json'   : 'https://drive.google.com/file/d/1JJeqKuuxQZhn5htjRXnUnWTI3h6rsKwR/view?usp=sharing',  # ← fix this
    'fyp/clean/instances_train2020_clean.json'      : '1fwr9wmodQqI66SmtvkiiGpFweEe1zHHR',
}


@st.cache_resource(show_spinner='Downloading model files from Drive...')
def download_models():
    import gdown
    for local_path, file_id in DRIVE_FILES.items():
        if 'REPLACE' in file_id:
            st.error(f'❌ Missing file ID for {os.path.basename(local_path)} — please fix DRIVE_FILES in app.py')
            st.stop()
        if not os.path.exists(local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            gdown.download(f'https://drive.google.com/uc?id={file_id}', local_path, quiet=True)


download_models()


# ── attribute rules + color map ───────────────────────────────────────────
CATEGORY_ATTR_RULES = {
    'shirt'     : ['textile pattern','length','silhouette','neckline type','opening type','waistline'],
    'blouse'    : ['textile pattern','length','silhouette','neckline type','opening type','waistline'],
    'top'       : ['textile pattern','length','silhouette','neckline type','opening type','waistline'],
    'sweater'   : ['textile pattern','length','silhouette','neckline type','opening type'],
    'sweatshirt': ['textile pattern','length','silhouette','neckline type','opening type'],
    't-shirt'   : ['textile pattern','length','silhouette','neckline type'],
    'jacket'    : ['textile pattern','length','silhouette','opening type'],
    'coat'      : ['textile pattern','length','silhouette','opening type'],
    'dress'     : ['textile pattern','length','silhouette','neckline type','opening type','waistline'],
    'jumpsuit'  : ['textile pattern','length','silhouette','neckline type','opening type','waistline'],
    'pants'     : ['textile pattern','length','silhouette','waistline','opening type'],
    'skirt'     : ['textile pattern','length','silhouette','waistline'],
    'shorts'    : ['textile pattern','length','silhouette','waistline'],
    'jeans'     : ['textile pattern','length','silhouette','waistline','opening type'],
    'leggings'  : ['textile pattern','length','silhouette'],
    'bag'       : ['textile pattern'], 'wallet': ['textile pattern'],
    'shoe': [], 'boot': [], 'sandal': [], 'sneaker': [], 'heel': [],
}
COLOR_RGB = {
    'black':(20,20,20),'white':(245,245,245),'gray':(128,128,128),
    'red':(220,20,60),'orange':(255,140,0),'yellow':(255,215,0),
    'green':(34,139,34),'blue':(30,144,255),'navy blue':(0,0,128),
    'purple':(138,43,226),'pink':(255,105,180),'brown':(139,69,19),
    'beige':(245,245,220),
}

def get_allowed(name):
    n = name.lower()
    for k, v in CATEGORY_ATTR_RULES.items():
        if k in n: return v
    return ['textile pattern','length','silhouette','neckline type','opening type','waistline']


# ── load all models (cached once) ─────────────────────────────────────────
@st.cache_resource(show_spinner='Loading AI models...')
def load_models():
    import clip, timm, chromadb

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    yolo = YOLO('fyp/runs/fashionpedia_yolo26s/weights/best.pt')
    with open('fyp/clean/instances_train2020_clean.json') as f:
        cd = json.load(f)
    id_to_name = {i: c['name'] for i, c in enumerate(sorted(cd['categories'], key=lambda x: x['id']))}

    clip_model, clip_prep = clip.load('ViT-B/32', device=DEVICE)
    clip_model.eval()

    COLORS = [
        'a black clothing item','a white clothing item','a gray clothing item',
        'a red clothing item','a orange clothing item','a yellow clothing item',
        'a green clothing item','a blue clothing item','a navy blue clothing item',
        'a purple clothing item','a pink clothing item','a brown clothing item',
        'a beige clothing item',
    ]
    color_tokens = clip.tokenize(COLORS).to(DEVICE)
    color_labels = [c.replace('a ','').replace(' clothing item','') for c in COLORS]

    with open('fyp/attribute_classifier/attribute_map.json') as f:
        attr_map = json.load(f)
    sel_attrs = attr_map['attributes']
    num_attrs = attr_map['num_attributes']

    class AttrNet(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.backbone = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(self.backbone.num_features, n))
        def forward(self, x): return self.head(self.backbone(x))

    attr_model = AttrNet(num_attrs).to(DEVICE)
    attr_model.load_state_dict(torch.load('fyp/attribute_classifier/best_model.pt', map_location=DEVICE))
    attr_model.eval()

    attr_tf = T.Compose([T.Resize((224,224)), T.ToTensor(),
                          T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    os.makedirs('/tmp/wardrobe_db', exist_ok=True)
    chroma   = chromadb.PersistentClient(path='/tmp/wardrobe_db')
    wardrobe = chroma.get_or_create_collection('wardrobe', metadata={'hnsw:space':'cosine'})

    return dict(device=DEVICE, yolo=yolo, id_to_name=id_to_name,
                clip=clip, clip_model=clip_model, clip_prep=clip_prep,
                color_tokens=color_tokens, color_labels=color_labels,
                attr_model=attr_model, attr_tf=attr_tf, sel_attrs=sel_attrs,
                wardrobe=wardrobe)


# ── inference helpers ─────────────────────────────────────────────────────
@torch.no_grad()
def get_color(crop, m):
    t  = m['clip_prep'](crop).unsqueeze(0).to(m['device'])
    i  = m['clip_model'].encode_image(t); i = i/i.norm(dim=-1,keepdim=True)
    tx = m['clip_model'].encode_text(m['color_tokens']); tx = tx/tx.norm(dim=-1,keepdim=True)
    p  = (i @ tx.T * 100).softmax(dim=-1).cpu().numpy()[0]
    best = m['color_labels'][int(p.argmax())]
    r,g,b = COLOR_RGB.get(best,(128,128,128))
    return {'rgb':(r,g,b),'hex':f'#{r:02x}{g:02x}{b:02x}','name':best}

@torch.no_grad()
def get_attributes(crop, class_name, m):
    t     = m['attr_tf'](crop).unsqueeze(0).to(m['device'])
    probs = torch.sigmoid(m['attr_model'](t)).cpu().numpy()[0]
    allowed = get_allowed(class_name)
    result  = {}
    for i, attr in enumerate(m['sel_attrs']):
        sc = attr['supercategory']
        if sc not in allowed: continue
        conf = float(probs[i])
        if sc not in result or conf > result[sc]['confidence']:
            result[sc] = {'label':attr['name'],'confidence':round(conf,4)}
    return result

@torch.no_grad()
def get_embedding(crop, m):
    t = m['clip_prep'](crop).unsqueeze(0).to(m['device'])
    e = m['clip_model'].encode_image(t); e = e/e.norm(dim=-1,keepdim=True)
    return e.cpu().numpy()[0].tolist()

def to_b64(crop):
    c = crop.copy(); c.thumbnail((224,224))
    buf = io.BytesIO(); c.save(buf,format='JPEG',quality=85)
    return base64.b64encode(buf.getvalue()).decode()

def run_pipeline(image, m):
    tmp = '/tmp/_upload.jpg'; image.save(tmp)
    W,H = image.size
    r   = m['yolo'].predict(source=tmp,conf=0.30,iou=0.45,imgsz=640,verbose=False)[0]
    if r.boxes is None or len(r.boxes)==0: return []
    items = []
    for i, box in enumerate(r.boxes):
        conf = float(box.conf.cpu())
        if conf < 0.30: continue
        cls_id = int(box.cls.cpu())
        name   = m['id_to_name'].get(cls_id,f'class_{cls_id}')
        x1,y1,x2,y2 = box.xyxy.cpu().numpy()[0]
        px=(x2-x1)*0.05; py=(y2-y1)*0.05
        crop = image.crop((max(0,x1-px),max(0,y1-py),min(W,x2+px),min(H,y2+py))).convert('RGB')
        items.append({'item_id':str(uuid.uuid4()),'class_name':name,'confidence':round(conf,4),
                      'color':get_color(crop,m),'attributes':get_attributes(crop,name,m),
                      'embedding':get_embedding(crop,m),'crop_b64':to_b64(crop),'crop_img':crop})
    return items

def insert(items, m):
    if not items: return 0
    ids=[]; emb=[]; metas=[]; docs=[]
    for item in items:
        meta = {'class_name':item['class_name'],'color_name':item['color']['name'],
                'color_hex':item['color']['hex'],'crop_b64':item['crop_b64']}
        for g,d in item['attributes'].items(): meta[f'attr_{g}']=d['label']
        doc = f"{item['class_name']}, {item['color']['name']}, "+', '.join(v['label'] for v in item['attributes'].values() if v)
        ids.append(item['item_id']); emb.append(item['embedding']); metas.append(meta); docs.append(doc)
    m['wardrobe'].upsert(ids=ids,embeddings=emb,metadatas=metas,documents=docs)
    return len(ids)


# ── weather ───────────────────────────────────────────────────────────────
def get_weather(city, key):
    try:
        d = requests.get(f'https://api.openweathermap.org/data/2.5/weather?q={city}&appid={key}&units=metric',timeout=10).json()
        t = round(d['main']['temp']); desc=d['weather'][0]['description']; cond=d['weather'][0]['main']
        if t<5:    adv='🧥 Very cold — heavy coat essential'
        elif t<12: adv='🧥 Cold — jacket needed'
        elif t<18: adv='🧤 Cool — light jacket recommended'
        elif t<24: adv='👕 Mild — light layers comfortable'
        elif t<30: adv='☀️ Warm — light breathable clothing'
        else:      adv='🌞 Hot — minimal clothing'
        rain = '☔ Rain expected — avoid delicate fabrics' if cond in ['Rain','Drizzle','Thunderstorm'] else ''
        return {'city':d['name'],'temp':t,'desc':desc,'advice':adv,'rain':rain,'ok':True}
    except:
        return {'city':city,'temp':20,'desc':'clear sky','advice':'👕 Mild — light layers','rain':'','ok':False}


# ── retrieval + LLM ───────────────────────────────────────────────────────
@torch.no_grad()
def retrieve(ctx, m, n=3):
    w = m['wardrobe']; total = w.count()
    if total==0: return {}
    tokens = m['clip'].tokenize([ctx]).to(m['device'])
    feat   = m['clip_model'].encode_text(tokens); feat=feat/feat.norm(dim=-1,keepdim=True)
    vec    = feat.cpu().numpy()[0].tolist()
    res    = w.query(query_embeddings=[vec],n_results=min(total,20),include=['metadatas','distances'])
    TOPS=['shirt','blouse','top','t-shirt','sweatshirt','sweater','jacket','coat','dress','jumpsuit']
    BOTS=['pants','skirt','shorts','jeans','leggings']
    SHOES=['shoe','boot','sandal','sneaker','heel']
    ACCS=['bag','wallet','belt','hat','scarf']
    def cat(name):
        nm=name.lower()
        for t in TOPS:
            if t in nm: return 'top'
        for b in BOTS:
            if b in nm: return 'bottom'
        for s in SHOES:
            if s in nm: return 'shoes'
        for a in ACCS:
            if a in nm: return 'accessory'
        return 'other'
    grouped={'top':[],'bottom':[],'shoes':[],'accessory':[]}
    for i in range(len(res['ids'][0])):
        meta=res['metadatas'][0][i]; ct=cat(meta.get('class_name',''))
        if ct in grouped and len(grouped[ct])<n:
            attrs={k.replace('attr_',''):v for k,v in meta.items() if k.startswith('attr_')}
            grouped[ct].append({'class_name':meta.get('class_name',''),'color':meta.get('color_name',''),
                                 'attributes':attrs,'crop_b64':meta.get('crop_b64','')})
    return {k:v for k,v in grouped.items() if v}


SYS = """You are Madame de la Grande Bouche — an expert personal fashion stylist with an elegant, warm personality.
Recommend the best outfit from the user's wardrobe based on weather, event, and personal style.
ONLY recommend items explicitly listed. Be specific, warm, and explain your reasoning beautifully."""


def recommend(city, event, notes, m, gkey, wkey):
    weather = get_weather(city, wkey)
    now=datetime.datetime.now(); h=now.hour
    tod='morning' if h<12 else 'afternoon' if h<17 else 'evening' if h<21 else 'night'
    sea='winter' if now.month in [12,1,2] else 'spring' if now.month in [3,4,5] else 'summer' if now.month in [6,7,8] else 'autumn'
    ctx=f"{event} outfit for {sea} {tod}, {weather['temp']}°C {weather['desc']}"
    grouped=retrieve(ctx,m)
    if not grouped: return weather,None,'Your wardrobe is empty. Please add some photos first.'
    all_items=[]; wlines=[]
    for cat,items in grouped.items():
        wlines.append(f'\n{cat.upper()}S:')
        for i,item in enumerate(items,1):
            astr=', '.join(f'{k}: {v}' for k,v in item['attributes'].items() if v)
            wlines.append(f"  {i}. {item['class_name']} — color: {item['color']}"+(f', {astr}' if astr else ''))
            all_items.append(item)
    prompt=f"""Context:
Date: {now.strftime('%A, %B %d %Y')}, {tod} | Season: {sea}
City: {weather['city']} — {weather['temp']}°C, {weather['desc']}
Weather advice: {weather['advice']}
{f"Rain: {weather['rain']}" if weather['rain'] else ''}
Event: {event}
{f"Notes: {notes}" if notes else ''}

Wardrobe:
{''.join(wlines)}

RECOMMENDED OUTFIT:
[list each item with a dash]

WHY THIS OUTFIT:
[warm reasoning about weather, event, color coordination]

STYLING TIPS:
[2-3 actionable tips]

WEATHER NOTE:
[one elegant sentence]"""
    gc=Groq(api_key=gkey)
    res=gc.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[{'role':'system','content':SYS},{'role':'user','content':prompt}],
        max_tokens=800,temperature=0.7)
    return weather,all_items,res.choices[0].message.content.strip()


# ── main ──────────────────────────────────────────────────────────────────
def main():
    st.markdown('<div class="main-header">👗 Madame de la Grande Bouche</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Your AI-Powered Personal Fashion Stylist</div>', unsafe_allow_html=True)
    st.markdown('---')

    with st.sidebar:
        st.markdown('<div class="sidebar-title">⚙️ Settings</div>', unsafe_allow_html=True)
        try:
            gkey  = st.secrets['GROQ_API_KEY']
            wkey  = st.secrets['OPENWEATHER_API_KEY']
            st.markdown('<span class="badge badge-green">✅ API keys loaded</span>', unsafe_allow_html=True)
        except:
            st.warning('Enter API keys manually:')
            gkey = st.text_input('Groq API Key', type='password')
            wkey = st.text_input('OpenWeather API Key', type='password')
        st.markdown('---')
        st.markdown('**How to use:**')
        st.markdown('1. 👗 Upload wardrobe photos')
        st.markdown('2. ✨ Enter your city and event')
        st.markdown('3. Get your perfect outfit!')
        st.markdown('---')
        st.caption('⚠️ Wardrobe resets on app restart (demo limitation)')
        st.caption('Madame de la Grande Bouche FYP — Lebanese University 2026')

    if not gkey or not wkey:
        st.warning('⚠️ API keys missing. Please check secrets or enter them in the sidebar.')
        st.stop()

    m = load_models()
    tab1, tab2, tab3 = st.tabs(['👗 Build Wardrobe', '✨ Get Recommendation', '👜 My Wardrobe'])

    # ── Tab 1 ─────────────────────────────────────────────────────────────
    with tab1:
        st.markdown('### Upload your clothing photos')
        st.markdown('Upload clear photos of your outfits. The AI will detect and analyze each clothing item automatically.')
        col1, col2 = st.columns([1, 2])
        with col1:
            uploaded = st.file_uploader('Choose photos', type=['jpg','jpeg','png','webp'],
                                         accept_multiple_files=True)
            btn = st.button('🔍 Analyze & Add to Wardrobe')
        with col2:
            if btn:
                if not uploaded:
                    st.warning('Please upload at least one photo.')
                else:
                    prog=st.progress(0); total_new=0; all_crops=[]
                    for i,file in enumerate(uploaded):
                        prog.progress((i+1)/len(uploaded), text=f'Processing {file.name}...')
                        try:
                            items=run_pipeline(Image.open(file).convert('RGB'),m)
                            if items:
                                total_new+=insert(items,m)
                                for item in items:
                                    all_crops.append((item['crop_img'],f"{item['class_name']}\n{item['color']['name']}"))
                            else:
                                st.warning(f'No clothing detected in {file.name}')
                        except Exception as e:
                            st.error(f'Error: {e}')
                    prog.empty()
                    st.success(f'✅ Added **{total_new}** items. Total: **{m["wardrobe"].count()}** items.')
                    if all_crops:
                        st.markdown('#### Detected items:')
                        cols=st.columns(min(len(all_crops),5))
                        for idx,(img,lbl) in enumerate(all_crops):
                            with cols[idx%5]:
                                st.image(img,use_column_width=True)
                                st.markdown(f'<div class="item-label">{lbl}</div>',unsafe_allow_html=True)

    # ── Tab 2 ─────────────────────────────────────────────────────────────
    with tab2:
        st.markdown('### Get your perfect outfit')
        col1, col2 = st.columns([1, 2])
        with col1:
            city  = st.text_input('📍 Your city', value='Beirut', placeholder='e.g. Beirut, Paris, London')
            event = st.text_input('📅 What are you doing today?', placeholder='e.g. business meeting, gym, first date...')
            notes = st.text_input('💬 Preferences? (optional)', placeholder='e.g. prefer dark colors...')
            go    = st.button('✨ Get My Outfit')
        with col2:
            if go:
                if not city.strip() or not event.strip():
                    st.warning('Please fill in your city and event.')
                elif m['wardrobe'].count()==0:
                    st.warning('Wardrobe is empty. Add photos first.')
                else:
                    with st.spinner('Generating your outfit recommendation...'):
                        weather,all_items,rec=recommend(city,event,notes,m,gkey,wkey)
                    rain_html=f'<p>⚠️ {weather["rain"]}</p>' if weather['rain'] else ''
                    st.markdown(f"""
                    <div class="weather-card">
                        <h4>🌤️ {weather['city']}</h4>
                        <p class="temp">{weather['temp']}°C</p>
                        <p>{weather['desc'].capitalize()}</p>
                        <p>{weather['advice']}</p>
                        {rain_html}
                    </div>""", unsafe_allow_html=True)
                    st.markdown(f'<div class="rec-card">{rec}</div>', unsafe_allow_html=True)
                    if all_items:
                        st.markdown('#### Items from your wardrobe:')
                        cols=st.columns(min(len(all_items),4))
                        for idx,item in enumerate(all_items):
                            if item.get('crop_b64'):
                                img=Image.open(io.BytesIO(base64.b64decode(item['crop_b64'])))
                                with cols[idx%4]:
                                    st.image(img,use_column_width=True)
                                    st.markdown(f'<div class="item-label"><b>{item["class_name"]}</b><br>{item["color"]}</div>',unsafe_allow_html=True)

    # ── Tab 3 ─────────────────────────────────────────────────────────────
    with tab3:
        st.markdown('### Your stored wardrobe')
        total=m['wardrobe'].count()
        c1,c2,_=st.columns([1,1,4])
        with c1:
            if st.button('🔄 Refresh'): st.rerun()
        with c2:
            if st.button('🗑️ Clear All'):
                ids=m['wardrobe'].get(limit=1000)['ids']
                if ids: m['wardrobe'].delete(ids=ids)
                st.success(f'Cleared {len(ids)} items.'); st.rerun()
        if total==0:
            st.info('Your wardrobe is empty. Go to Build Wardrobe to add items.')
        else:
            st.markdown(f'**{total} items stored**')
            results=m['wardrobe'].get(limit=100,include=['metadatas'])
            cols=st.columns(5)
            for idx,meta in enumerate(results['metadatas']):
                if meta.get('crop_b64'):
                    img=Image.open(io.BytesIO(base64.b64decode(meta['crop_b64'])))
                    attrs={k.replace('attr_',''):v for k,v in meta.items() if k.startswith('attr_')}
                    astr=' · '.join(v for v in list(attrs.values())[:2] if v)
                    with cols[idx%5]:
                        st.image(img,use_column_width=True)
                        st.markdown(f'<div class="item-label"><b>{meta.get("class_name","")}</b><br>{meta.get("color_name","")}<br><small>{astr}</small></div>',unsafe_allow_html=True)


if __name__=='__main__':
    main()
