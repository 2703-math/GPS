
import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="GPS Visual Lab", page_icon="🛰️", layout="wide")

# ===================== CONSTANTES =====================
C = 299_792_458.0
R_EARTH = 6_371_000.0

# ===================== ESTILO =========================
st.markdown("""
<style>
.block-container{padding-top:1.2rem}
.hero{padding:22px 26px;border-radius:18px;
background:linear-gradient(135deg,#101d35,#0a1222);
border:1px solid rgba(100,150,210,.25);margin-bottom:16px}
.hero h1{margin:0;color:#f4f8ff;font-size:2.35rem}
.hero p{color:#aebbd0;margin:.45rem 0 0}
.card{background:rgba(18,28,48,.72);border:1px solid rgba(120,150,190,.2);
border-radius:14px;padding:14px;min-height:90px}
.small{color:#94a5bd;font-size:.78rem;text-transform:uppercase;letter-spacing:.08em}
.big{color:#f4f8ff;font-size:1.5rem;font-weight:700;margin-top:5px}
.explain{border-left:3px solid #4ea1ff;background:rgba(60,110,180,.08);
padding:11px 15px;border-radius:0 10px 10px 0;color:#cbd6e6}
</style>
""", unsafe_allow_html=True)

# ===================== FUNÇÕES ========================
def ecef(lat, lon, alt=0):
    a, b = np.radians(lat), np.radians(lon)
    r = R_EARTH + alt
    return np.array([r*np.cos(a)*np.cos(b),
                     r*np.cos(a)*np.sin(b),
                     r*np.sin(a)])

def latlon(p):
    r=np.linalg.norm(p)
    return np.degrees(np.arcsin(p[2]/r)), np.degrees(np.arctan2(p[1],p[0])), r-R_EARTH

def rotz(P, ang):
    c,s=np.cos(ang),np.sin(ang)
    return P @ np.array([[c,-s,0],[s,c,0],[0,0,1]]).T

def satellites(n=4):
    # Constelação fictícia, mas em escala compatível com MEO/GPS.
    presets=[(55,20),(62,145),(38,230),(60,310),(72,85),(42,270)]
    out=[]
    for lat,lon in presets[:n]:
        out.append(ecef(lat,lon,20_200_000))
    return np.array(out)

def measure(sats, receiver, bias_s, noise_m):
    d=np.linalg.norm(sats-receiver,axis=1)
    rng=np.random.default_rng(42)
    return d+C*bias_s+rng.normal(0,noise_m,len(sats))

def solve(sats, rho, iterations=15):
    # incógnitas: x,y,z e erro de relógio b
    q=np.array([R_EARTH,0.,0.,0.])
    for _ in range(iterations):
        p=q[:3]; b=q[3]
        d=np.linalg.norm(sats-p,axis=1)
        pred=d+C*b
        res=rho-pred
        J=np.zeros((len(sats),4))
        for i in range(len(sats)):
            J[i,:3]=(p-sats[i])/max(d[i],1)
            J[i,3]=C
        delta=np.linalg.lstsq(J,res,rcond=None)[0]
        q += delta
        if np.linalg.norm(delta[:3])<1e-3:
            break
    return q[:3],q[3]

def sphere_mesh(center, radius, scale=1e6):
    u=np.linspace(0,2*np.pi,34)
    v=np.linspace(0,np.pi,18)
    x=center[0]+radius*np.outer(np.cos(u),np.sin(v))
    y=center[1]+radius*np.outer(np.sin(u),np.sin(v))
    z=center[2]+radius*np.outer(np.ones_like(u),np.cos(v))
    return x/scale,y/scale,z/scale

# ===================== CABEÇALHO =====================
st.markdown("""
<div class="hero">
<h1>🛰️ GPS Visual Lab</h1>
<p>Uma experiência visual para entender como sinais de rádio, tempo de voo,
trilateração e correção do relógio permitem determinar uma posição.</p>
</div>
""",unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Experimento")
    n=st.slider("Número de satélites",3,6,4)
    lat=st.slider("Latitude do receptor", -70.,70.,-22.56,.01)
    lon=st.slider("Longitude do receptor", -170.,170.,-47.40,.01)
    alt=st.slider("Altitude (m)",0.,5000.,100.,50.)
    bias_us=st.slider("Erro do relógio (µs)",-100.,100.,12.,1.)
    noise=st.slider("Ruído do sinal (m)",0.,50.,2.,.5)
    scene=st.slider("Tempo da cena",0.,30.,0.,.5)
    show_spheres=st.checkbox("Mostrar superfícies de distância",True)
    st.divider()
    st.caption("Modelo didático. As órbitas, mensagens e propagação são simplificadas para tornar os conceitos visíveis.")

receiver=ecef(lat,lon,alt)
sats=rotz(satellites(n),0.035*scene)
rho=measure(sats,receiver,bias_us*1e-6,noise)
estimated,bias_est=solve(sats,rho)
err=np.linalg.norm(estimated-receiver)
lat_e,lon_e,alt_e=latlon(estimated)
times=rho/C

# ===================== MÉTRICAS ======================
a,b,c,d=st.columns(4)
for col,title,value in [
    (a,"Erro de posição",f"{err:.2f} m"),
    (b,"Relógio real",f"{bias_us:+.1f} µs"),
    (c,"Relógio estimado",f"{bias_est*1e6:+.2f} µs"),
    (d,"Tempo de voo S1",f"{times[0]*1000:.2f} ms")]:
    with col:
        st.markdown(f'<div class="card"><div class="small">{title}</div><div class="big">{value}</div></div>',unsafe_allow_html=True)

st.write("")
st.markdown("""
<div class="explain">
<b>Ideia física:</b> cada satélite envia uma mensagem de rádio.
O receptor compara o instante de transmissão com o instante de chegada.
Como o sinal se propaga aproximadamente à velocidade da luz,
<code>distância ≈ c × tempo</code>. Várias distâncias restringem a posição.
O receptor ainda precisa descobrir o próprio erro de relógio.
</div>
""",unsafe_allow_html=True)

# ===================== 1. CENA 3D ====================
st.subheader("🌎 1. Constelação, transmissão e receptor")

u=np.linspace(0,2*np.pi,45); v=np.linspace(-np.pi/2,np.pi/2,24)
xe=R_EARTH*np.outer(np.cos(v),np.cos(u))/1e6
ye=R_EARTH*np.outer(np.cos(v),np.sin(u))/1e6
ze=R_EARTH*np.outer(np.sin(v),np.ones_like(u))/1e6

fig=go.Figure()
fig.add_trace(go.Surface(x=xe,y=ye,z=ze,colorscale=[[0,"#071426"],[.5,"#153b65"],[1,"#2b668d"]],
                         showscale=False,opacity=.82,hoverinfo="skip"))
fig.add_trace(go.Scatter3d(x=[receiver[0]/1e6],y=[receiver[1]/1e6],z=[receiver[2]/1e6],
                           mode="markers+text",marker=dict(size=8,color="#ff4f70"),
                           text=["RECEPTOR"],textposition="top center",name="Receptor real"))
fig.add_trace(go.Scatter3d(x=[estimated[0]/1e6],y=[estimated[1]/1e6],z=[estimated[2]/1e6],
                           mode="markers",marker=dict(size=6,color="#55e6a5",symbol="diamond"),
                           name="Posição calculada"))
fig.add_trace(go.Scatter3d(x=sats[:,0]/1e6,y=sats[:,1]/1e6,z=sats[:,2]/1e6,
                           mode="markers+text",marker=dict(size=7,color="#ffd166"),
                           text=[f"S{i+1}" for i in range(n)],textposition="top center",
                           name="Satélites"))

# Raios/sinais
for i,s in enumerate(sats):
    # ponto que representa a frente do sinal; muda com o tempo da cena
    phase=(scene*0.35+i/n)
    frac=phase%1
    pulse=s+(receiver-s)*frac
    fig.add_trace(go.Scatter3d(x=[s[0]/1e6,pulse[0]/1e6],
                               y=[s[1]/1e6,pulse[1]/1e6],
                               z=[s[2]/1e6,pulse[2]/1e6],
                               mode="lines+markers",
                               marker=dict(size=4,color="#4ea1ff"),
                               line=dict(color="#4ea1ff",width=4),
                               name=f"S{i+1} → receptor",showlegend=False))
    fig.add_trace(go.Scatter3d(x=[s[0]/1e6,receiver[0]/1e6],
                               y=[s[1]/1e6,receiver[1]/1e6],
                               z=[s[2]/1e6,receiver[2]/1e6],
                               mode="lines",line=dict(color="rgba(78,161,255,.18)",width=2),
                               showlegend=False,hoverinfo="skip"))

fig.update_layout(height=620,margin=dict(l=0,r=0,t=10,b=0),
                  paper_bgcolor="rgba(0,0,0,0)",
                  scene=dict(aspectmode="data",
                             xaxis_title="X (Mm)",yaxis_title="Y (Mm)",zaxis_title="Z (Mm)",
                             xaxis=dict(gridcolor="rgba(140,160,190,.15)"),
                             yaxis=dict(gridcolor="rgba(140,160,190,.15)"),
                             zaxis=dict(gridcolor="rgba(140,160,190,.15)"),
                             camera=dict(eye=dict(x=1.55,y=1.55,z=.95))),
                  legend=dict(bgcolor="rgba(8,15,28,.72)"))
st.plotly_chart(fig,use_container_width=True)

st.caption("Arraste a cena para girar. O controle “Tempo da cena” faz os satélites e os pulsos de sinal se deslocarem.")

# ===================== 2. TEMPO/DISTÂNCIA ============
st.subheader("📡 2. Tempo de voo → distância")

fig2=go.Figure()
fig2.add_trace(go.Bar(
    x=rho/1e6,y=[f"S{i+1}" for i in range(n)],orientation="h",
    marker_color="#4ea1ff",
    text=[f"{x/1e6:.3f} Mm" for x in rho],textposition="auto",
    hovertemplate="Pseudodistância: %{x:.3f} Mm<extra></extra>"))
fig2.update_layout(height=280,margin=dict(l=30,r=20,t=10,b=35),
                   xaxis_title="Pseudodistância (milhões de metros)",
                   paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig2,use_container_width=True)

st.info(f"Exemplo: S1 leva aproximadamente **{times[0]*1000:.2f} ms** para chegar ao receptor. Um erro de 1 µs no tempo corresponde a aproximadamente **{C*1e-6:.1f} m** de distância.")

# ===================== 3. TRILATERAÇÃO ================
st.subheader("📐 3. Trilateração: a interseção das distâncias")

st.markdown("""
Na realidade o cálculo é tridimensional: cada medida define uma **esfera**
ao redor do satélite. Para tornar o conceito intuitivo, abaixo mostramos uma
projeção 2D da geometria. O cálculo numérico mostrado na aplicação continua
sendo feito em 3D.
""")

# Projeção local em km
east=np.array([-np.sin(np.radians(lon)),np.cos(np.radians(lon)),0.])
north=np.array([-np.sin(np.radians(lat))*np.cos(np.radians(lon)),
                -np.sin(np.radians(lat))*np.sin(np.radians(lon)),
                np.cos(np.radians(lat))])
east/=np.linalg.norm(east); north/=np.linalg.norm(north)
xy=np.array([[np.dot(s-receiver,east)/1000,np.dot(s-receiver,north)/1000] for s in sats])

# Para não destruir a escala visual, normalizamos a representação.
# Os círculos representam a ideia de restrição de distância, não uma planta em escala.
scale=max(np.max(np.linalg.norm(xy,axis=1)),1)
fig3=go.Figure()
fig3.add_trace(go.Scatter(x=[0],y=[0],mode="markers+text",
                          marker=dict(size=13,color="#ff4f70"),text=["RECEPTOR"],
                          textposition="top center",name="Posição real"))
de=estimated-receiver
fig3.add_trace(go.Scatter(x=[np.dot(de,east)/1000],y=[np.dot(de,north)/1000],
                          mode="markers",marker=dict(size=10,color="#55e6a5",symbol="diamond"),
                          name="Calculada"))

th=np.linspace(0,2*np.pi,300)
for i in range(n):
    # raio aparente para visualização didática
    r=max(np.linalg.norm(xy[i]),1)*1.0
    fig3.add_trace(go.Scatter(x=xy[i,0]+r*np.cos(th),y=xy[i,1]+r*np.sin(th),
                              mode="lines",line=dict(color="rgba(78,161,255,.35)",width=2),
                              showlegend=False,hoverinfo="skip"))
    fig3.add_trace(go.Scatter(x=[xy[i,0]],y=[xy[i,1]],mode="markers+text",
                              marker=dict(size=8,color="#ffd166"),text=[f"S{i+1}"],
                              textposition="top center",name=f"S{i+1}"))
lim=max(1000,scale*1.2)
fig3.update_layout(height=560,margin=dict(l=30,r=20,t=10,b=35),
                   xaxis=dict(title="Leste-Oeste (km)",range=[-lim,lim],scaleanchor="y",scaleratio=1),
                   yaxis=dict(title="Norte-Sul (km)",range=[-lim,lim]),
                   paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig3,use_container_width=True)

st.warning("Os círculos são uma representação didática 2D. O GPS real usa superfícies esféricas em 3D e resolve simultaneamente a posição e o erro do relógio.")

# ===================== 4. SOLUÇÃO =====================
st.subheader("🧠 4. O receptor resolve 4 incógnitas")

st.latex(r"\rho_i=\sqrt{(x-x_i)^2+(y-y_i)^2+(z-z_i)^2}+c\,b")

st.markdown("""
As quatro incógnitas do modelo são **x, y, z** e **b** (erro do relógio).
Cada satélite fornece uma equação. Com quatro satélites, temos informação
suficiente para resolver o sistema em condições geométricas adequadas.
Satélites adicionais fornecem redundância.
""")

cc1,cc2,cc3=st.columns(3)
cc1.metric("Latitude real",f"{lat:.3f}°")
cc2.metric("Latitude calculada",f"{lat_e:.3f}°")
cc3.metric("Altitude calculada",f"{alt_e:.1f} m")

# ===================== 5. CONVERGÊNCIA ===============
st.subheader("🔬 5. Convergência do método numérico")

q=np.array([R_EARTH,0.,0.,0.])
conv=[]
for _ in range(12):
    p=q[:3]; bb=q[3]
    dist=np.linalg.norm(sats-p,axis=1)
    residual=rho-(dist+C*bb)
    J=np.zeros((n,4))
    for i in range(n):
        J[i,:3]=(p-sats[i])/max(dist[i],1); J[i,3]=C
    q+=np.linalg.lstsq(J,residual,rcond=None)[0]
    conv.append(np.linalg.norm(q[:3]-receiver))

fig4=go.Figure()
fig4.add_trace(go.Scatter(x=list(range(1,len(conv)+1)),y=conv,
                          mode="lines+markers",line=dict(color="#55e6a5",width=4),
                          marker=dict(size=8),hovertemplate="Iteração %{x}<br>Erro: %{y:.3f} m<extra></extra>"))
fig4.update_layout(height=320,margin=dict(l=30,r=20,t=10,b=40),
                   xaxis_title="Iteração",yaxis_title="Erro de posição (m)",
                   paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig4,use_container_width=True)

# ===================== 6. TABELA ======================
st.subheader("📊 Dados dos satélites")
rows=[]
for i in range(n):
    rows.append({
        "Satélite":f"S{i+1}",
        "Distância (km)":round(rho[i]/1000,3),
        "Tempo (ms)":round(times[i]*1000,5),
        "X (Mm)":round(sats[i,0]/1e6,3),
        "Y (Mm)":round(sats[i,1]/1e6,3),
        "Z (Mm)":round(sats[i,2]/1e6,3),
    })
st.dataframe(rows,use_container_width=True,hide_index=True)

st.divider()
st.markdown("""
<div style="text-align:center;color:#8190a7;font-size:.82rem">
<b>GPS Visual Lab</b> · Física Computacional para Ensino Médio<br>
Modelo educacional simplificado de posicionamento por satélite.
</div>
""",unsafe_allow_html=True)
