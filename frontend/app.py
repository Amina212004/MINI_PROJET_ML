# frontend/app.py
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import folium
from streamlit_folium import st_folium
from PIL import Image
import io

# Page config
st.set_page_config(page_title="Meteorite Type Predictor", layout="wide", initial_sidebar_state="expanded")

# -------------------------
# 1) Galaxy animated background (FULLSCREEN FIX)
# -------------------------
components.html(
    """
    <style>
    /* Reset all main containers to transparent */
    .main .block-container, .main, #MainMenu, header, footer {
        background: transparent !important;
    }
    
    /* Make sure our content areas are visible */
    .glass {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.08);
        backdrop-filter: blur(6px) saturate(120%);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 6px 30px rgba(2,6,23,0.6);
        position: relative;
        z-index: 1;
    }

    /* Canvas covers ENTIRE screen */
    #galaxy-root {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: -9999;
        pointer-events: none;
    }
    
    #galaxy {
        width: 100%;
        height: 100%;
    }
    
    /* Ensure streamlit content is above canvas */
    .main .block-container {
        position: relative;
        z-index: 1;
    }
    </style>

    <div id="galaxy-root">
      <canvas id="galaxy"></canvas>
    </div>

    <script>
    // Galaxy animation (same code but now covers full screen)
    (function(){
      const canvas = document.getElementById('galaxy');
      const ctx = canvas.getContext('2d');

      function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
      }
      resize();
      window.addEventListener('resize', resize);

      // Nebula layers
      function drawNebula(t) {
        const w = canvas.width, h = canvas.height;
        const g = ctx.createLinearGradient(0, 0, w, h);
        g.addColorStop(0, 'rgba(24,0,40,0.35)');
        g.addColorStop(0.4, 'rgba(8,0,60,0.28)');
        g.addColorStop(1, 'rgba(0,0,12,0.4)');
        ctx.fillStyle = g;
        ctx.fillRect(0, 0, w, h);

        // moving fog
        ctx.globalAlpha = 0.06;
        ctx.fillStyle = 'rgba(120,60,210,0.18)';
        ctx.beginPath();
        ctx.ellipse(w*0.35 + Math.sin(t/120)*200, h*0.45 + Math.cos(t/140)*120, 520, 260, 0, 0, Math.PI*2);
        ctx.fill();

        ctx.fillStyle = 'rgba(10,140,255,0.06)';
        ctx.beginPath();
        ctx.ellipse(w*0.7 + Math.cos(t/100)*200, h*0.6 + Math.sin(t/170)*150, 420, 220, 0, 0, Math.PI*2);
        ctx.fill();
        ctx.globalAlpha = 1;
      }

      // Stars
      const stars = [];
      for (let i=0;i<420;i++){
        stars.push({
          x: Math.random()*canvas.width,
          y: Math.random()*canvas.height,
          r: Math.random()*1.3 + 0.2,
          a: Math.random(),
          flick: 0.002 + Math.random()*0.03
        });
      }

      // Shooting stars
      let shoots = [];

      function spawnShoot() {
        if (Math.random() < 0.017) {
          shoots.push({
            x: Math.random()*canvas.width,
            y: Math.random()*canvas.height*0.4,
            vx: -4 - Math.random()*6,
            vy: 3 + Math.random()*6,
            life: 0,
            max: 60 + Math.random()*60
          });
        }
      }

      let t = 0;
      function render() {
        t++;
        ctx.clearRect(0,0,canvas.width, canvas.height);

        // background gradient
        const bg = ctx.createRadialGradient(canvas.width/2, canvas.height/2, 50, canvas.width/2, canvas.height/2, canvas.width);
        bg.addColorStop(0, 'rgba(8,6,24,0.6)');
        bg.addColorStop(1, 'rgba(0,0,6,0.9)');
        ctx.fillStyle = bg;
        ctx.fillRect(0,0,canvas.width, canvas.height);

        drawNebula(t);

        // draw stars
        for (let s of stars) {
          ctx.beginPath();
          ctx.arc(s.x, s.y, s.r, 0, Math.PI*2);
          const alpha = 0.2 + 0.8*(0.5 + 0.5*Math.sin(t*s.flick*10 + s.x*0.001));
          ctx.fillStyle = 'rgba(255,255,255,' + Math.min(1, alpha*s.a) + ')';
          ctx.fill();
        }

        // shooting stars
        spawnShoot();
        for (let i=shoots.length-1;i>=0;i--){
          const sh = shoots[i];
          ctx.beginPath();
          ctx.moveTo(sh.x, sh.y);
          const nx = sh.x + sh.vx*4;
          const ny = sh.y + sh.vy*4;
          ctx.lineTo(nx, ny);
          ctx.strokeStyle = 'rgba(255,255,255,0.95)';
          ctx.lineWidth = 2;
          ctx.stroke();

          // tail
          for(let k=0;k<8;k++){
            const px = sh.x + (sh.vx*(k/8));
            const py = sh.y + (sh.vy*(k/8));
            ctx.beginPath();
            ctx.arc(px, py, (8-k)*0.5, 0, Math.PI*2);
            ctx.fillStyle = 'rgba(255,200,120,' + (0.25*(1 - k/8)) + ')';
            ctx.fill();
          }

          sh.x += sh.vx;
          sh.y += sh.vy;
          sh.life++;
          if (sh.life > sh.max || sh.x < -100 || sh.y > canvas.height + 100) shoots.splice(i,1);
        }

        requestAnimationFrame(render);
      }

      render();
    })();
    </script>
    """,
    height=1,  # ← CHANGÉ À 0 pour prendre toute la hauteur
    scrolling=False,
)

# -------------------------
# 2) Top bar with logo and title (glass)
# -------------------------
# Put a small header inside two columns so it centers nicely.
col1, col2, col3 = st.columns([1,6,1])
with col1:
    st.write("")  # spacer
with col2:
    # use the uploaded screenshot as a small logo/header image if present
    try:
        logo = Image.open("/mnt/data/Ícone Configurar.jpg")
        st.image(logo, width=120)
    except Exception:
        st.markdown("<h2 style='text-align:center;color:white'>🪐 Meteorite Type Predictor</h2>", unsafe_allow_html=True)
with col3:
    st.write("")


# -------------------------
# 3) Sidebar (filters + neon button)
# -------------------------
st.sidebar.markdown("<div class='glass' style='padding:10px;'>", unsafe_allow_html=True)
st.sidebar.header("🔍 Filtres de Recherche")
year = st.sidebar.selectbox("Époque", ["All", "1800-1900", "1900-1950", "1950-2000", "2000-2020"])
continent = st.sidebar.selectbox("Continent", ["All", "Africa", "Asia", "Europe", "North America", "South America", "Oceania"])
mass = st.sidebar.slider("Masse (g)", 1, 100000, 1000)

st.sidebar.markdown("<hr style='border-color: rgba(255,255,255,0.06);'>", unsafe_allow_html=True)
predict_clicked = st.sidebar.button("Predict", key="predict", help="Click to call the backend API (mock for now)")
st.sidebar.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# 4) Main content layout: left results / right map
# -------------------------
left, right = st.columns([3,5])

with left:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    # Results Card
    st.markdown("### 🔮 Résultat de la prédiction")
    # Mock values — will be replaced by backend results later
    if predict_clicked:
        st.success("Type le plus probable : **L6**")
        st.info("Confiance estimée : **86%**")
        st.markdown("<p style='color:#ffd28a'>📊 5.2x plus probable qu'au hasard</p>", unsafe_allow_html=True)
    else:
        st.info("Appuyez sur **Predict** pour lancer la prédiction (mock pour l'instant)")

    # Top-3
    st.markdown("### 🥇 Top 3 types probables")
    if predict_clicked:
        st.markdown("""
        - **L6** — 86%  
        - **LL5** — 43%  
        - **H4** — 29%  
        """, unsafe_allow_html=True)
    else:
        st.write("- L6  - LL5  - H4  (top candidates)")

    st.markdown("<hr style='border-color: rgba(255,255,255,0.06);'>", unsafe_allow_html=True)

    # Proofs / Stats simple mock
    st.markdown("### 📈 Statistiques & Preuves")
    st.write("- Nombre de hits dans la base : **124**")
    st.write("- Moyenne des masses (g) : **4,200**")
    st.write("- Intervalle d'années : 1901 — 1998")

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("### 🌍 Carte des météorites")

    # Dataset demo
    df = pd.DataFrame({
        "lat": [34.2, 28.3, 51.5, -33.9, 40.7, 35.6, 48.8, -25.3],
        "lon": [3.1, 2.9, -0.1, 151.2, -74.0, 139.7, 2.4, 133.0],
        "type": ["L6","H5","LL3","L6","LL5","H4","L6","CM2"],
        "mass": [1200, 5400, 320, 70000, 1500, 8900, 4300, 2800],
        "year": [1890, 1948, 1965, 2001, 1987, 1975, 1992, 1968]
    })

    # Create Folium map with BETTER VISIBILITY
    m = folium.Map(
        location=[20, 0], 
        zoom_start=2, 
        tiles="CartoDB positron",  # ← CHANGÉ : plus clair!
        attr="CartoDB"
    )
    
    # Color coding for meteorite types
    type_colors = {
        "L6": "#ff6b6b",   # Red
        "H5": "#4ecdc4",   # Teal  
        "LL3": "#45b7d1",  # Blue
        "LL5": "#96ceb4",  # Green
        "H4": "#feca57",   # Yellow
        "CM2": "#ff9ff3"   # Pink
    }

    for _, row in df.iterrows():
        color = type_colors.get(row["type"], "#ffb347")
        
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=8 + (row["mass"] / 10000),  # Size based on mass
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            weight=2,
            popup=folium.Popup(
                f"""
                <div style='font-family: Arial; color: #333;'>
                    <b>Type:</b> {row['type']}<br>
                    <b>Masse:</b> {row['mass']} g<br>
                    <b>Année:</b> {row['year']}<br>
                    <b>Position:</b> {row['lat']:.1f}, {row['lon']:.1f}
                </div>
                """,
                max_width=250
            )
        ).add_to(m)

    st_folium(m, width=900, height=560)
    
    # Légende
    st.markdown("""
    <div style='background: rgba(255,255,255,0.1); padding: 10px; border-radius: 8px; margin-top: 10px;'>
    <h4 style='color: white; margin: 0;'>🎨 Légende des couleurs :</h4>
    <div style='display: flex; flex-wrap: wrap; gap: 10px; margin-top: 8px;'>
        <div style='display: flex; align-items: center;'><div style='width: 12px; height: 12px; background: #ff6b6b; border-radius: 50%; margin-right: 5px;'></div><span style='color: white;'>L6</span></div>
        <div style='display: flex; align-items: center;'><div style='width: 12px; height: 12px; background: #4ecdc4; border-radius: 50%; margin-right: 5px;'></div><span style='color: white;'>H5</span></div>
        <div style='display: flex; align-items: center;'><div style='width: 12px; height: 12px; background: #45b7d1; border-radius: 50%; margin-right: 5px;'></div><span style='color: white;'>LL3</span></div>
        <div style='display: flex; align-items: center;'><div style='width: 12px; height: 12px; background: #96ceb4; border-radius: 50%; margin-right: 5px;'></div><span style='color: white;'>LL5</span></div>
        <div style='display: flex; align-items: center;'><div style='width: 12px; height: 12px; background: #feca57; border-radius: 50%; margin-right: 5px;'></div><span style='color: white;'>H4</span></div>
        <div style='display: flex; align-items: center;'><div style='width: 12px; height: 12px; background: #ff9ff3; border-radius: 50%; margin-right: 5px;'></div><span style='color: white;'>CM2</span></div>
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

s