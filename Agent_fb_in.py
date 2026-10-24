import sys, faulthandler
print("▶ Python démarré, chargement des modules…", flush=True)
faulthandler.enable()
# Si le script reste figé, affiche la pile d'appel toutes les 90 s (montre OÙ il bloque)
faulthandler.dump_traceback_later(90, repeat=True, file=sys.stderr)
import os
import io
import json
import re
import time
import urllib.parse
from pathlib import Path
from datetime import datetime, timezone

import requests
import feedparser
import cloudinary
import cloudinary.uploader
from bs4 import BeautifulSoup
from PIL import (
    Image, ImageDraw, ImageFont, ImageOps, ImageEnhance,
    ImageFilter, ImageChops, ImageStat, features,
)
from groq import Groq
from langchain_groq import ChatGroq
from huggingface_hub import InferenceClient
print("▶ Modules chargés.", flush=True)
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    ARABIC_OK = True
except ImportError:
    ARABIC_OK = False
    print("ATTENTION : pip install arabic-reshaper python-bidi (sinon l'arabe sera mal affiché)")
print("▶ Modules arabic_reshaper  chargés.", flush=True)
# ======================================================================
# CONFIGURATION
# ======================================================================
TRUSTED_WEB_SOURCES = {
    "FXStreet": "https://www.fxstreet.com/rss/news",
    "InvestingLive Macro": "https://investinglive.com/rss/news",
    "ActionForex Fundamental": "https://www.actionforex.com/category/fundamental-analysis/feed/",
}

# Modifiables par variables d'environnement (sans toucher au code)
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
# Pour une qualité maximale (plus lent, crédits HF plus consommés) :
#   HF_IMAGE_MODEL=black-forest-labs/FLUX.1-dev
HF_IMAGE_MODEL = os.getenv("HF_IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell")
GRAPH_URL = "https://graph.facebook.com/v26.0"

# Génération directement en portrait -> aucun recadrage destructif
IMG_W, IMG_H = 1024, 1280      # multiples de 16 (requis par FLUX)
OUT_W, OUT_H = 1080, 1350      # sortie finale 4:5 (Facebook / Instagram)

# Nombre de candidats générés ; le meilleur est sélectionné automatiquement
IMAGE_CANDIDATES = int(os.getenv("IMAGE_CANDIDATES", "2"))

# Contrôle de publication
MAX_POSTS_PER_RUN = int(os.getenv("MAX_POSTS_PER_RUN", "1"))
MAX_ATTEMPTS_PER_RUN = int(os.getenv("MAX_ATTEMPTS_PER_RUN", "3"))
MAX_NEWS_AGE_HOURS = int(os.getenv("MAX_NEWS_AGE_HOURS", "24"))
IG_CAPTION_LIMIT = 2200

# Identité de la page
BRAND_NAME = ("Money", ".impact")
BRAND_HANDLE = "@money.impact.today"
BRAND_ACCENT = (255, 196, 0)
# Police arabe grasse. Idéal : Cairo-Bold.ttf ou Tajawal-Bold.ttf dans le dossier ./fonts/
AR_FONT_ENV = os.getenv("AR_FONT_PATH", "")

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Logo de la page (PNG transparent conseillé), affiché en haut à gauche
LOGO_PATH = Path(os.getenv("LOGO_PATH", str(BASE_DIR / "imgs" / "logo.png")))
# Mettre LOGO_BACKDROP=1 si ton logo est sombre (ajoute un fond foncé derrière)
LOGO_BACKDROP = os.getenv("LOGO_BACKDROP", "0") == "1"

# Scènes de secours si le LLM échoue (mots-clés -> scène photo précise)
FALLBACK_SCENES = [
    (("gold", "xau", "bullion"),
     "Stack of gold bars on dark polished surface inside a bank vault, blurred safety deposit boxes behind"),
    (("oil", "crude", "wti", "brent", "opec"),
     "Oil refinery towers at dusk with glowing flare stacks, blurred pipelines and storage tanks in background"),
    (("bitcoin", "crypto", "btc", "ethereum"),
     "Single physical gold bitcoin coin standing on dark reflective glass, blurred server racks behind"),
    (("dollar", "usd", "fed ", "fomc", "powell", "treasury", "yield"),
     "Stack of US hundred-dollar bills on dark marble desk, blurred columns of a federal building behind"),
    (("euro", "eur", "ecb", "lagarde"),
     "Euro banknotes fanned out on dark stone surface, blurred European central bank facade behind"),
    (("yen", "jpy", "boj"),
     "Japanese yen banknotes and coins on dark lacquered surface, blurred Tokyo skyline at dusk behind"),
    (("pound", "gbp", "boe", "sterling"),
     "British pound notes on dark oak-free slate surface, blurred London financial district behind"),
    (("stock", "nasdaq", "s&p", "dow", "equities", "wall street"),
     "New York Stock Exchange facade with large columns and flags, dramatic evening light, blurred street below"),
]
DEFAULT_SCENE = (
    "Stack of gold bars on dark marble surface, blurred bank vault door in background"
)

STYLE_SUFFIX = (
    "warm cinematic side lighting, shot on Canon EOS R5, 85mm f/1.4 lens, "
    "shallow depth of field, creamy bokeh, tack-sharp focus on the hero subject, "
    "editorial photography, professional color grading in deep navy blue and warm gold, "
    "photorealistic, ultra detailed, 8k, hero subject in the upper 60% of the frame, lower 40% of the frame dark, smooth and uncluttered, "
    "no text, no letters, no logos, no people"
)


# ======================================================================
# GÉNÉRATION DE POST (image + publication)
# ======================================================================
class TextPost:
    def __init__(self):
        self.llm = ChatGroq(
            model_name=GROQ_MODEL,
            temperature=0.4,
            groq_api_key=os.getenv("TOKEN_AI_DESCRYB_IMG", ""),
        )

    # ------------------------------------------------------------------
    # 1) Prompt d'image
    # ------------------------------------------------------------------
    def donne_moi_prompt_images(self, news):
        system_prompt = (
            "You are an award-winning Reuters financial photojournalist and editorial "
            "art director. You write vivid, specific, physically realistic photo "
            "descriptions. You NEVER use vague terms like 'trading screens' or "
            "'financial charts'. You always name exact objects, materials, lighting "
            "and depth of field. You forbid abstract designs, 3D renders, neon, "
            "wireframes and futuristic shapes."
        )

        user_prompt = f"""
        FINANCIAL NEWS:
        "{(news or '')[:800]}"

        Identify the MAIN SUBJECT of this news and write ONE photorealistic editorial
        photo description that represents it visually.

        Subject mapping examples:
        - bond yields / Treasuries -> stack of Treasury bonds on a desk
        - gold -> gold bars in a vault
        - oil -> refinery at dusk or oil barrels
        - central bank -> central bank building facade
        - US dollar -> dollar banknotes
        - stock selloff / rally -> stock exchange facade
        - crypto -> physical coin on reflective surface

        Rules:
        1. The FIRST WORDS must name the exact physical hero subject.
        2. Hero subject sits in the UPPER 60% of the frame, sharp focus, slightly off-center.
        3. Background: relevant but blurred (bokeh). Specific lighting (e.g. golden
           hour side-light, tungsten vault light, cold blue dawn light).
        4. The lower 40% of the frame must be dark, smooth and uncluttered (a headline will be printed there).
        5. Palette: deep navy blue and warm gold.
        6. FORBIDDEN: people, faces, hands, text, letters, numbers, logos, watermarks,
           abstract shapes, 3D renders, neon, computer screens with charts.
        7. Max 60 words. End with: photorealistic, 85mm lens, sharp focus.

        OUTPUT ONLY this format:
        POSITIVE PROMPT:
        [description]

        NEGATIVE PROMPT:
        people, faces, hands, text, letters, numbers, logos, watermark, abstract,
        3d render, neon, cartoon, illustration, painting, blurry, low quality
        """

        try:
            return self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])
        except Exception as e:
            print(f"Erreur LLM prompt generation: {e}")

            class Fallback:
                content = ""

            return Fallback()

    @staticmethod
    def _parse_prompts(text):
        text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I)
        pos = re.search(r"POSITIVE PROMPT:\s*(.*?)\s*NEGATIVE PROMPT:", text, re.S | re.I)
        neg = re.search(r"NEGATIVE PROMPT:\s*(.*)", text, re.S | re.I)
        if not pos or not neg:
            return None, None
        positive = pos.group(1).strip().strip('"').strip("[]")
        return positive, neg.group(1).strip()

    @staticmethod
    def _fallback_scene(news):
        low = (news or "").lower()
        for keywords, scene in FALLBACK_SCENES:
            if any(k in low for k in keywords):
                return scene
        return DEFAULT_SCENE

    def _build_prompt(self, news):
        text = self.donne_moi_prompt_images(news).content
        positive, _ = self._parse_prompts(text)
        if not positive or len(positive.split()) < 6:
            positive = self._fallback_scene(news)
        # FLUX ignore le negative prompt : les interdits sont dans le positif
        positive = " ".join(positive.split()[:70])
        return f"{positive}, {STYLE_SUFFIX}"

    # ------------------------------------------------------------------
    # 2) Génération d'image (HF FLUX -> fallback Pollinations)
    # ------------------------------------------------------------------
    def _image_hf(self, prompt, seed=None):
        token = os.getenv("HF_TOKEN", "")
        if not token:
            print("HF_TOKEN absent : passage au fallback Pollinations.")
            return None

        model = HF_IMAGE_MODEL
        is_schnell = "schnell" in model.lower()
        # schnell : distillé pour 1-4 steps, ignore la guidance
        # dev : 28-35 steps, guidance ~3.5
        params = {"width": IMG_W, "height": IMG_H,
                  "num_inference_steps": 4 if is_schnell else 30}
        if not is_schnell:
            params["guidance_scale"] = 3.5
        if seed is not None:
            params["seed"] = seed

        client = InferenceClient(api_key=token)

        # Dégradation progressive des paramètres si l'API en refuse certains
        attempts = [
            params,
            {k: v for k, v in params.items() if k != "seed"},
            {"width": IMG_W, "height": IMG_H},
            {},
        ]
        for i, p in enumerate(attempts):
            try:
                image = client.text_to_image(prompt, model=model, **p)
                return image.convert("RGB")
            except Exception as e:
                print(f"HF essai {i + 1} échoué : {e}")
                time.sleep(2)
        return None

    def _image_pollinations(self, prompt, seed=None):
        seed = seed if seed is not None else int(time.time())
        url = (
            "https://image.pollinations.ai/prompt/"
            + urllib.parse.quote(prompt)
            + f"?model=flux&width={IMG_W}&height={IMG_H}&nologo=true&seed={seed}&enhance=false"
        )
        try:
            r = requests.get(url, timeout=180)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            # Retire un éventuel bandeau/filigrane en bas
            w, h = img.size
            return img.crop((0, 0, w, int(h * 0.94)))
        except Exception as e:
            print(f"Pollinations indisponible : {e}")
            return None

    def _generate_one(self, prompt, seed):
        return self._image_hf(prompt, seed) or self._image_pollinations(prompt, seed)

    # ------------------------------------------------------------------
    # 3) Sélection du meilleur candidat
    # ------------------------------------------------------------------
    @staticmethod
    def _score_image(img):
        """Score = netteté + contraste, pénalise images trop sombres/claires."""
        small = img.resize((512, int(512 * img.height / img.width)))
        gray = small.convert("L")
        edges = ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES))
        sharp = edges.var[0] ** 0.5
        contrast = ImageStat.Stat(gray).stddev[0]
        mean = ImageStat.Stat(gray).mean[0]
        penalty = 0
        if mean < 45:
            penalty = (45 - mean) * 1.5
        elif mean > 200:
            penalty = (mean - 200) * 1.5
        return sharp * 1.0 + contrast * 0.6 - penalty

    def _best_image(self, prompt):
        candidates = []
        for i in range(max(1, IMAGE_CANDIDATES)):
            seed = int(time.time() * 1000) % 2_000_000_000 + i * 7919
            img = self._generate_one(prompt, seed)
            if img is not None and img.size[0] >= 512:
                score = self._score_image(img)
                print(f"Candidat {i + 1} : score {score:.1f}")
                candidates.append((score, img))
        if not candidates:
            return None
        return max(candidates, key=lambda c: c[0])[1]

    # ------------------------------------------------------------------
    # 4) Post-traitement (netteté, couleur, vignette, grain)
    # ------------------------------------------------------------------
    @staticmethod
    def _enhance_image(img):
        img = img.filter(ImageFilter.UnsharpMask(radius=1.6, percent=60, threshold=3))
        img = ImageEnhance.Contrast(img).enhance(1.07)
        img = ImageEnhance.Color(img).enhance(1.05)
        img = ImageEnhance.Sharpness(img).enhance(1.15)

        w, h = img.size
        # Vignette douce
        mask = Image.radial_gradient("L").resize((w, h), Image.LANCZOS)
        mask = mask.point(lambda p: int(max(0, p - 90) * 0.55))
        img = Image.composite(Image.new("RGB", (w, h), (3, 8, 22)), img, mask)

        # Grain très léger : casse l'aspect "lisse" typique des images IA
        noise = Image.effect_noise((w, h), 14).convert("RGB")
        img = Image.blend(img, ImageChops.soft_light(img, noise), 0.30)
        return img

    @staticmethod
    def _load_font(size):
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        ]
        for chemin in candidates:
            if os.path.exists(chemin):
                return ImageFont.truetype(chemin, size)
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()

    # ------------------------------------------------------------------
    # Titre arabe + branding style "chaîne d'info" (comme Al Arabiya)
    # ------------------------------------------------------------------
    # L'affichage de l'arabe dépend de la version de Pillow (Raqm ou non). Au premier
    # appel, le script teste le rendu réel et choisit un mode qui fonctionne :
    #   "raqm"   : texte brut, Raqm/HarfBuzz fait le shaping + le sens RTL
    #   "bidi"   : arabic_reshaper + python-bidi, moteur BASIC
    #   "manual" : arabic_reshaper + inversion des lettres, moteur BASIC
    # Dans tous les cas, l'ordre des mots est géré ici (un mot = un dessin), donc
    # il ne peut plus être inversé par le moteur de texte.
    _ar_mode = None
    _AR_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")
    _NUM_RE = re.compile(r"([0-9\u0660-\u0669][0-9\u0660-\u0669.,%]*)")

    @staticmethod
    def _ar_font_candidates():
        return [
            AR_FONT_ENV,
            str(BASE_DIR / "fonts" / "Cairo-Bold.ttf"),
            str(BASE_DIR / "fonts" / "Tajawal-Bold.ttf"),
            str(BASE_DIR / "fonts" / "NotoSansArabic-Bold.ttf"),
            "/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf",
            "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        ]

    @classmethod
    def _open_ar_font(cls, size, mode):
        layout_cls = getattr(ImageFont, "Layout", None)
        layout = None
        if layout_cls is not None:
            layout = layout_cls.RAQM if mode == "raqm" else layout_cls.BASIC
        for chemin in cls._ar_font_candidates():
            if chemin and os.path.exists(chemin):
                try:
                    if layout is not None:
                        return ImageFont.truetype(chemin, size, layout_engine=layout)
                    return ImageFont.truetype(chemin, size)
                except Exception:
                    continue
        raise RuntimeError("Aucune police arabe trouvée")

    @classmethod
    def _load_arabic_font(cls, size):
        mode = cls._detect_ar_mode()
        try:
            return cls._open_ar_font(size, mode)
        except Exception:
            print("ATTENTION : aucune police arabe trouvée (mets Cairo-Bold.ttf dans ./fonts/)")
            return TextPost._load_font(size)

    @classmethod
    def _ar_word_visual(cls, word, mode):
        """Un mot arabe -> chaîne à dessiner (gauche vers droite)."""
        if mode == "raqm" or not ARABIC_OK:
            return word
        shaped = arabic_reshaper.reshape(word)
        if mode == "bidi":
            return get_display(shaped, base_dir="R")
        # "manual" : lettres inversées, les chiffres gardent leur ordre
        parts = cls._NUM_RE.split(shaped)
        pieces = [p if i % 2 else p[::-1] for i, p in enumerate(parts)]
        return "".join(reversed(pieces))

    @classmethod
    def _ar_selftest(cls, mode):
        """Dessine 'اب' : l'alef (grande barre) doit être à DROITE du bêh."""
        font = cls._open_ar_font(100, mode)
        kw = {"direction": "rtl", "language": "ar"} if mode == "raqm" else {}
        img = Image.new("L", (400, 260), 0)
        ImageDraw.Draw(img).text(
            (60, 40), cls._ar_word_visual("اب", mode), font=font, fill=255, **kw
        )
        bbox = img.getbbox()
        if not bbox:
            return False
        x0, y0, x1, y1 = bbox
        strip = max(4, (x1 - x0) // 5)
        left = img.crop((x0, y0, x0 + strip, y1)).getbbox()
        right = img.crop((x1 - strip, y0, x1, y1)).getbbox()
        if not left or not right:
            return False
        return right[1] < left[1] - 5

    @classmethod
    def _detect_ar_mode(cls):
        if cls._ar_mode:
            return cls._ar_mode
        candidates = []
        try:
            if features.check("raqm") and features.check("fribidi") and features.check("harfbuzz"):
                candidates.append("raqm")
        except Exception:
            pass
        if ARABIC_OK:
            candidates += ["bidi", "manual"]

        chosen = None
        for mode in candidates:
            try:
                if cls._ar_selftest(mode):
                    chosen = mode
                    break
                print(f"Mode arabe '{mode}' : rendu incorrect, essai suivant.")
            except Exception as e:
                print(f"Mode arabe '{mode}' indisponible : {e}")
        if chosen is None:
            chosen = "bidi" if ARABIC_OK else "plain"
            print("ATTENTION : aucun mode arabe validé, l'affichage peut être incorrect.")
        cls._ar_mode = chosen
        print(f"Mode d'écriture arabe retenu : {chosen}")
        return chosen

    @classmethod
    def _ar_runs(cls, text):
        """Découpe en blocs : mots arabes seuls, mots latins/nombres regroupés (ordre LTR)."""
        runs = []
        for tok in text.split():
            is_ar = bool(cls._AR_RE.search(tok))
            if runs and not is_ar and not runs[-1][1]:
                runs[-1] = (runs[-1][0] + " " + tok, False)
            else:
                runs.append((tok, is_ar))
        return runs

    @classmethod
    def _run_args(cls, run, is_ar):
        mode = cls._detect_ar_mode()
        if is_ar:
            kw = {"direction": "rtl", "language": "ar"} if mode == "raqm" else {}
            return cls._ar_word_visual(run, mode), kw
        return run, ({"direction": "ltr"} if mode == "raqm" else {})

    def _line_width(self, draw, text, font):
        runs = self._ar_runs(text)
        space = draw.textlength(" ", font=font)
        total = 0
        for run, is_ar in runs:
            s, kw = self._run_args(run, is_ar)
            total += draw.textlength(s, font=font, **kw)
        return total + space * max(0, len(runs) - 1)

    def _draw_rtl_line(self, draw, right_x, y, text, font, fill):
        """Dessine une ligne arabe alignée à droite : 1er mot à droite, suivants vers la gauche."""
        space = draw.textlength(" ", font=font)
        x = right_x
        for run, is_ar in self._ar_runs(text):
            s, kw = self._run_args(run, is_ar)
            rw = draw.textlength(s, font=font, **kw)
            x -= rw
            draw.text((x, y), s, font=font, fill=fill, **kw)
            x -= space

    def _wrap(self, draw, text, font, max_w):
        lines, cur = [], []
        for word in text.split():
            test = " ".join(cur + [word])
            if cur and self._line_width(draw, test, font) > max_w:
                lines.append(" ".join(cur))
                cur = [word]
            else:
                cur.append(word)
        if cur:
            lines.append(" ".join(cur))
        return lines

    def _fit_headline(self, draw, text, max_w, max_h, w):
        """Réduit la taille de police jusqu'à ce que le titre tienne (max 4 lignes)."""
        size, min_size, max_lines = int(w * 0.078), int(w * 0.05), 4
        while True:
            font = self._load_arabic_font(size)
            lines = self._wrap(draw, text, font, max_w)
            line_h = int(size * 1.4)
            fits = len(lines) <= max_lines and len(lines) * line_h <= max_h
            if fits or size <= min_size:
                break
            size -= 2
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            lines[-1] = lines[-1].rstrip(" ،,.") + "…"
        return font, lines, line_h

    @staticmethod
    def _load_logo(w, h):
        """Charge imgs/logo.png (PNG transparent conseillé). None si absent."""
        try:
            if not LOGO_PATH.is_file():
                print(f"Logo introuvable ({LOGO_PATH}) : logo texte utilisé à la place.")
                return None
            logo = Image.open(LOGO_PATH).convert("RGBA")
            ratio = min(int(w * 0.24) / logo.width, int(h * 0.11) / logo.height)
            size = (max(1, int(logo.width * ratio)), max(1, int(logo.height * ratio)))
            return logo.resize(size, Image.LANCZOS)
        except Exception as e:
            print(f"Logo illisible : {e}")
            return None

    def _add_branding(self, img, headline=None):
        img = img.convert("RGBA")
        w, h = img.size
        margin = int(w * 0.07)
        has_title = bool(headline and headline.strip())
        logo = self._load_logo(w, h)

        # ---- 1) Dégradés (bas fort pour le titre, haut léger pour le logo) ----
        grad = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        gd = ImageDraw.Draw(grad)
        bottom_h = int(h * (0.50 if has_title else 0.28))
        for i in range(bottom_h):
            a = int(235 * (i / bottom_h) ** 1.5)
            gd.line([(0, h - bottom_h + i), (w, h - bottom_h + i)], fill=(4, 10, 28, a))
        top_h = int(h * 0.16)
        for i in range(top_h):
            a = int(160 * (1 - i / top_h) ** 1.6)
            gd.line([(0, i), (w, i)], fill=(4, 10, 28, a))
        img = Image.alpha_composite(img, grad)
        probe = ImageDraw.Draw(img)

        # ---- 2) Géométrie ----
        lx, ly = int(w * 0.04), int(h * 0.03)
        part_a, part_b = BRAND_NAME

        if logo is not None:
            logo_box = [lx, ly, lx + logo.width, ly + logo.height]
        else:  # secours : logo en texte
            f_logo = self._load_font(int(w * 0.046))
            wa = probe.textlength(part_a, font=f_logo)
            wb = probe.textlength(part_b, font=f_logo)
            logo_bb = f_logo.getbbox(part_a + part_b)
            pad_x, pad_y = int(w * 0.022), int(w * 0.013)
            logo_box = [lx, ly, lx + wa + wb + 2 * pad_x, ly + logo_bb[3] + 2 * pad_y]

        f_sub = self._load_font(int(w * 0.027))
        handle_y = h - int(h * 0.055)

        title_data = None
        if has_title:
            bottom = handle_y - int(h * 0.028)
            font, lines, line_h = self._fit_headline(
                probe, headline.strip(), w - 2 * margin, int(h * 0.30), w
            )
            top = bottom - len(lines) * line_h

            # Étiquette = nom de la page (Money.impact)
            f_tag = self._load_font(int(w * 0.038))
            ta = probe.textlength(part_a, font=f_tag)
            tb_w = probe.textlength(part_b, font=f_tag)
            tbb = f_tag.getbbox(part_a + part_b)
            tp_x, tp_y = int(w * 0.022), int(w * 0.011)
            bar_w = max(8, int(w * 0.009))
            box_w = int(ta + tb_w) + 2 * tp_x + bar_w
            box_h = tbb[3] + 2 * tp_y
            tag_right = w - margin
            gap = int(h * 0.022)
            tag_box = [tag_right - box_w, top - gap - box_h, tag_right, top - gap]
            title_data = (font, lines, line_h, top, f_tag, ta, tag_box, tp_x, tp_y, bar_w)

        # ---- 3) Ombres floutées ----
        soft = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        sd = ImageDraw.Draw(soft)
        if logo is not None:
            shadow = Image.new("RGBA", logo.size, (0, 0, 0, 0))
            shadow.putalpha(logo.getchannel("A").point(lambda a: int(a * 0.75)))
            soft.alpha_composite(shadow, (lx + 3, ly + 4))
        if title_data:
            font, lines, line_h, top = title_data[:4]
            for i, line in enumerate(lines):
                self._draw_rtl_line(sd, w - margin, top + i * line_h + 4, line, font,
                                    (0, 0, 0, 220))
        soft = soft.filter(ImageFilter.GaussianBlur(5))
        img = Image.alpha_composite(img, soft)

        # ---- 4) Fond translucide net (logo texte ou fond du logo image) ----
        crisp = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        cd = ImageDraw.Draw(crisp)
        if logo is None:
            cd.rectangle(logo_box, fill=(4, 10, 28, 190), outline=(255, 255, 255, 220), width=2)
        elif LOGO_BACKDROP:
            p = int(w * 0.012)
            cd.rectangle([logo_box[0] - p, logo_box[1] - p, logo_box[2] + p, logo_box[3] + p],
                         fill=(4, 10, 28, 190), outline=(255, 255, 255, 220), width=2)
        img = Image.alpha_composite(img, crisp)

        # ---- 5) Éléments opaques ----
        if logo is not None:
            img.alpha_composite(logo, (lx, ly))
        draw = ImageDraw.Draw(img)

        if logo is None:
            draw.text((lx + pad_x, ly + pad_y), part_a, font=f_logo, fill=(255, 255, 255, 255))
            draw.text((lx + pad_x + wa, ly + pad_y), part_b, font=f_logo,
                      fill=BRAND_ACCENT + (255,))
            draw.rectangle([logo_box[0], logo_box[3] - 5, logo_box[2], logo_box[3]],
                           fill=BRAND_ACCENT + (255,))

        if title_data:
            font, lines, line_h, top, f_tag, ta, tag_box, tp_x, tp_y, bar_w = title_data
            draw.rectangle(tag_box, fill=(255, 255, 255, 255))
            draw.rectangle([tag_box[2] - bar_w, tag_box[1], tag_box[2], tag_box[3]],
                           fill=BRAND_ACCENT + (255,))
            draw.text((tag_box[0] + tp_x, tag_box[1] + tp_y), part_a, font=f_tag,
                      fill=(5, 12, 30, 255))
            draw.text((tag_box[0] + tp_x + ta, tag_box[1] + tp_y), part_b, font=f_tag,
                      fill=(214, 148, 0, 255))
            for i, line in enumerate(lines):
                self._draw_rtl_line(draw, w - margin, top + i * line_h, line, font,
                                    (255, 255, 255, 255))

        draw.text((int(w * 0.04), handle_y), BRAND_HANDLE, font=f_sub,
                  fill=(225, 230, 240, 235))

        return img.convert("RGB")

    # ------------------------------------------------------------------
    # 5) Pipeline image complet
    # ------------------------------------------------------------------
    def generer_image(self, news_original=None, headline=None):
        print("Génération de l'image...")
        prompt = self._build_prompt(news_original)
        print("Prompt image :", prompt)

        img = self._best_image(prompt)
        if img is None:
            raise RuntimeError("Aucune image générée : publication annulée.")

        # Même ratio (4:5) -> recadrage minimal, pas de perte de détail
        img = ImageOps.fit(img, (OUT_W, OUT_H), Image.LANCZOS, centering=(0.5, 0.4))
        img = self._enhance_image(img)
        img = self._add_branding(img, headline)

        output_png = OUTPUT_DIR / "img.png"
        img.save(output_png)
        # JPEG haute qualité : utilisé pour Facebook et Instagram (léger, accepté partout)
        output_jpg = OUTPUT_DIR / "img.jpg"
        img.save(output_jpg, format="JPEG", quality=95, optimize=True, subsampling=0)
        print(f"Image sauvegardée : {output_jpg}")
        return output_jpg

    # ------------------------------------------------------------------
    # 6) Publication Facebook + Instagram
    # ------------------------------------------------------------------
    @staticmethod
    def _json_or_text(response):
        if response.headers.get("content-type", "").startswith("application/json"):
            return response.json()
        return {"message": response.text[:1000]}

    def publier_texte_partout(self, news_original, news_translited, headline=None):
        resultats = {"image": None, "facebook": False, "instagram": False}

        try:
            chemin = Path(self.generer_image(news_original=news_original, headline=headline))
            if not chemin.is_file():
                raise FileNotFoundError(f"Image générée introuvable : {chemin.resolve()}")
            resultats["image"] = str(chemin.resolve())

            page_id = os.getenv("PAGE_ID", "")
            token = os.getenv("PAGE_ACCESS_TOKEN", "")
            ig_id = os.getenv("INSTAGRAM_BUSINESS_ID", "")

            cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME", "")
            cloud_key = os.getenv("CLOUDINARY_API_KEY", "")
            cloud_secret = os.getenv("CLOUDINARY_API_SECRET", "")

            if not token:
                raise ValueError("PAGE_ACCESS_TOKEN est absent.")

            # ---------------- Facebook ----------------
            if page_id:
                try:
                    with chemin.open("rb") as fichier:
                        response = requests.post(
                            f"{GRAPH_URL}/{page_id}/photos",
                            data={"message": news_translited, "access_token": token},
                            files={"source": (chemin.name, fichier, "image/jpeg")},
                            timeout=120,
                        )
                    data_fb = self._json_or_text(response)
                    if response.ok and "error" not in data_fb:
                        resultats["facebook"] = True
                        print("Facebook : publication réussie.")
                    else:
                        print("Facebook : échec :", data_fb)
                except Exception as erreur:
                    print("Erreur Facebook :", erreur)

            # ---------------- Instagram ----------------
            if all([cloud_name, cloud_key, cloud_secret]) and ig_id:
                try:
                    cloudinary.config(
                        cloud_name=cloud_name,
                        api_key=cloud_key,
                        api_secret=cloud_secret,
                        secure=True,
                    )
                    upload = cloudinary.uploader.upload(
                        str(chemin), resource_type="image", folder="actualites_financieres"
                    )
                    image_url = upload.get("secure_url")

                    if image_url and image_url.startswith("https://"):
                        caption = news_translited[:IG_CAPTION_LIMIT]
                        creation_response = requests.post(
                            f"{GRAPH_URL}/{ig_id}/media",
                            data={"image_url": image_url, "caption": caption,
                                  "access_token": token},
                            timeout=90,
                        )
                        creation_data = self._json_or_text(creation_response)

                        if (creation_response.ok and "error" not in creation_data
                                and creation_data.get("id")):
                            creation_id = creation_data["id"]
                            statut = None
                            for _ in range(12):
                                statut_response = requests.get(
                                    f"{GRAPH_URL}/{creation_id}",
                                    params={"fields": "status_code", "access_token": token},
                                    timeout=30,
                                )
                                statut = statut_response.json().get("status_code")
                                if statut in ("FINISHED", "ERROR"):
                                    break
                                time.sleep(5)

                            if statut == "FINISHED":
                                pub_response = requests.post(
                                    f"{GRAPH_URL}/{ig_id}/media_publish",
                                    data={"creation_id": creation_id, "access_token": token},
                                    timeout=90,
                                )
                                pub_data = self._json_or_text(pub_response)
                                if pub_response.ok and "error" not in pub_data:
                                    resultats["instagram"] = True
                                    print("Instagram : publication réussie.")
                                else:
                                    print("Instagram : échec publication :", pub_data)
                            else:
                                print("Instagram : statut du média :", statut)
                        else:
                            print("Instagram : échec création média :", creation_data)
                except Exception as erreur:
                    print("Erreur Instagram :", erreur)

        except Exception as erreur:
            print("Erreur de génération ou de préparation :", erreur)

        return resultats


# ======================================================================
# EXTRACTION DES NEWS + TRADUCTION
# ======================================================================
class Extract_post:
    def __init__(self):
        self.model_groq_name = GROQ_MODEL
        self.client = Groq(api_key=os.getenv("TOKEN_AI_DESCRYB_IMG", ""))
        self.TRUSTED_WEB_SOURCES = TRUSTED_WEB_SOURCES

    def clean_html(self, text):
        if not text:
            return ""
        soup = BeautifulSoup(text, "html.parser")
        for element in soup(["script", "style"]):
            element.decompose()
        return soup.get_text(separator=" ", strip=True)

    def get_publication_date(self, entry):
        date_struct = entry.get("published_parsed") or entry.get("updated_parsed")
        if not date_struct:
            return None, None
        date_utc = datetime(*date_struct[:6], tzinfo=timezone.utc)
        return date_utc.strftime("%Y-%m-%d %H:%M"), date_utc

    def fetch_all_news(self):
        articles = []
        seen_links = set()

        for source_name, feed_url in self.TRUSTED_WEB_SOURCES.items():
            try:
                feed = feedparser.parse(feed_url)
                if getattr(feed, "status", 200) >= 400:
                    continue

                for entry in feed.entries:
                    link = entry.get("link", "").strip()
                    title = entry.get("title", "").strip()

                    if not title or not link or link in seen_links:
                        continue

                    raw_text = (
                        entry.get("content", [{}])[0].get("value", "")
                        if entry.get("content")
                        else entry.get("summary", "")
                    )

                    clean_text = self.clean_html(raw_text)
                    clean_text = re.sub(r"\s*The post .*? appeared first on .*$", "",
                                        clean_text, flags=re.S)
                    clean_text = clean_text.replace("[...]", "…").strip()
                    publication_text, publication_datetime = self.get_publication_date(entry)

                    if publication_text is None:
                        continue

                    seen_links.add(link)
                    articles.append({
                        "source": source_name,
                        "title": self.clean_html(title),
                        "date": publication_text,
                        "continue": clean_text,
                        "datetime": publication_datetime,
                    })
            except Exception as error:
                print(f"Erreur pour {source_name}: {error}")

        articles.sort(key=lambda item: item["datetime"], reverse=True)
        return articles

    def tradication_of_text(self, text):
        prompt = """
        You are a professional Arabic translator specializing in financial news.
        Translate into natural Modern Standard Arabic. Preserve financial meaning and facts.
        Return ONLY a valid JSON object with two fields:
        {"caption": "full translation of the text",
         "headline": "short punchy Arabic news headline"}
        The headline: 8 to 15 words, newsroom style, captures the key fact (and the
        market impact if any), no emojis, no hashtags, no quotation marks, no source name.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_groq_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"SOURCE TEXT TO TRANSLATE:\n{text}"},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Erreur lors de l'appel Groq : {e}")
            return None


# ======================================================================
# MAIN
# ======================================================================
def charger_historique(fichier_json):
    if os.path.exists(fichier_json):
        try:
            with open(fichier_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    with open(fichier_json, "w", encoding="utf-8") as f:
        json.dump({}, f, ensure_ascii=False, indent=4)
    return {}


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fichier_json = os.path.join(base_dir, "historique_posts.json")

    extract_post = Extract_post()
    text_post = TextPost()
    historique = charger_historique(fichier_json)

    all_news = extract_post.fetch_all_news()
    now = datetime.now(timezone.utc)
    publies = 0
    tentatives = 0

    for news in all_news:
        if publies >= MAX_POSTS_PER_RUN or tentatives >= MAX_ATTEMPTS_PER_RUN:
            break

        vrai_source = news["source"]
        title = news["title"]
        content = news["continue"]
        date = news["date"]
        post_id = f"{vrai_source}_{title}_{date}"

        if post_id in historique:
            print(f"Déjà traité : {title}")
            continue

        age_h = (now - news["datetime"]).total_seconds() / 3600
        if age_h > MAX_NEWS_AGE_HOURS:
            print(f"Trop ancienne ({age_h:.0f} h) : {title}")
            continue

        text_original = f"{title}\n{content}\n\nsource: {vrai_source}, date: {date}"
        text_translated = extract_post.tradication_of_text(text_original)

        # Correction du bug "None" publié comme légende
        if not isinstance(text_translated, dict):
            print("Traduction échouée, on passe.")
            continue
        caption = str(text_translated.get("caption") or "").strip()
        if not caption:
            print("Traduction vide, on passe.")
            continue

        headline = str(text_translated.get("headline") or "").strip()
        if not headline:  # secours : début de la légende
            headline = " ".join(caption.split()[:14])

        tentatives += 1
        try:
            resultat = text_post.publier_texte_partout(text_original, caption, headline)

            if resultat and (resultat.get("facebook") or resultat.get("instagram")):
                historique[post_id] = {"source": vrai_source, "titre": title, "date": date}
                with open(fichier_json, "w", encoding="utf-8") as f:
                    json.dump(historique, f, ensure_ascii=False, indent=4)
                publies += 1
                print(f"Historique sauvegardé : {title}")
            else:
                print(f"Publication non confirmée : {title}.")
        except Exception as e:
            print(f"Erreur publication de '{title}' : {e}")
            continue

    print(f"Fin de cette exécution ({publies} publication(s)).")


if __name__ == "__main__":
    main()
