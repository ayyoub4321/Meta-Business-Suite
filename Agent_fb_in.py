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
    ImageFilter, ImageChops, ImageStat,
)
from groq import Groq
from langchain_groq import ChatGroq
from huggingface_hub import InferenceClient

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
MAX_POSTS_PER_RUN = int(os.getenv("MAX_POSTS_PER_RUN", "15"))
MAX_ATTEMPTS_PER_RUN = int(os.getenv("MAX_ATTEMPTS_PER_RUN", "15"))
MAX_NEWS_AGE_HOURS = int(os.getenv("MAX_NEWS_AGE_HOURS", "200"))
IG_CAPTION_LIMIT = 2200

# Identité de la page
BRAND_NAME = ("Money", ".impact")
BRAND_HANDLE = "@money.impact.today"
BRAND_ACCENT = (255, 196, 0)

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    "photorealistic, ultra detailed, 8k, bottom fifth of the frame dark and uncluttered, "
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
        2. Hero subject occupies 40-60% of the frame, sharp focus, slightly off-center.
        3. Background: relevant but blurred (bokeh). Specific lighting (e.g. golden
           hour side-light, tungsten vault light, cold blue dawn light).
        4. Bottom 20% of the frame must be dark and uncluttered.
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

    def _add_branding(self, img):
        """Branding net (jamais déformé par l'IA), transparence réellement gérée."""
        img = img.convert("RGBA")
        w, h = img.size

        # Dégradé doux en bas
        grad_h = int(h * 0.28)
        grad = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        gd = ImageDraw.Draw(grad)
        for i in range(grad_h):
            alpha = int(215 * (i / grad_h) ** 1.7)
            gd.line([(0, h - grad_h + i), (w, h - grad_h + i)], fill=(4, 10, 28, alpha))
        img = Image.alpha_composite(img, grad)

        font_main = self._load_font(int(w * 0.062))
        font_sub = self._load_font(int(w * 0.027))
        probe = ImageDraw.Draw(img)

        part_a, part_b = BRAND_NAME
        wa = probe.textlength(part_a, font=font_main)
        wb = probe.textlength(part_b, font=font_main)
        x = (w - (wa + wb)) / 2
        y_main = h - int(h * 0.108)

        line_w = int(w * 0.14)
        y_line = y_main - int(h * 0.016)
        line_box = [(w - line_w) / 2, y_line, (w + line_w) / 2, y_line + 5]

        # Couche ombre + glow (floutée), composée avec vraie transparence
        fx = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        fd = ImageDraw.Draw(fx)
        fd.text((x + 3, y_main + 3), part_a, font=font_main, fill=(0, 0, 0, 200))
        fd.text((x + wa + 3, y_main + 3), part_b, font=font_main, fill=(0, 0, 0, 200))
        fx = fx.filter(ImageFilter.GaussianBlur(4))
        glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(glow).rectangle(line_box, fill=BRAND_ACCENT + (170,))
        glow = glow.filter(ImageFilter.GaussianBlur(7))
        img = Image.alpha_composite(img, fx)
        img = Image.alpha_composite(img, glow)

        # Éléments nets
        draw = ImageDraw.Draw(img)
        draw.rectangle(line_box, fill=BRAND_ACCENT + (255,))
        draw.text((x, y_main), part_a, font=font_main, fill=(255, 255, 255, 255))
        draw.text((x + wa, y_main), part_b, font=font_main, fill=BRAND_ACCENT + (255,))

        wh = draw.textlength(BRAND_HANDLE, font=font_sub)
        y_sub = y_main + int(w * 0.062) + int(h * 0.012)
        draw.text(((w - wh) / 2, y_sub), BRAND_HANDLE, font=font_sub,
                  fill=(225, 230, 240, 235))

        return img.convert("RGB")

    # ------------------------------------------------------------------
    # 5) Pipeline image complet
    # ------------------------------------------------------------------
    def generer_image(self, news_original=None):
        print("Génération de l'image...")
        prompt = self._build_prompt(news_original)
        print("Prompt image :", prompt)

        img = self._best_image(prompt)
        if img is None:
            raise RuntimeError("Aucune image générée : publication annulée.")

        # Même ratio (4:5) -> recadrage minimal, pas de perte de détail
        img = ImageOps.fit(img, (OUT_W, OUT_H), Image.LANCZOS, centering=(0.5, 0.4))
        img = self._enhance_image(img)
        img = self._add_branding(img)

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

    def publier_texte_partout(self, news_original, news_translited):
        resultats = {"image": None, "facebook": False, "instagram": False}

        try:
            chemin = Path(self.generer_image(news_original=news_original))
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
        Return ONLY a valid JSON object:
        {"caption": "translated_text_here"}
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

        tentatives += 1
        try:
            resultat = text_post.publier_texte_partout(text_original, caption)

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
