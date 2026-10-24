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
from PIL import Image, ImageDraw, ImageFont, ImageOps
from groq import Groq
from langchain_groq import ChatGroq
from huggingface_hub import InferenceClient

TRUSTED_WEB_SOURCES = {
    "FXStreet": "https://www.fxstreet.com/rss/news",
    "InvestingLive Macro": "https://investinglive.com/rss/news",
    "ActionForex Fundamental": "https://www.actionforex.com/category/fundamental-analysis/feed/",
}

GROQ_MODEL = "qwen/qwen3.8-27b"
HF_IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"
GRAPH_URL = "https://graph.facebook.com/v26.0"

# Génération en portrait (rendu optimal sur Facebook et Instagram)
IMG_W, IMG_H = 1024, 1280      # taille demandée au modèle
OUT_W, OUT_H = 1080, 1350      # taille finale publiée (ratio 4:5)

# Identité de la page
BRAND_NAME = ("Money", ".impact")
BRAND_HANDLE = "@money.impact.today"
BRAND_ACCENT = (255, 196, 0)   # or

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class TextPost:
    def __init__(self):
        self.llm = ChatGroq(
            model_name=GROQ_MODEL,
            temperature=0.4,
            groq_api_key=os.getenv("TOKEN_AI_DESCRYB_IMG", ""),
        )

    # ------------------------------------------------------------------
    # 1) Prompt d'image à partir de la news
    # ------------------------------------------------------------------
    def donne_moi_prompt_images(self, news):
        system_prompt = (
            "You are an uncompromising, world-class Financial Photojournalist and "
            "Editorial Art Director. Your sole task is to write realistic visual "
            "prompts for real-world financial news photos. You strictly forbid "
            "abstract designs, 3D wireframes, glowing neon elements and generic "
            "futuristic shapes."
        )

        user_prompt = f"""
        FINANCIAL NEWS:
        "{(news or '')[:600]}"

        Write ONE photorealistic editorial photo description that visually represents
        the MAIN SUBJECT of this news (e.g. bond yields falling -> Treasury bonds and a
        bond trading desk; tech selloff -> trading screens with red candles and a
        stock exchange floor; gold -> gold bars; oil -> refinery or oil barrels;
        central bank -> central bank building; US dollar -> dollar banknotes).
        The first words must name that main subject. Max 50 words.
        No text, no letters, no logos, no people, no faces, no abstract shapes,
        no leather, no wood, no generic office props.
        Composition: one strong hero subject in the foreground, relevant blurred background,
        deep blue and gold color palette, bottom 20% of the frame darker and uncluttered.
        End with: photorealistic, 85mm lens, natural cinematic lighting, sharp focus.

        OUTPUT ONLY this format:
        POSITIVE PROMPT:
        [description]

        NEGATIVE PROMPT:
        abstract, 3d render, neon, text, letters, numbers, watermark, human, face, blurry, cartoon
        """

        try:
            return self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])
        except Exception as e:
            print(f"Erreur LLM prompt generation: {e}")

            class Fallback:
                content = (
                    "POSITIVE PROMPT:\nstock market trading floor screens with green and red "
                    "candlestick charts, stacks of banknotes, editorial photo, 85mm, sharp focus\n"
                    "NEGATIVE PROMPT:\nabstract, 3d render, neon, text, letters, human, face, "
                    "watermark, blurry"
                )

            return Fallback()

    @staticmethod
    def _parse_prompts(text):
        text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I)
        pos = re.search(r"POSITIVE PROMPT:\s*(.*?)\s*NEGATIVE PROMPT:", text, re.S | re.I)
        neg = re.search(r"NEGATIVE PROMPT:\s*(.*)", text, re.S | re.I)
        if not pos or not neg:
            return None, None
        return pos.group(1).strip(), neg.group(1).strip()

    # ------------------------------------------------------------------
    # 2) Génération de l'image via API (rapide, pas de GPU nécessaire)
    # ------------------------------------------------------------------
    def _image_hf(self, prompt):
        token = os.getenv("HF_TOKEN", "")
        if not token:
            print("HF_TOKEN absent : passage au fallback Pollinations.")
            return None
        try:
            client = InferenceClient(api_key=token)
            model = os.getenv("HF_IMAGE_MODEL", HF_IMAGE_MODEL)
            try:
                image = client.text_to_image(prompt, model=model, width=IMG_W, height=IMG_H)
            except Exception:
                image = client.text_to_image(prompt, model=model)
            return image.convert("RGB")
        except Exception as e:
            print(f"HF inference indisponible : {e}")
            return None

    def _image_pollinations(self, prompt):
        url = (
            "https://image.pollinations.ai/prompt/"
            + urllib.parse.quote(prompt)
            + f"?model=flux&width={IMG_W}&height={IMG_H}&nologo=true&seed={int(time.time())}"
        )
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            # Retire le bandeau/filigrane Pollinations en bas de l'image
            w, h = img.size
            return img.crop((0, 0, w, int(h * 0.94)))
        except Exception as e:
            print(f"Pollinations indisponible : {e}")
            return None

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
        """Ajoute le nom de la page (net, jamais déformé comme avec une IA)."""
        img = img.convert("RGBA")
        w, h = img.size

        # Dégradé sombre en bas pour la lisibilité
        grad_h = int(h * 0.24)
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        for i in range(grad_h):
            alpha = int(215 * (i / grad_h) ** 1.4)
            y = h - grad_h + i
            d.line([(0, y), (w, y)], fill=(4, 10, 28, alpha))
        img = Image.alpha_composite(img, overlay)

        draw = ImageDraw.Draw(img)
        font_main = self._load_font(int(w * 0.062))
        font_sub = self._load_font(int(w * 0.027))

        part_a, part_b = BRAND_NAME
        wa = draw.textlength(part_a, font=font_main)
        wb = draw.textlength(part_b, font=font_main)
        x = (w - (wa + wb)) / 2
        y_main = h - int(h * 0.108)

        # Filet doré
        line_w = int(w * 0.14)
        y_line = y_main - int(h * 0.016)
        draw.rectangle(
            [(w - line_w) / 2, y_line, (w + line_w) / 2, y_line + 5],
            fill=BRAND_ACCENT + (255,),
        )

        # Ombre puis texte
        draw.text((x + 2, y_main + 2), part_a, font=font_main, fill=(0, 0, 0, 170))
        draw.text((x + wa + 2, y_main + 2), part_b, font=font_main, fill=(0, 0, 0, 170))
        draw.text((x, y_main), part_a, font=font_main, fill=(255, 255, 255, 255))
        draw.text((x + wa, y_main), part_b, font=font_main, fill=BRAND_ACCENT + (255,))

        # Identifiant de la page
        wh = draw.textlength(BRAND_HANDLE, font=font_sub)
        y_sub = y_main + int(w * 0.062) + int(h * 0.012)
        draw.text(((w - wh) / 2, y_sub), BRAND_HANDLE, font=font_sub, fill=(220, 225, 235, 230))

        return img.convert("RGB")

    def generer_image(self, news_original=None):
        print("Génération de l'image...")
        text_describe = self.donne_moi_prompt_images(news_original).content
        positive_prompt, _ = self._parse_prompts(text_describe)

        if not positive_prompt:
            titre = (news_original or "").split("\n")[0][:150]
            positive_prompt = (
                f"editorial news photo illustrating: {titre}, "
                "photorealistic, 85mm lens, sharp focus"
            )

        # FLUX ignore le negative prompt : les interdits sont mis dans le positif
        positive_prompt = " ".join(positive_prompt.split()[:60])
        positive_prompt += (
            ", no text, no letters, no people, photorealistic, ultra detailed, "
            "shot on Canon EOS R5, professional color grading, shallow depth of field, "
            "dramatic lighting, deep blue and gold tones"
        )
        print("Prompt image :", positive_prompt)

        img = self._image_hf(positive_prompt) or self._image_pollinations(positive_prompt)
        if img is None:
            raise RuntimeError("Aucune image générée : publication annulée.")

        img = ImageOps.fit(img, (OUT_W, OUT_H), Image.LANCZOS, centering=(0.5, 0.4))
        img = self._add_branding(img)

        output = OUTPUT_DIR / "img.png"
        img.save(output)
        print(f"Image sauvegardée : {output}")
        return output

    # ------------------------------------------------------------------
    # 3) Publication Facebook + Instagram
    # ------------------------------------------------------------------
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
                mime = "image/jpeg" if chemin.suffix.lower() in (".jpg", ".jpeg") else "image/png"
                try:
                    with chemin.open("rb") as fichier:
                        response = requests.post(
                            f"{GRAPH_URL}/{page_id}/photos",
                            data={"message": news_translited, "access_token": token},
                            files={"source": (chemin.name, fichier, mime)},
                            timeout=120,
                        )
                    if response.headers.get("content-type", "").startswith("application/json"):
                        data_fb = response.json()
                    else:
                        data_fb = {"message": response.text[:1000]}

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
                    jpeg_path = chemin.with_name(f"{chemin.stem}_instagram.jpg")
                    with Image.open(chemin) as image:
                        image.convert("RGB").save(jpeg_path, format="JPEG", quality=92, optimize=True)

                    upload = cloudinary.uploader.upload(
                        str(jpeg_path), resource_type="image", folder="actualites_financieres"
                    )
                    image_url = upload.get("secure_url")

                    if image_url and image_url.startswith("https://"):
                        creation_response = requests.post(
                            f"{GRAPH_URL}/{ig_id}/media",
                            data={"image_url": image_url, "caption": news_translited, "access_token": token},
                            timeout=90,
                        )
                        creation_data = creation_response.json()

                        if creation_response.ok and "error" not in creation_data and creation_data.get("id"):
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
                                pub_data = pub_response.json()
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
                    # Retire les restes du flux RSS ("The post ... appeared first on ...", "[...]")
                    clean_text = re.sub(r"\s*The post .*? appeared first on .*$", "", clean_text, flags=re.S)
                    clean_text = clean_text.replace("[...]", "…").strip()
                    publication_text, publication_datetime = self.get_publication_date(entry)

                    if publication_text is None:
                        continue

                    seen_links.add(link)
                    articles.append({
                        "source_name": source_name,
                        "title": self.clean_html(title),
                        "date": publication_text,
                        "continue": clean_text,
                        "_datetime": publication_datetime,
                    })
            except Exception as error:
                print(f"Erreur pour {source_name}: {error}")

        articles.sort(key=lambda item: item["_datetime"], reverse=True)

        result = {}
        for article in articles:
            src = article.pop("source_name")
            article.pop("_datetime")
            result[f"{src}_{article['date']}"] = article
        return result

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


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fichier_json = os.path.join(base_dir, "historique_posts.json")

    extract_post = Extract_post()
    text_post = TextPost()

    historique = {}

    if os.path.exists(fichier_json):
        try:
            with open(fichier_json, "r", encoding="utf-8") as f:
                historique = json.load(f)
            if not isinstance(historique, dict):
                historique = {}
        except Exception:
            historique = {}
    else:
        with open(fichier_json, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=4)

    all_news = extract_post.fetch_all_news()

    for source, news in all_news.items():
        vrai_source = source.split("_")[0]
        title = news["title"]
        content = news["continue"]
        date = news["date"]

        post_id = f"{vrai_source}_{title}_{date}"

        if post_id in historique:
            print(f"Déjà traité : {title}")
            continue

        text_original = f"{title}\n{content}\n\nsource: {vrai_source}, date: {date}"
        text_translated = extract_post.tradication_of_text(text_original)

        if isinstance(text_translated, dict):
            text_translated_net = " ".join(str(value) for value in text_translated.values())
        else:
            text_translated_net = str(text_translated)

        try:
            resultat = text_post.publier_texte_partout(text_original, text_translated_net)

            if resultat and (resultat.get("facebook") or resultat.get("instagram")):
                historique[post_id] = {"source": vrai_source, "titre": title, "date": date}
                with open(fichier_json, "w", encoding="utf-8") as f:
                    json.dump(historique, f, ensure_ascii=False, indent=4)
                print(f"Historique sauvegardé : {title}")
            else:
                print(f"Publication non confirmée : {title}.")
        except Exception as e:
            print(f"Erreur publication de '{title}' : {e}")
            continue
        break

    print("Fin de cette exécution.")


if __name__ == "__main__":
    main()
