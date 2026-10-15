import re
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import quote

import requests

USER_AGENT = "devinsa-image-scraper/1.0"
MIN_IMAGE_WIDTH = 60
IGNORED_IMAGES = re.compile(r"Flag_of|Drapeau|Blason|Icon|Logo|wordmark|Wikidata|\.svg/", re.IGNORECASE)
DISAMBIGUATION = re.compile(r'"wgCategories":\[[^\]]*"(Homonymie|Disambiguation pages)"')


def is_valid_image(url: str, width: str) -> bool:
    return width.isdigit() and int(width) >= MIN_IMAGE_WIDTH and not IGNORED_IMAGES.search(url)


class ImageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.infobox_image: str | None = None
        self.preview_image: str | None = None
        self.depth = 0
        self.infobox_depth: int | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key: value or "" for key, value in attrs}

        if tag == "meta" and attributes.get("property") == "og:image":
            image = attributes.get("content", "")
            if not IGNORED_IMAGES.search(image):
                self.preview_image = image

        if tag in ("table", "div"):
            self.depth += 1
            if self.infobox_depth is None and "infobox" in attributes.get("class", ""):
                self.infobox_depth = self.depth

        if tag == "img" and self.infobox_depth and self.infobox_image is None:
            if is_valid_image(attributes.get("src", ""), attributes.get("width", "")):
                self.infobox_image = attributes["src"]

    def handle_endtag(self, tag: str) -> None:
        if tag in ("table", "div"):
            if self.depth == self.infobox_depth:
                self.infobox_depth = None
            self.depth -= 1


@dataclass
class ScrapeResult:
    page_url: str
    image_url: str | None
    is_disambiguation: bool


class WikipediaImageScraper:
    def __init__(self, lang: str = "fr") -> None:
        self.base_url = f"https://{lang}.wikipedia.org"
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT

    def find_image(self, name: str) -> str | None:
        names = [name]
        if short_name(name) != name:
            names.append(short_name(name))

        for candidate in names:
            result = self.scrape(candidate)
            if result is not None and result.image_url and not result.is_disambiguation:
                return result.image_url

        return None

    def scrape(self, name: str) -> ScrapeResult | None:
        response = self.get(f"{self.base_url}/wiki/{quote(name.replace(' ', '_'))}")

        if response is None:
            response = self.get(f"{self.base_url}/w/index.php", params={"search": name, "go": "Go"})
            if response is None or "search=" in response.url:
                return None

        parser = ImageParser()
        parser.feed(response.text)
        image = parser.infobox_image or parser.preview_image

        if image:
            image = clean_image_url(image)

        return ScrapeResult(
            page_url=response.url,
            image_url=image,
            is_disambiguation=DISAMBIGUATION.search(response.text) is not None,
        )

    def get(self, url: str, params: dict[str, str] | None = None) -> requests.Response | None:
        try:
            response = self.session.get(url, params=params, timeout=10)
        except requests.RequestException as error:
            print(f"Request failed for {url}: {error}")
            return None

        if response.status_code != 200:
            return None

        return response


def short_name(name: str) -> str:
    return re.split(r"[(,;:|]", name)[0].strip()


def clean_image_url(url: str) -> str:
    url = url.split("?")[0]
    if url.startswith("//"):
        url = "https:" + url

    return url
