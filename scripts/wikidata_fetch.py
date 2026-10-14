"""Minimal online demo, object‑oriented, with hard‑coded parameters.

Features:
- Object‑oriented `WikidataClient` and `Person` model
- Fetches a few fields (label, gender, citizenships, occupations, birth/death year, image)
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Any

import requests

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
WIKIPEDIA_TOP_URL = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/top/{lang}.wikipedia/all-access/{yyyy}/{mm}/{dd}"
)
USER_AGENT = "devinsa-wikidata-minimal/0.5"


@dataclass
class Person:
    qid: str
    label: str
    gender: str | None
    citizenships: list[str]
    occupations: list[str]
    birth_year: str | None
    death_year: str | None
    image: str | None
    sitelinks: int | None = None

    @property
    def url(self) -> str:
        return f"https://www.wikidata.org/wiki/{self.qid}"


class WikidataClient:
    def __init__(self, *, lang: str = "en,fr", timeout: float = 30.0, max_workers: int = 8) -> None:
        self.lang = lang
        self.timeout = timeout
        self.max_workers = max_workers
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    # use wikipedia pageviews top to pick candidates, then map to wikidata
    def fetch_top_people(self, limit: int = 100, wiki_lang: str = "en") -> list[Person]:
        titles = self._get_top_titles(wiki_lang, limit * 3)
        entities = self._get_entities_for_titles(titles, site=f"{wiki_lang}wiki")

        human_qid = "Q5"
        humans: list[dict] = []
        value_qids: set[str] = set()
        for ent in entities:
            if not ent or ent.get("missing") is not None:
                continue
            claims = ent.get("claims", {})
            if not self._has_item(claims, "P31", human_qid):
                continue
            humans.append(ent)
            value_qids.update(self._collect_value_qids(claims, ["P21", "P27", "P106"]))

        labels_map = self._get_labels_for_qids(value_qids)

        people: list[Person] = []
        for ent in humans[:limit]:
            people.append(self._entity_to_person(ent, labels_map))
        return people

    def _get_top_titles(self, wiki_lang: str, grab: int) -> list[str]:
        y = (datetime.now(timezone.utc) - timedelta(days=1)).date()
        url = WIKIPEDIA_TOP_URL.format(lang=wiki_lang, yyyy=y.year, mm=f"{y.month:02d}", dd=f"{y.day:02d}")
        r = self.session.get(url, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        articles = data.get("items", [{}])[0].get("articles", [])
        titles: list[str] = []
        for a in articles:
            title = a.get("article", "")
            if not title:
                continue
            if ":" in title or title == "Main_Page":
                continue
            titles.append(title)
            if len(titles) >= grab:
                break
        return titles

    def _get_entities_for_titles(self, titles: list[str], site: str) -> list[dict]:
        entities: list[dict] = []
        chunk = 50
        
        batches = [titles[i:i+chunk] for i in range(0, len(titles), chunk)]
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {
                executor.submit(self._fetch_entities_batch, batch, site): batch
                for batch in batches
            }
            
            for future in as_completed(future_to_batch):
                try:
                    batch_entities = future.result()
                    entities.extend(batch_entities)
                except Exception as exc:
                    print(f"Entity batch failed: {exc}", file=sys.stderr)
        
        return entities

    def _fetch_entities_batch(self, batch: list[str], site: str) -> list[dict]:
        """Helper method to fetch a single batch of entities."""
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})
        
        resp = session.get(
            WIKIDATA_API_URL,
            params={
                "action": "wbgetentities",
                "sites": site,
                "titles": "|".join(batch),
                "props": "labels|claims|sitelinks",
                "languages": self.lang,
                "format": "json",
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return [v for v in resp.json().get("entities", {}).values()]

    def _get_labels_for_qids(self, qids: set[str]) -> dict[str, str]:
        if not qids:
            return {}
        
        labels: dict[str, str] = {}
        chunk = 50
        ids = list(qids)
        
        batches = [ids[i:i+chunk] for i in range(0, len(ids), chunk)]
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {
                executor.submit(self._fetch_labels_batch, batch): batch
                for batch in batches
            }
            
            for future in as_completed(future_to_batch):
                try:
                    batch_labels = future.result()
                    labels.update(batch_labels)
                except Exception as exc:
                    print(f"Labels batch failed: {exc}", file=sys.stderr)
        
        return labels

    def _fetch_labels_batch(self, batch: list[str]) -> dict[str, str]:
        """Helper method to fetch a single batch of labels."""
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})
        
        resp = session.get(
            WIKIDATA_API_URL,
            params={
                "action": "wbgetentities",
                "ids": "|".join(batch),
                "props": "labels",
                "languages": self.lang,
                "format": "json",
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        
        labels: dict[str, str] = {}
        j = resp.json().get("entities", {})
        pref_lang = self.lang.split(",")[0]
        for qid, ent in j.items():
            lbl = ent.get("labels", {}).get(pref_lang, {}).get("value")
            if not lbl and ent.get("labels"):
                any_lbl = next(iter(ent["labels"].values()))
                lbl = any_lbl.get("value")
            if lbl:
                labels[qid] = lbl
        return labels

    @staticmethod
    def _collect_value_qids(claims: dict, props: list[str]) -> set[str]:
        out: set[str] = set()
        for p in props:
            for c in claims.get(p, []):
                try:
                    vid = c["mainsnak"]["datavalue"]["value"]["id"]
                    if isinstance(vid, str):
                        out.add(vid)
                except Exception:
                    continue
        return out

    @staticmethod
    def _has_item(claims: dict, prop: str, qid: str) -> bool:
        for c in claims.get(prop, []):
            try:
                vid = c["mainsnak"]["datavalue"]["value"]["id"]
                if vid == qid:
                    return True
            except Exception:
                continue
        return False

    def _entity_to_person(self, ent: dict, labels_map: dict[str, str]) -> Person:
        pref_lang = self.lang.split(",")[0]
        qid = ent.get("id", "")
        label = ent.get("labels", {}).get(pref_lang, {}).get("value") or qid
        claims = ent.get("claims", {})

        gender_id = self._first_item_qid(claims, "P21")
        gender = labels_map.get(gender_id) if gender_id else None

        cit_ids = self._list_item_qids(claims, "P27")
        citizenships = [labels_map.get(q, q) for q in cit_ids if q]

        occ_ids = self._list_item_qids(claims, "P106")
        occupations = [labels_map.get(q, q) for q in occ_ids if q]

        image = self._first_commons_file(claims, "P18")
        birth_year = self._first_time_year(claims, "P569")
        death_year = self._first_time_year(claims, "P570")

        sitelinks = None
        if "sitelinks" in ent and isinstance(ent["sitelinks"], dict):
            sitelinks = len(ent["sitelinks"])

        return Person(
            qid=qid,
            label=label,
            gender=gender,
            citizenships=citizenships,
            occupations=occupations,
            birth_year=birth_year,
            death_year=death_year,
            image=image,
            sitelinks=sitelinks,
        )

    @staticmethod
    def _first_item_qid(claims: dict, prop: str) -> str | None:
        for c in claims.get(prop, []):
            try:
                return c["mainsnak"]["datavalue"]["value"]["id"]
            except Exception:
                continue
        return None

    @staticmethod
    def _list_item_qids(claims: dict, prop: str) -> list[str]:
        ids: list[str] = []
        for c in claims.get(prop, []):
            try:
                vid = c["mainsnak"]["datavalue"]["value"]["id"]
                if isinstance(vid, str):
                    ids.append(vid)
            except Exception:
                continue
        return ids

    @staticmethod
    def _first_time_year(claims: dict, prop: str) -> str | None:
        for c in claims.get(prop, []):
            try:
                t = c["mainsnak"]["datavalue"]["value"]["time"]
                if isinstance(t, str) and len(t) >= 5:
                    return t[1:5] if t.startswith("+") else t[:4]
            except Exception:
                continue
        return None

    @staticmethod
    def _first_commons_file(claims: dict, prop: str) -> str | None:
        for c in claims.get(prop, []):
            try:
                fname = c["mainsnak"]["datavalue"]["value"]
                if isinstance(fname, str):
                    return fname
            except Exception:
                continue
        return None


def print_text(people: list[Person]) -> None:
    for p in people:
        print(p.label)
        print(f"  URL        : {p.url}")
        if p.sitelinks is not None:
            print(f"  Sitelinks  : {p.sitelinks}")
        if p.birth_year or p.death_year:
            print(f"  Lifespan   : {p.birth_year or '?'} – {p.death_year or ''}")
        if p.gender:
            print(f"  Gender     : {p.gender}")
        if p.citizenships:
            print(f"  Citizenship: {', '.join(p.citizenships)}")
        if p.occupations:
            print(f"  Occupations: {', '.join(p.occupations)}")
        if p.image:
            print(f"  Image      : {p.image}")
        print()


def main() -> int:
    # parameters
    TOP_N = 100              # target number of people
    lang = "fr"              # label languages preference
    timeout = 60.0           # HTTP timeout
    max_workers = 6          # number of concurrent threads

    client = WikidataClient(lang=lang, timeout=timeout, max_workers=max_workers)
    try:
        people = client.fetch_top_people(limit=TOP_N, wiki_lang="en")
    except Exception as exc:
        print(f"Error querying Wikidata: {exc}", file=sys.stderr)
        return 1

    print_text(people)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
