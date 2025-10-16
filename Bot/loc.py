from __future__ import annotations
import json
import random
import logging
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional

logger = logging.getLogger(__name__)

def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)  # merge recursivo
        else:
            out[k] = v
    return out

class Localizer:
    """
    Carrega bundles JSON por idioma, com:
    - fallback chain (ex.: pt-BR → pt → en)
    - cache por idioma
    - acesso a chaves aninhadas via "a.b.c"
    - placeholders opcionais via % (ex.: "%(user)s")
    """
    def __init__(
        self,
        base_dir: str | Path = "./Bot/Static/locales",
        filename: str = "lib.json",
        default_lang: str = "en",
        fallback: Optional[Mapping[str, Iterable[str]]] = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.filename = filename
        self.default_lang = default_lang
        self.fallback = dict(fallback or {})
        # indexa arquivos existentes: { "pt": Path(.../pt/lib.json), ... }
        self._index: Dict[str, Path] = {
            p.parent.name: p
            for p in self.base_dir.glob(f"*/{self.filename}")
        }

    @lru_cache(maxsize=64)
    def _raw_bundle(self, lang: str) -> Mapping[str, Any]:
        path = self._index.get(lang)
        if not path:
            return MappingProxyType({})
        try:
            data = _read_json(path)
            if not isinstance(data, dict):
                logger.warning("Bundle %s não é um objeto JSON de nível raiz.", path)
                data = {}
            return MappingProxyType(data)
        except Exception as e:
            logger.warning("Falha ao ler %s: %s", path, e)
            return MappingProxyType({})

    @lru_cache(maxsize=64)
    def bundle(self, lang: str) -> Mapping[str, Any]:
        """
        Retorna o bundle final com fallback aplicado (imutável).
        Ordem: [*fallbacks..., default_lang] → merge → lang (por último).
        """
        chain = list(self.fallback.get(lang, []))
        if lang != self.default_lang:
            chain.append(self.default_lang)

        merged: Mapping[str, Any] = {}
        for fb in chain:
            merged = _deep_merge(merged, self._raw_bundle(fb))
        merged = _deep_merge(merged, self._raw_bundle(lang))

        # imutável p/ evitar modificações acidentais
        return MappingProxyType(merged)

    def get(self, key: str, lang: Optional[str] = None, default: Any = None, **fmt) -> Any:
        """
        Busca "a.b.c" no bundle. Se string, formata com %(**fmt) se fornecido.
        """
        bundle = self.bundle(lang or self.default_lang)
        node: Any = bundle
        for part in key.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]

        if isinstance(node, str) and fmt:
            try:
                return node % fmt
            except Exception:
                # retorna sem formatar em caso de erro
                return node
        return node

    def get_choice(self, key: str, lang: Optional[str] = None, default: Any = None, **fmt) -> Any:
        node = random.choice(self.get(key, lang))

        if isinstance(node, str) and fmt:
            try:
                return node % fmt
            except Exception:
                return node
        return node

    def _lang_chain(self, lang: str):
        lang = (lang or "").lower()
        chain = [lang] + list(self.fallback.get(lang, []))
        if self.default_lang not in chain:
            chain.append(self.default_lang)
        # remove duplicados preservando a ordem
        seen, out = set(), []
        for c in chain:
            if c and c not in seen:
                seen.add(c); out.append(c)
        return out

    def get_file(self, filename: str, lang: str) -> str:
        for code in self._lang_chain(lang):
            p = self.base_dir / code / filename
            if p.exists():
                return p.read_text(encoding="utf-8")
        raise FileNotFoundError(filename)

    def get_random_line(self, filename: str, lang: str, comment_prefix: str = "#") -> str:
        #print(lang,filename)
        text = self.get_file(filename, lang)
        lines = [ln.strip() for ln in text.splitlines()]
        lines = [ln for ln in lines if ln and not ln.startswith(comment_prefix)]
        if not lines:
            return ""
        return random.choice(lines)

# ---------- Exemplo de uso ----------

# Fallbacks opcionais (ajuste aos seus códigos)
fallback = {
    "pt-BR": ["pt"],
    "es-AR": ["es"],
    "en": ["eng"],
    "pt": ["eng"],
    "es": ["eng"],
}

i18n = Localizer(
    base_dir="./Static/locales",
    filename="lib.json",
    default_lang="eng",
    fallback=fallback,
)

VALID_LANGS = ["pt", "es", "eng"]

# Acessos:
# i18n.get("greeting", lang="pt")
# i18n.get("welcome", lang="es", user="Felipe")
