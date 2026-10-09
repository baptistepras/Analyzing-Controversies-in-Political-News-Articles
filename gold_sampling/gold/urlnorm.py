"""URL normalisation shared by article raw_url and Reddit submission url.

key = host (no www./m./amp.) + path (no AMP markers, index pages, trailing
slash) + sorted non-tracking query params. Scheme and fragment are dropped.
"""

import html
import re
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit

TRACKING = re.compile(
    r"^(utm_.*|ftag|fbclid|gclid|dclid|mc_cid|mc_eid|cmpid|cmp|ocid|smid|smtyp|partner|ref|"
    r"ref_src|referrer|source|src|via|ito|xid|soc_src|soc_trk|_ga|amp|amp_js_v|usqp|outputtype|"
    r"feature|sh|spref|rss|mod|cid|sr_share|share|s|trk|tag|taid|ns_.*|icid|int_.*|wt\.mc_id|"
    r"guccounter|guce_referrer.*|__twitter_impression|fb_.*|hootpostid|n_cid|cmpgn|campaign)$", re.I)
AMP_CACHE = re.compile(r"^[^/]+\.cdn\.ampproject\.org/(?:[a-z]/)*(?:s/)?(.+)$", re.I)
GOOGLE_AMP = re.compile(r"^(?:www\.)?google\.[a-z.]+/amp/(?:s/)?(.+)$", re.I)
HOST_PREFIX = re.compile(r"^(?:www\d?|m|mobile|amp|amp-|edition)\.", re.I)


def normalize_url(url):
    if not url or not isinstance(url, str) or url in ("N/A",):
        return None
    u = html.unescape(url.strip())
    u = re.sub(r"^[a-z][a-z0-9+.-]*://", "", u, flags=re.I)
    for rx in (AMP_CACHE, GOOGLE_AMP):
        m = rx.match(u)
        if m:
            u = re.sub(r"^[a-z][a-z0-9+.-]*://", "", unquote(m.group(1)), flags=re.I)
    try:
        sp = urlsplit("//" + u)
    except ValueError:
        return None
    host = (sp.hostname or "").lower().rstrip(".")
    while HOST_PREFIX.match(host):
        host = HOST_PREFIX.sub("", host, count=1)
    if not host or "." not in host:
        return None
    path = re.sub(r"/{2,}", "/", unquote(sp.path or ""))
    path = re.sub(r"(/amp)+/?$|\.amp(\.html)?$|/amp(?=/)", "", path, flags=re.I)
    path = re.sub(r"/(index|default)\.(html?|php|aspx?)$", "", path, flags=re.I)
    path = path.rstrip("/")
    q = sorted((k, v) for k, v in parse_qsl(sp.query, keep_blank_values=False) if not TRACKING.match(k))
    return host + path + ("?" + urlencode(q) if q else "")
