"""Political-news lexicon, applied at TOPIC level (not article level).

The metadata DB's hyper-clusters are size-balanced partitions used for the
topic OoD split, not themes (each mixes crime, pets, recipes and politics), so
we classify the 41,705 per-year BERTopic topics instead: a topic is political
if enough of its article titles hit the lexicon, or its 5-keyword
representation contains political terms. Every article inherits its topic's
decision, so near-identical stories are kept or dropped together.

Lexicon = the pilot script's POLITICAL regex, plus parties, office holders,
institutions and contested policy areas. Ambiguous words that are frequent in
sports/business titles (bill, campaign, rally, poll, trade, war, president
alone) are left out or only used in multi-word forms.
"""

import re

# RE2-compatible (used inside DuckDB regexp_matches) and Python-compatible.
TERMS = [
    # --- pilot script's POLITICAL regex (minus 'bill', 'policy', 'president', 'border', 'protest',
    #     which are re-added below in less ambiguous forms)
    r"abortion", r"immigration", r"immigrants?", r"migrants?", r"asylum", r"gun control",
    r"firearms?", r"second amendment", r"climate", r"emissions?", r"vaccine mandates?",
    r"vaccination mandates?", r"mandates?", r"lockdowns?", r"elections?", r"ballots?",
    r"voters?", r"senate", r"congress", r"parliament", r"prime minister", r"supreme court",
    r"legislation", r"tax(es)?", r"tariffs?", r"healthcare", r"health care", r"welfare",
    r"minimum wage", r"pensions?", r"brexit", r"referendum", r"impeach\w*", r"sanctions?",
    r"deportations?", r"refugees?", r"racism", r"police reform", r"defund",
    # --- institutions, offices, process
    r"senators?", r"congressional", r"congress(man|woman)", r"lawmakers?", r"legislat\w+",
    r"parliamentary", r"mps?", r"white house", r"downing street", r"kremlin", r"westminster",
    r"cabinet", r"ministers?", r"ministry", r"presiden(t|cy|tial) (trump|biden|obama|putin|xi|macron|erdogan|zelensk\w+)",
    r"presidential", r"vice president", r"governors?", r"gubernatorial", r"mayor(al)?",
    r"state department", r"pentagon", r"justice department", r"attorney general",
    r"filibuster", r"veto(es|ed)?", r"executive orders?", r"(spending|relief|infrastructure|stimulus|budget|gun|abortion|voting) bill",
    r"bill (passes|signed)", r"signs? bill", r"nominees?", r"confirmation hearings?",
    r"gerrymander\w*", r"redistricting", r"midterms?", r"primar(y|ies) (race|election|challenger|win|victory)",
    r"caucus(es)?", r"electoral", r"campaign trail", r"(presidential|senate|reelection|re-election) campaign",
    r"polls? (show|shows|finds?)", r"approval rating", r"shutdown", r"debt ceiling", r"deficit",
    r"stimulus", r"constitution(al)?", r"amendment",
    # --- parties and office holders
    r"republicans?", r"democrats?", r"democratic party", r"gop", r"dnc", r"rnc", r"tor(y|ies)",
    r"labour", r"lib dems?", r"snp", r"trump\w*", r"biden\w*", r"obama\w*", r"clinton", r"pelosi",
    r"mcconnell", r"schumer", r"bernie", r"sanders", r"aoc", r"ocasio-cortez", r"kamala", r"pence",
    r"desantis", r"mccarthy", r"boris johnson", r"theresa may", r"corbyn", r"starmer", r"sunak",
    r"liz truss", r"macron", r"merkel", r"putin", r"zelensk\w+", r"xi jinping", r"netanyahu",
    r"erdogan", r"bolsonaro", r"modi", r"trudeau", r"kim jong", r"maduro", r"orban",
    r"cuomo", r"de blasio", r"newsom", r"whitmer", r"hochul", r"giuliani", r"bannon", r"mueller",
    r"comey", r"kavanaugh", r"pompeo", r"bolton", r"manafort", r"cheney", r"mnuchin", r"fauci",
    # --- international politics and conflict
    r"nato", r"european union", r"eu", r"united nations", r"treaty", r"diplomat(s|ic)?",
    r"embassy", r"foreign policy", r"nuclear deal", r"invasion", r"ceasefire", r"coup",
    r"annex\w*", r"troops", r"government", r"ukrain\w*", r"russia(n|ns)?", r"israel(i|is)?",
    r"palestin\w+", r"gaza", r"hamas", r"hezbollah", r"taliban", r"afghanistan", r"iran(ian)?",
    r"north korea(n)?", r"syria(n)?", r"venezuela(n)?", r"hong kong protests?", r"uyghurs?",
    # --- contested social issues
    r"protest(s|ers?)?", r"insurrection", r"capitol (riot|attack|breach)", r"jan(uary)? 6(th)?",
    r"racist", r"black lives matter", r"blm", r"critical race theory", r"crt", r"systemic racism",
    r"transgender", r"lgbt\w*", r"same-sex", r"gay marriage", r"religious freedom", r"free speech",
    r"censorship", r"pro-life", r"pro-choice", r"planned parenthood", r"roe", r"nra",
    r"gun rights", r"assault weapons?", r"border (wall|crisis|patrol|security)", r"daca", r"dreamers",
    r"obamacare", r"affordable care act", r"medicare", r"medicaid", r"social security",
    r"green new deal", r"paris (agreement|accord)", r"carbon tax", r"vaccine passports?",
    r"mask mandates?", r"civil rights", r"voting rights", r"voter (fraud|suppression|id)",
    r"election (fraud|integrity)", r"stolen election", r"antitrust", r"regulators?",
]

PATTERN = r"(?i)\b(" + "|".join(TERMS) + r")\b"
RX = re.compile(PATTERN)


def rep_hits(representation):
    """Number of the topic's 5 representation keywords that hit the lexicon."""
    kws = [k.strip().replace("_", " ") for k in (representation or "").split(",")]
    return sum(bool(RX.search(k)) for k in kws if k)
