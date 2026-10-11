# -*- coding: utf-8 -*-
"""AngelWhite独自回答エンジン。

検索→文章抽出→関連度/出典数による信頼度推定→知識候補保存を行う。
外部LLM/APIキーには依存しない。回答生成は検索結果からの抽出的・テンプレート的生成であり、
ニューラルネットワークによる文章生成ではない。
"""
from __future__ import annotations

import html
import json
import re
from difflib import SequenceMatcher
import urllib.parse
import urllib.request
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path


class _SearchParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.items = []
        self.current = None
        self.capture = None
        self.buf = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a" and (
                "result__a" in attrs.get("class", "")
                or "result-link" in attrs.get("class", "")
        ):
            self.current = {
                "title": "",
                "url": attrs.get("href", ""),
                "snippet": "",
            }
            self.items.append(self.current)
            self.capture = "title"
            self.buf = []
        elif self.current and tag in ("a", "div", "td") and (
                "result__snippet" in attrs.get("class", "")
                or "result-snippet" in attrs.get("class", "")
        ):
            self.capture = "snippet"
            self.buf = []

    def handle_data(self, data):
        if self.capture:
            self.buf.append(data)

    def handle_endtag(self, tag):
        if self.capture == "title" and tag == "a" and self.current:
            self.current["title"] = " ".join("".join(self.buf).split())
            self.capture = None
            self.buf = []
        elif self.capture == "snippet" and tag in ("div", "td") and self.current:
            self.current["snippet"] = " ".join("".join(self.buf).split())
            self.capture = None
            self.buf = []


class JapaneseAI:
    # 「紫が好きです」のような文でも色として扱うための簡易色リスト。
    # 必要に応じてあとから追加できる。
    COLOR_WORDS = (
        "紫色", "紫", "青色", "青", "水色", "紺色", "紺", "赤色", "赤",
        "桃色", "ピンク", "オレンジ色", "オレンジ", "黄色", "黄",
        "緑色", "緑", "黄緑", "黒色", "黒", "白色", "白", "灰色", "グレー",
        "茶色", "ベージュ", "金色", "銀色", "虹色",
    )

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.knowledge_file = self.data_dir / "ai_knowledge.json"
        self.unlearned_file = self.data_dir / "ai_unlearned.json"
        self.memory_file = self.data_dir / "ai_context_memory.json"
        self.research_file = self.data_dir / "ai_research_log.json"

        self.knowledge = self._load_json(
            self.knowledge_file, {"pairs": {}, "counts": {}}
        )
        self.unlearned = self._load_json(self.unlearned_file, [])
        self.context_memory = self._load_json(
            self.memory_file, {"turns": [], "facts": []}
        )
        self.research_log = self._load_json(self.research_file, [])

        if not isinstance(self.knowledge, dict):
            self.knowledge = {"pairs": {}, "counts": {}}
        self.knowledge.setdefault("pairs", {})
        self.knowledge.setdefault("counts", {})

        if not isinstance(self.unlearned, list):
            self.unlearned = []

        if not isinstance(self.context_memory, dict):
            self.context_memory = {"turns": [], "facts": []}
        self.context_memory.setdefault("turns", [])
        self.context_memory.setdefault("facts", [])

        if not isinstance(self.context_memory["turns"], list):
            self.context_memory["turns"] = []
        if not isinstance(self.context_memory["facts"], list):
            self.context_memory["facts"] = []

        if not isinstance(self.research_log, list):
            self.research_log = []

        # メモリのON/OFFはmain.py側の設定と連動する。
        # 初期値はON。main.pyが設定値を読み込んで変更する場合はそちらを優先する。
        self.memory_enabled = True
        self._save_knowledge()

    @staticmethod
    def _load_json(path, default):
        try:
            if path.exists():
                with path.open("r", encoding="utf-8") as file:
                    return json.load(file)
        except (OSError, json.JSONDecodeError):
            pass
        return default

    @staticmethod
    def _atomic_write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        temp.replace(path)

    @staticmethod
    def _normalize(text):
        text = text.strip().lower()
        text = re.sub(r"[\s　]+", "", text)
        return re.sub(r"[、。，．.!！?？…〜～]+$", "", text)

    @classmethod
    def _is_greeting(cls, text):
        """短い挨拶に限って、軽い誤字・文字の入れ替わりを許容する。"""
        normalized = cls._normalize(text)
        if not normalized:
            return False

        # よくある表記ゆれ・入力ミスは明示的に正規化する。
        aliases = {
            "こんにちわ": "こんにちは",
            "こんちわ": "こんにちは",
            "こんちは": "こんにちは",
            "こんいちは": "こんにちは",
            "こんんちは": "こんにちは",
            "こんいちわ": "こんにちは",
            "おはよー": "おはよう",
            "おはよお": "おはよう",
            "ばんわ": "こんばんは",
        }
        normalized = aliases.get(normalized, normalized)
        greetings = (
            "こんにちは", "やあ", "おはよう", "おはよ", "こんばんは",
            "おっす", "どうも", "こんちゃ",
        )
        if normalized in greetings:
            return True

        # 短い定型挨拶だけに類似度判定を適用し、普通の質問を誤分類しにくくする。
        for greeting in greetings:
            if abs(len(normalized) - len(greeting)) > 1:
                continue
            if SequenceMatcher(None, normalized, greeting).ratio() >= 0.76:
                return True
        return False

    def _save_knowledge(self):
        self._atomic_write(self.knowledge_file, self.knowledge)

    def _save_unlearned(self):
        self._atomic_write(self.unlearned_file, self.unlearned[-1000:])

    def _save_research(self):
        """検索で得た回答候補の履歴をJSONへ安全に保存する。"""
        self._atomic_write(self.research_file, self.research_log[-1000:])

    def set_memory_enabled(self, enabled: bool):
        """会話メモリのON/OFFを設定する。"""
        self.memory_enabled = bool(enabled)

    def _save_memory(self):
        """会話履歴と、チャットをまたいで使うユーザー情報を永続保存する。"""
        self._atomic_write(
            self.memory_file,
            {
                "turns": self.context_memory["turns"][-300:],
                "facts": self.context_memory["facts"][-500:],
            },
        )

    @classmethod
    def _detect_fact(cls, text):
        """発言からプロフィール情報を抽出する。未知の話題も汎用メモとして保存する。"""
        text = text.strip()
        if not text:
            return None
        clean = re.sub(r"[。！？!?]+$", "", text).strip()
        if not clean or len(clean) > 500:
            return None

        # 明示的な項目形式：「好きな色は紫」「趣味はゲーム」など。
        explicit_patterns = (
            (r"(?:(?:俺|僕|私|わたし|自分)の)?好きな色は\s*(.+)$", "好きな色"),
            (r"(?:(?:俺|僕|私|わたし|自分)の)?好きな動物は\s*(.+)$", "好きな動物"),
            (r"(?:(?:俺|僕|私|わたし|自分)の)?趣味は\s*(.+)$", "趣味"),
            (r"(?:(?:俺|僕|私|わたし|自分)の)?目標は\s*(.+)$", "目標"),
            (r"(?:(?:俺|僕|私|わたし|自分)の)?夢は\s*(.+)$", "夢"),
            (r"(?:(?:俺|僕|私|わたし|自分)の)?名前は\s*(.+)$", "名前"),
            (r"(?:(?:俺|僕|私|わたし|自分)は)?\s*住んでいる場所は\s*(.+)$", "住んでいる場所"),
        )
        for pattern, label in explicit_patterns:
            match = re.search(pattern, clean)
            if match:
                value = re.sub(r"(?:だよ|です|だ)$", "", match.group(1).strip()).strip()
                if value and len(value) <= 200:
                    return label, value

        # 「俺は猫が好き」「私は数学が苦手」など。
        preference = re.search(
            r"^(?:(俺|僕|私|わたし|自分)は)?\s*(.+?)が(好き|嫌い|苦手|得意)(?:です|だよ|だ)?$",
            clean,
        )
        if preference:
            subject = preference.group(2).strip()
            kind = preference.group(3)
            if subject and len(subject) <= 100:
                animal_words = ("猫", "ねこ", "ネコ", "犬", "いぬ", "イヌ", "鳥", "うさぎ", "ウサギ", "ハムスター", "魚", "金魚", "亀", "カメ", "馬", "イルカ", "パンダ", "ライオン", "ペット")
                color_words = cls.COLOR_WORDS
                if kind == "好き" and any(word in subject for word in color_words):
                    label = "好きな色"
                    for color in sorted(color_words, key=len, reverse=True):
                        if color in subject:
                            subject = color
                            break
                elif kind == "好き" and any(word in subject for word in animal_words):
                    label = "好きな動物"
                elif kind == "好き":
                    label = "好きなもの"
                else:
                    label = {"嫌い": "嫌いなもの", "苦手": "苦手なもの", "得意": "得意なもの"}[kind]
                return label, subject

        # 名前の書き方の揺れを補助。
        match = re.search(r"^(?:俺|僕|私|わたし|自分)の名前は\s*(.+)$", clean)
        if match:
            return "名前", match.group(1).strip()

        # 質問・命令ではない発言は、特定のカテゴリに分類できなくても会話メモとして残す。
        if (not re.search(r"[?？]", text)
                and not re.match(r"^(?:覚えて|検索して|調べて|教えて|説明して|計算して|翻訳して)", clean)
                and len(clean) >= 4
                and not re.fullmatch(r"(?:こんにちは|こんばんは|おはよう|ありがとう|どうも|またね|はい|いいえ)", clean)):
            return "会話メモ", clean
        return None

    def _remember_turn(self, q, a):
        if not self.memory_enabled:
            return

        now = datetime.now().isoformat(timespec="seconds")
        self.context_memory["turns"].append(
            {"user": q, "assistant": a, "timestamp": now}
        )

        detected = self._detect_fact(q)
        if detected:
            label, value = detected
            facts = self.context_memory["facts"]
            # 同じ項目は最新の値で更新する。
            facts[:] = [fact for fact in facts if fact.get("label") != label]
            facts.append(
                {"label": label, "value": value, "updated": now}
            )

        self._save_memory()

    def _memory_answer(self, query):
        """カテゴリ別の記憶を優先し、見つからなければ過去の発言を照合する。"""
        if not self.memory_enabled:
            return None
        q = self._normalize(query)
        facts = self.context_memory.get("facts", [])

        if re.search(r"(好きな色|好きなカラー|何色が好き|何色好き|カラーは何色)", q):
            for fact in reversed(facts):
                if fact.get("label") == "好きな色":
                    return f'前に教えてくれた好きな色は{fact.get("value")}だよ！'

        if re.search(r"(好きな動物|好きなペット|好きな生き物|何の動物が好き|動物は何が好き|どんな動物が好き)", q):
            for fact in reversed(facts):
                if fact.get("label") == "好きな動物":
                    return f'前に教えてくれた好きな動物は{fact.get("value")}だよ！'
                if fact.get("label") == "好きなもの":
                    value = str(fact.get("value", ""))
                    if any(word in value for word in ("猫", "ねこ", "ネコ", "犬", "いぬ", "イヌ", "鳥", "うさぎ", "ウサギ", "ハムスター", "魚", "金魚", "亀", "カメ", "馬", "イルカ", "パンダ", "ライオン", "ペット")):
                        return f'前に教えてくれた好きな動物は{value}だよ！'

        category_patterns = (
            (r"(趣味|何をするのが好き|暇なとき何をする)", "趣味", "趣味"),
            (r"(目標|何を目指している|何を達成したい)", "目標", "目標"),
            (r"(夢は何|将来の夢|夢を教えて)", "夢", "夢"),
            (r"(名前は何|名前はなに|俺の名前|僕の名前|私の名前|わたしの名前)", "名前", "名前"),
            (r"(住んでいる場所|どこに住んでいる|住んでる場所)", "住んでいる場所", "住んでいる場所"),
            (r"(何が嫌い|嫌いなもの)", "嫌いなもの", "嫌いなもの"),
            (r"(何が苦手|苦手なもの)", "苦手なもの", "苦手なもの"),
            (r"(何が得意|得意なもの)", "得意なもの", "得意なもの"),
        )
        for pattern, label, display in category_patterns:
            if re.search(pattern, q):
                for fact in reversed(facts):
                    if fact.get("label") == label:
                        return f'前に教えてくれた{display}は{fact.get("value")}だよ！'

        if re.search(r"(好きなもの|何が好き|何がすき)", q):
            for fact in reversed(facts):
                if fact.get("label") in ("好きなもの", "好きな動物", "好きな色"):
                    return f'前に教えてくれた好きなものは{fact.get("value")}だよ！'

        # 汎用検索：質問に含まれる単語と過去の記憶・発言を照合する。
        query_tokens = self._tokens(q)
        candidates = []
        for fact in facts:
            value = str(fact.get("value", ""))
            label = str(fact.get("label", ""))
            text = label + value
            tokens = self._tokens(text)
            overlap = len(query_tokens & tokens) / max(1, len(query_tokens))
            if overlap >= 0.25:
                candidates.append((overlap, fact))
        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1].get("updated", "")), reverse=True)
            fact = candidates[0][1]
            return f'前に教えてくれた内容だと、「{fact.get("value", "")}」だよ。'

        turns = self.context_memory.get("turns", [])
        if turns and re.search(r"(さっき|前に|前の会話|何て言った|何と言った|何を話した)", q):
            latest = turns[-1]
            return f'直近の会話では、あなたは「{latest.get("user", "")}"と話していたよ。'
        return None

    def _teach(self, text):
        match = re.match(
            r"^覚えて[：:]\s*(.+?)\s*(?:=>|＝|->|→)\s*(.+)$",
            text.strip(),
        )
        if not match:
            return None

        question = match.group(1).strip()
        answer = match.group(2).strip()
        if (
                not question
                or not answer
                or len(question) > 200
                or len(answer) > 2000
        ):
            return "その内容は保存できなかったよ。質問と返答を短くしてね。"

        key = self._normalize(question)
        self.knowledge["pairs"][key] = {
            "question": question,
            "answer": answer,
            "confidence": 1.0,
            "source": "user_taught",
            "sources": [],
            "updated": datetime.now().isoformat(timespec="seconds"),
        }
        self.knowledge["counts"][key] = (
                self.knowledge["counts"].get(key, 0) + 1
        )
        self._save_knowledge()
        return f'覚えたよ！ 次に「{question}」と聞かれたら、その返答を使うね。'

    @staticmethod
    def _tokens(text):
        # 軽量な関連度推定。意味を完全に理解する処理ではない。
        text = text.lower()
        words = re.findall(
            r"[a-z0-9][a-z0-9._+-]*|[\u3040-\u30ff\u4e00-\u9fff]{2,}",
            text,
        )
        expanded = []
        for word in words:
            expanded.append(word)
            if len(word) > 3:
                expanded.extend(
                    word[index:index + 2]
                    for index in range(len(word) - 1)
                )
        return set(expanded)

    def _web_search(self, query):
        url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode(
            {"q": query}
        )
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/124 Safari/537.36"
                )
            },
        )
        with urllib.request.urlopen(request, timeout=4) as response:
            raw = response.read(2_000_000).decode("utf-8", "replace")

        parser = _SearchParser()
        parser.feed(raw)
        results = []

        for item in parser.items:
            title = html.unescape(item.get("title", "")).strip()
            link = html.unescape(item.get("url", "")).strip()
            snippet = html.unescape(item.get("snippet", "")).strip()

            if link.startswith("//"):
                link = "https:" + link
            if link.startswith("/l/?"):
                query_string = urllib.parse.parse_qs(
                    urllib.parse.urlparse(link).query
                )
                link = query_string.get("uddg", [link])[0]

            if (
                    title
                    and link.startswith("http")
                    and not any(result["url"] == link for result in results)
            ):
                results.append(
                    {"title": title, "url": link, "snippet": snippet}
                )
            if len(results) >= 7:
                break

        return results

    def _research_answer(self, query):
        try:
            results = self._web_search(query)
        except Exception as error:
            return (
                None,
                0.0,
                [],
                "Web検索に接続できなかったよ"
                f"（{type(error).__name__}）。インターネット接続を確認して、"
                "もう一度試してね。",
            )

        if not results:
            return (
                None,
                0.0,
                [],
                "検索結果を取得できなかったよ。少し言い方を変えて質問してみてね。",
            )

        query_tokens = self._tokens(query)
        ranked = []
        for result in results:
            result_text = result["title"] + " " + result["snippet"]
            result_tokens = self._tokens(result_text)
            overlap = len(query_tokens & result_tokens) / max(
                1, len(query_tokens)
            )
            result["relevance"] = round(overlap, 3)
            ranked.append(result)

        ranked.sort(
            key=lambda item: (
                item["relevance"],
                bool(item["snippet"]),
            ),
            reverse=True,
        )
        useful = [
            item
            for item in ranked
            if item["relevance"] >= 0.08
               and (item["snippet"] or item["title"])
        ][:4]
        if not useful:
            useful = ranked[:3]

        # 信頼度は関連度と情報源の多様性に基づく簡易スコア。
        domains = []
        for result in useful:
            try:
                domains.append(
                    urllib.parse.urlparse(result["url"])
                    .netloc.lower()
                    .removeprefix("www.")
                )
            except Exception:
                pass

        diversity = len(set(domains))
        relevance = sum(
            result["relevance"] for result in useful[:3]
        ) / max(1, min(3, len(useful)))
        confidence = (
            min(
                0.88,
                0.28
                + min(relevance, 0.5) * 0.7
                + max(0, diversity - 1) * 0.09,
                )
            if useful
            else 0.0
        )

        lines = []
        for result in useful[:3]:
            detail = result["snippet"] or result["title"]
            detail = re.sub(r"\s+", " ", detail).strip()
            if detail:
                lines.append(
                    f'・{detail}\n  出典: {result["title"]} — {result["url"]}'
                )

        if not lines:
            return (
                None,
                confidence,
                useful,
                "検索結果は見つかったけれど、回答に使える説明文を抽出できなかったよ。",
            )

        answer = "Web検索で見つかった情報をまとめるね。\n\n"
        answer += "\n\n".join(lines)
        if diversity < 2:
            answer += (
                "\n\n※確認できた情報源が限られているため、"
                "内容は未確認の候補として扱うよ。"
            )
        else:
            answer += (
                "\n\n※検索結果の要約だよ。重要な判断では出典ページも確認してね。"
            )

        return answer, confidence, useful, None

    def _save_research_candidate(self, query, answer, confidence, sources):
        key = self._normalize(query)
        now = datetime.now().isoformat(timespec="seconds")
        source_data = [
            {
                "title": source.get("title", ""),
                "url": source.get("url", ""),
                "relevance": source.get("relevance", 0),
            }
            for source in sources[:5]
        ]
        item = {
            "question": query,
            "answer": answer,
            "confidence": round(confidence, 3),
            "sources": source_data,
            "timestamp": now,
            "status": "candidate",
        }
        self.research_log.append(item)
        self._save_research()

        record = next(
            (
                entry
                for entry in self.unlearned
                if isinstance(entry, dict) and entry.get("key") == key
            ),
            None,
        )
        if record:
            record["seen_count"] = int(record.get("seen_count", 1)) + 1
            record.update(
                {
                    "last_seen": now,
                    "latest_confidence": round(confidence, 3),
                    "latest_sources": source_data,
                }
            )
        else:
            self.unlearned.append(
                {
                    "key": key,
                    "text": query,
                    "mode": "web_research",
                    "seen_count": 1,
                    "confidence": round(confidence, 3),
                    "status": "candidate",
                    "first_seen": now,
                    "last_seen": now,
                    "answer_candidate": answer,
                    "sources": source_data,
                }
            )

        # 2ドメイン以上かつスコア0.68以上の場合だけ自動登録する。
        domains = set()
        for source in sources:
            try:
                domains.add(
                    urllib.parse.urlparse(source["url"])
                    .netloc.lower()
                    .removeprefix("www.")
                )
            except Exception:
                pass

        if confidence >= 0.68 and len(domains) >= 2 and answer:
            self.knowledge["pairs"][key] = {
                "question": query,
                "answer": answer,
                "confidence": round(confidence, 3),
                "source": "web_auto",
                "sources": source_data,
                "updated": now,
            }
            self.knowledge["counts"][key] = (
                    self.knowledge["counts"].get(key, 0) + 1
            )
            item["status"] = "auto_registered"
            self._save_knowledge()
            self._save_research()
            if record:
                record["status"] = "auto_registered"
            else:
                self.unlearned[-1]["status"] = "auto_registered"

        self._save_unlearned()

    def _should_web_search(self, query: str) -> bool:
        """明示的に外部情報が必要な質問だけWeb検索する。"""
        query = query.strip()
        if not query:
            return False

        explicit_patterns = (
            r"https?://|www\.",
            r"(検索して|検索してみて|ウェブ検索|web検索|ネットで調べて|調べて|調査して|詳しく調査|情報を集めて)",
            r"(最新|今日のニュース|最新ニュース|最近のニュース|速報|現在の.*(価格|値段|状況|バージョン|天気)|今の天気|今日の天気)",
            r"(発売日|販売価格|現在価格|株価|為替レート|公式サイト|公式ページ|営業時間|運行状況)",
        )
        return any(
            re.search(pattern, query, re.IGNORECASE)
            for pattern in explicit_patterns
        )

    def respond(self, text: str, mode: str = "一般モード", context=None) -> str:
        raw = text.strip()
        if not raw:
            return "何か話しかけてみてね。"

        # 1. まずユーザーが新しく教えた事実を判定する。
        # 記憶検索を先にすると「俺の好きな動物はライオンです」を質問と誤認し、
        # 古い「犬」の記憶を返してしまうため、明示的な新情報を先に保存する。
        taught_fact = self._detect_fact(raw)
        if (
                taught_fact is not None
                and taught_fact[0] != "会話メモ"
                and not re.search(r"[?？]", raw)
        ):
            label, value = taught_fact
            acknowledgements = {
                "好きな色": f"{value}が好きなんだね！ 覚えたよ。",
                "好きな動物": f"好きな動物は{value}なんだね！ 記憶を更新したよ。",
                "好きなもの": f"{value}が好きなんだね！ 覚えたよ。",
                "嫌いなもの": f"{value}が嫌いなんだね！ 覚えたよ。",
                "苦手なもの": f"{value}が苦手なんだね！ 覚えたよ。",
                "得意なもの": f"{value}が得意なんだね！ 覚えたよ。",
            }
            answer = acknowledgements.get(
                label, f"{label}は{value}だね！ 記憶したよ。"
            )
            self._remember_turn(raw, answer)
            return answer

        # 2. 保存済みのユーザー記憶を確認する。Web検索より優先。
        remembered = self._memory_answer(raw)
        if remembered is not None:
            self._remember_turn(raw, remembered)
            return remembered

        # 3. ユーザーが明示的に教えた知識を保存する。
        taught = self._teach(raw)
        if taught is not None:
            self._remember_turn(raw, taught)
            return taught

        # 4. 登録済みの質問と回答を確認する。
        key = self._normalize(raw)
        pair = self.knowledge["pairs"].get(key)
        if pair:
            self.knowledge["counts"][key] = (
                    self.knowledge["counts"].get(key, 0) + 1
            )
            self._save_knowledge()
            answer = pair.get("answer", "まだ返答が登録されていないよ。")
            self._remember_turn(raw, answer)
            return answer

        now = datetime.now()

        if self._is_greeting(raw):
            answer = "こんにちは！ AngelWhiteだよ。何について話そうか？"
        elif re.search(r"(ありがとう|ありがと|助かった|感謝)", raw):
            answer = "どういたしまして！ またいつでも話しかけてね。"
        elif re.search(r"(さようなら|じゃあね|またね|おやすみ)", raw):
            answer = "またね！ いつでも話しかけてね。"
        elif re.search(r"(君の名前|あなたの名前|名前は何|誰なの|何者)", raw):
            answer = (
                "僕はAngelWhite。独自の回答エンジンと知識記憶を"
                "少しずつ開発しているよ。"
            )
        elif re.search(r"(何時|今の時間|現在時刻)", raw):
            answer = f'このPCの現在時刻は {now.strftime("%H:%M")} だよ。'
        elif re.search(r"(今日の日付|今日は何日|何月何日|今日の日にち)", raw):
            answer = f'今日は {now.strftime("%Y年%m月%d日")} だよ。'
        elif re.search(r"^覚えて[：:]?", raw):
            answer = "覚えたい内容は「覚えて: 質問 => 返答」の形で教えてね。"
        elif re.search(
                r"(?:俺|僕|私|わたし|自分)は.{1,80}(?:が好き|が嫌い|が苦手|が得意)",
                raw,
        ) or re.search(r"^.{1,30}が好き(?:です|だよ|だ|！|!|。)?$", raw):
            # この返答をした後、_remember_turn() が発言内容からプロフィール情報を保存する。
            detected = self._detect_fact(raw)
            if detected:
                label, value = detected
                if label == "好きな色":
                    answer = f"{value}が好きなんだね！ 教えてくれてありがとう。"
                elif label == "好きなもの":
                    answer = f"{value}が好きなんだね！ 教えてくれてありがとう。"
                else:
                    answer = "そうなんだね！ 教えてくれてありがとう。"
            else:
                match = re.search(
                    r"(?:俺|僕|私|わたし|自分)は(.{1,60}?)(が好き|が嫌い|が苦手|が得意)",
                    raw,
                )
                if match:
                    reactions = {
                        "が好き": "が好きなんだね！ 教えてくれてありがとう。",
                        "が嫌い": "は嫌いなんだね。話してくれてありがとう。",
                        "が苦手": "が苦手なんだね。教えてくれてありがとう。",
                        "が得意": "が得意なんだね、いいね！",
                    }
                    answer = f"{match.group(1).strip()}{reactions[match.group(2)]}"
                else:
                    answer = "そうなんだね！ 教えてくれてありがとう。"
        elif self._should_web_search(raw):
            researched, confidence, sources, error = self._research_answer(raw)
            if researched:
                answer = researched
                self._save_research_candidate(raw, answer, confidence, sources)
                if mode == "思考モード":
                    answer += (
                        "\n\n検索・評価メモ：関連度と情報源の数から推定した"
                        f"信頼度スコア {confidence:.2f}"
                        "（確率として保証された値ではないよ）。"
                    )
            else:
                answer = (
                                 error or "Web検索で情報を取得できなかったよ。"
                         ) + "\n\n質問の内容を変えるか、少し時間をおいてもう一度試してね。"
                self._save_research_candidate(raw, answer, confidence, sources)
        else:
            answer = (
                "その質問は、今の僕の内蔵知識だけでは正確に答えられないみたい。\n"
                "Web検索が必要なら「〇〇を検索して」や"
                "「〇〇の最新情報を調べて」と頼んでね。"
            )
            key = self._normalize(raw)
            self.unlearned.append(
                {
                    "key": key,
                    "text": raw,
                    "mode": "local_fallback",
                    "seen_count": 1,
                    "confidence": 0.0,
                    "status": "candidate",
                    "first_seen": now.isoformat(timespec="seconds"),
                    "last_seen": now.isoformat(timespec="seconds"),
                    "answer_candidate": answer,
                    "sources": [],
                }
            )
            self._save_unlearned()

        # 4. 発言を履歴に保存し、プロフィール情報を抽出する。
        self._remember_turn(raw, answer)
        return answer
