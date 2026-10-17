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
import random
from difflib import SequenceMatcher
import urllib.parse
import urllib.request
from datetime import datetime
from html.parser import HTMLParser
import xml.etree.ElementTree as ET
from pathlib import Path


class _SearchParser(HTMLParser):
    """検索結果HTMLからリンクを抽出する。既知のクラス名と予備候補の両方を扱う。"""
    def __init__(self):
        super().__init__()
        self.items = []
        self.fallback_items = []
        self.current = None
        self.capture = None
        self.buf = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get("class", "") or ""
        if tag == "a" and attrs.get("href"):
            is_known_result = (
                    "result__a" in classes
                    or "result-link" in classes
                    or "result__title" in classes
                    or "web-result" in classes
            )
            item = {
                "title": "",
                "url": attrs.get("href", ""),
                "snippet": "",
                "_known_result": is_known_result,
            }
            (self.items if is_known_result else self.fallback_items).append(item)
            self.current = item
            self.capture = "title"
            self.buf = []
        elif self.current and tag in ("a", "div", "td", "span") and (
                "result__snippet" in classes
                or "result-snippet" in classes
                or "b_caption" in classes
                or "snippet" in classes
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
        elif self.capture == "snippet" and tag in ("div", "td", "span") and self.current:
            self.current["snippet"] = " ".join("".join(self.buf).split())
            self.capture = None
            self.buf = []


class JapaneseAI:
    # 日本語の通常文ではほとんど使われない中国語の簡体字・繁体字を検出する。
    # 日本語でも使う漢字全体を禁止せず、明確な混入の手掛かりを使う。
    _CHINESE_ONLY_CHARS = frozenset(
        "这说为与从来对时会发后里开关国过还进远运边达实现点间东书车门问题学"
        "语汉龙马风云电气体无见听让给别种长条变数线场业产当经样现务区结论"
        "這說為與從來對時會發後裡開關國過還進遠運邊達實現點間東書車門問題學"
        "漢龍馬風雲電氣體無見聽讓給別種長條變數線場業產當經樣現務區結論"
    )

    @classmethod
    def _is_usable_japanese_candidate(cls, text: str) -> bool:
        """中国語の混入や日本語の説明になっていない断片を候補から外す。"""
        value = str(text or "").strip()
        if len(value) < 12 or any(char in cls._CHINESE_ONLY_CHARS for char in value):
            return False
        # かながなく、漢字と英数字だけでできた文は中国語の可能性が高い。
        if not re.search(r"[\u3040-\u30ff]", value):
            if re.search(r"[\u4e00-\u9fff]", value):
                if not re.search(r"(?:です|ます|である|とは|として|など|ため|から|による|された|される)", value):
                    return False
            elif re.search(r"[A-Za-z]{3,}\s+[A-Za-z]{3,}", value) and not re.search(
                    r"(?:```|print\s*\(|import\s+\w+|def\s+\w+|class\s+\w+|\.py\b)", value, re.IGNORECASE
            ):
                # 説明文が英語だけの場合は、日本語回答の候補にしない。
                return False
        return True

    # 「紫が好きです」のような文でも色として扱うための簡易色リスト。
    # 必要に応じてあとから追加できる。
    COLOR_WORDS = (
        "紫色", "紫", "青色", "青", "水色", "紺色", "紺", "赤色", "赤",
        "桃色", "ピンク", "オレンジ色", "オレンジ", "黄色", "黄",
        "緑色", "緑", "黄緑", "黒色", "黒", "白色", "白", "灰色", "グレー",
        "茶色", "ベージュ", "金色", "銀色", "虹色",
    )

    # 好きな動物の記憶を検証するための代表的な動物名・表記ゆれ。
    ANIMAL_WORDS = (
        "ハムスター", "モルモット", "チンチラ", "フクロウ", "ペンギン",
        "カピバラ", "アルパカ", "レッサーパンダ", "ホッキョクグマ",
        "シマウマ", "ライオン", "トラ", "ゾウ", "キリン", "ゴリラ",
        "チンパンジー", "オオカミ", "キツネ", "タヌキ", "リス", "ウサギ",
        "うさぎ", "ネコ", "ねこ", "猫", "イヌ", "いぬ", "犬", "鳥",
        "小鳥", "インコ", "文鳥", "カラス", "イルカ", "クジラ",
        "サメ", "金魚", "魚", "カメ", "亀", "ヘビ", "蛇", "カエル",
        "蛙", "馬", "パンダ", "クマ", "熊", "猿", "サル", "ヤギ",
        "ヒツジ", "羊", "牛", "ウシ", "豚", "ブタ", "アザラシ",
        "コアラ", "カンガルー", "ワニ", "トカゲ", "カメレオン",
        "ペット",
    )
    FACT_INSTRUCTION_WORDS = (
        "答えて", "教えてください", "教えて", "何ですか", "なんですか",
        "今まで言った", "今まで教えて", "これまで言った", "一覧にして",
        "全部答えて", "質問", "何個", "何匹", "何種類",
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
    def _clean_fact_value(cls, value):
        """プロフィール値から語尾だけを整え、質問・指示文の混入を拒否する。"""
        value = str(value or "").strip()
        value = re.sub(r"[。！!]+$", "", value).strip()
        value = re.sub(r"(?:だよ|です|だ)$", "", value).strip()
        if not value or len(value) > 80:
            return None
        # 「猫？犬？何ですか？」や、質問文を連結した誤登録を許さない。
        if re.search(r"[?？]", value):
            return None
        if any(word in value for word in cls.FACT_INSTRUCTION_WORDS):
            return None
        # 1つのプロフィール欄に複数の文が混ざっている場合も拒否。
        if re.search(r"[。\n\r]", value):
            return None
        return value

    @classmethod
    def _canonical_animal_name(cls, value):
        """動物名そのものだけを認める。文中に動物名が含まれるだけでは認めない。"""
        value = str(value or "").strip()
        value = re.sub(r"[。！!、，,．.]+$", "", value).strip()
        value = re.sub(r"(?:だよ|です|だ)$", "", value).strip()
        value = re.sub(r"[\s　]+", "", value)
        aliases = {
            "ねこ": "猫", "ネコ": "猫", "イヌ": "犬", "いぬ": "犬",
            "うさぎ": "うさぎ", "ウサギ": "うさぎ", "亀": "カメ",
            "熊": "クマ", "くま": "クマ", "象": "ゾウ", "ぞう": "ゾウ",
            "さる": "サル", "狼": "オオカミ", "おおかみ": "オオカミ",
            "きつね": "キツネ", "とら": "トラ", "ぶた": "ブタ",
            "うし": "ウシ", "ひつじ": "ヒツジ", "かえる": "カエル",
            "へび": "ヘビ", "へび": "ヘビ", "うま": "馬",
        }
        known = set(cls.ANIMAL_WORDS)
        if value in aliases:
            return aliases[value]
        # 検索文字列や文章から部分一致で抽出しない。値全体が動物名である場合のみ有効。
        if value in known:
            return aliases.get(value, value)
        return None

    @classmethod
    def _animal_list_parts(cls, value):
        """動物名だけで構成された明確な列挙なら、要素に分けて返す。"""
        value = str(value or "").strip()
        parts = [part.strip() for part in re.split(r"(?:、|,|・|と|や|および|\s+)", value) if part.strip()]
        if not parts:
            return []
        canonical = [cls._canonical_animal_name(part) for part in parts]
        if all(canonical):
            return parts
        return []

    @classmethod
    def _is_valid_animal_value(cls, value):
        """動物名単体、または動物名だけの列挙かを検証する。"""
        value = str(value or "").strip()
        if not value or len(value) > 40:
            return False
        if re.search(r"[?？\n\r]", value):
            return False
        if any(word in value for word in cls.FACT_INSTRUCTION_WORDS):
            return False
        if re.search(r"[。！？!?]", value):
            # 末尾の語尾・句読点のみを _canonical_animal_name 側で扱う。
            cleaned = re.sub(r"[。！!]+$", "", value).strip()
            if cleaned != value and cls._canonical_animal_name(cleaned):
                return True
            return False
        if cls._canonical_animal_name(value):
            return True
        return bool(cls._animal_list_parts(value))

    @classmethod
    def _split_animal_values(cls, value):
        """「猫と犬」「猫、犬」のような動物名だけの列挙を分割する。"""
        value = str(value or "").strip()
        parts = cls._animal_list_parts(value)
        if parts:
            return parts
        animal = cls._canonical_animal_name(value)
        return [animal] if animal else []

    @classmethod
    def _detect_fact(cls, text):
        """明確な本人の発言だけをプロフィールとして検出する。"""
        text = str(text or "").strip()
        if not text or len(text) > 500:
            return None
        # 句末だけでなく文中の疑問符もチェックする。
        # 質問を列挙した文章を「好きな動物」の事実として誤保存しない。
        if re.search(r"[?？]", text):
            return None
        if any(word in text for word in ("答えて", "何ですか", "なんですか", "教えてください")):
            return None
        clean = re.sub(r"[。！!]+$", "", text).strip()
        if not clean:
            return None

        # 明示的なプロフィール形式は、全文がその形に一致した場合のみ受け付ける。
        explicit_patterns = (
            (r"^(?:(?:俺|僕|私|わたし|自分)の)?好きな色は\s*(.+?)$", "好きな色"),
            (r"^(?:(?:俺|僕|私|わたし|自分)の)?好きな動物は\s*(.+?)$", "好きな動物"),
            (r"^(?:(?:俺|僕|私|わたし|自分)の)?趣味は\s*(.+?)$", "趣味"),
            (r"^(?:(?:俺|僕|私|わたし|自分)の)?目標は\s*(.+?)$", "目標"),
            (r"^(?:(?:俺|僕|私|わたし|自分)の)?夢は\s*(.+?)$", "夢"),
            (r"^(?:(?:俺|僕|私|わたし|自分)の)?名前は\s*(.+?)$", "名前"),
            (r"^(?:(?:俺|僕|私|わたし|自分)は)?\s*住んでいる場所は\s*(.+?)$", "住んでいる場所"),
        )
        for pattern, label in explicit_patterns:
            match = re.fullmatch(pattern, clean)
            if match:
                value = cls._clean_fact_value(match.group(1))
                if not value:
                    return None
                if label == "好きな動物" and not cls._is_valid_animal_value(value):
                    return None
                return label, value

        # 「猫が好き」「俺は紫が好き」「数学が苦手」など。
        preference = re.fullmatch(
            r"^(?:(俺|僕|私|わたし|自分)は)?\s*(.+?)が(好き|嫌い|苦手|得意)(?:です|だよ|だ)?$",
            clean,
        )
        if preference:
            subject = cls._clean_fact_value(preference.group(2))
            kind = preference.group(3)
            if not subject:
                return None
            if kind == "好き" and any(word in subject for word in cls.COLOR_WORDS):
                label = "好きな色"
                for color in sorted(cls.COLOR_WORDS, key=len, reverse=True):
                    if color in subject:
                        subject = color
                        break
            elif kind == "好き" and cls._is_valid_animal_value(subject):
                label = "好きな動物"
            elif kind == "好き":
                label = "好きなもの"
            else:
                label = {"嫌い": "嫌いなもの", "苦手": "苦手なもの", "得意": "得意なもの"}[kind]
            return label, subject

        return None

    def _remember_turn(self, q, a):
        if not self.memory_enabled:
            return

        now = datetime.now().isoformat(timespec="seconds")
        self.context_memory["turns"].append(
            {"user": q, "assistant": a, "timestamp": now}
        )
        # メモリ上の履歴も上限を設け、長時間利用で無制限に増えないようにする。
        self.context_memory["turns"] = self.context_memory["turns"][-300:]

        detected = self._detect_fact(q)
        if detected:
            label, value = detected
            facts = self.context_memory["facts"]
            # 同じ項目は最新の値で更新する。
            if label == "好きな動物":
                # 好きな動物は複数保存。列挙された動物は分割し、誤った値は保存しない。
                for animal in self._split_animal_values(value):
                    if not any(
                            isinstance(f, dict)
                            and f.get("label") == label
                            and str(f.get("value", "")).casefold() == animal.casefold()
                            for f in facts
                    ):
                        facts.append({"label": label, "value": animal, "updated": now})
            else:
                # それ以外のプロフィール項目は最新の値で更新する。
                facts[:] = [fact for fact in facts if fact.get("label") != label]
                facts.append({"label": label, "value": value, "updated": now})
            self.context_memory["facts"] = self.context_memory["facts"][-500:]

        self._save_memory()

    def _memory_answer(self, query):
        """質問の意図を判定し、プロフィール記憶や会話履歴から回答を組み立てる。"""
        if not self.memory_enabled:
            return None

        q = self._normalize(query)
        facts = self.context_memory.get("facts", [])
        turns = self.context_memory.get("turns", [])

        # 「今まで」「全部」「何個」などがあれば、カテゴリ名が省略されても
        # 好きな動物をまとめて尋ねている可能性を優先して扱う。
        asks_all_animals = bool(re.search(
            r"(今まで|これまで|前に|さっき).*(言った|教えた|話した|伝えた).*(全部|すべて|みんな|何|答えて|教えて)?|"
            r"(全部|すべて|みんな).*(好きな動物|動物|ペット|生き物)|"
            r"(好きな動物|好きなペット|好きな生き物).*(全部|すべて|何個|何匹|何種類|まとめて|答えて|教えて)|"
            r"(何個|何匹|何種類).*(動物|ペット|生き物|答えて|教えて)|"
            r"今まで言ったやつ|今まで教えてくれたもの",
            q,
        ))
        asks_animal = asks_all_animals or bool(re.search(
            r"(好きな動物|好きなペット|好きな生き物|何の動物が好き|動物は何が好き|"
            r"どんな動物が好き|何の動物|動物は何|動物を教えて|動物を答えて|最初.*動物|最後.*動物|最近.*動物)", q
        ))

        if asks_animal:
            animal_values = []
            animal_words = (
                "猫", "ねこ", "ネコ", "犬", "いぬ", "イヌ", "鳥", "うさぎ", "ウサギ",
                "ハムスター", "魚", "金魚", "亀", "カメ", "馬", "イルカ", "パンダ",
                "ライオン", "ペット", "きつね", "キツネ", "狼", "オオカミ", "虎", "トラ",
                "ゾウ", "象", "キリン", "うま", "ヘビ", "蛇", "カエル", "蛙",
            )
            for fact in facts:  # 保存順を維持する＝最初に教えた動物も分かる
                if not isinstance(fact, dict):
                    continue
                value = str(fact.get("value", "")).strip()
                if not value:
                    continue
                label = fact.get("label")
                is_animal_fact = label == "好きな動物" and self._is_valid_animal_value(value)
                is_animal_favorite = (
                        label == "好きなもの"
                        and self._is_valid_animal_value(value)
                        and not any(word in value for word in self.FACT_INSTRUCTION_WORDS)
                )
                if is_animal_fact or is_animal_favorite:
                    candidates = self._split_animal_values(value)
                    for candidate in candidates:
                        if not any(existing.casefold() == candidate.casefold() for existing in animal_values):
                            animal_values.append(candidate)

            if animal_values:
                if re.search(r"(最初|一番最初|初めに|はじめに|最初に教えた)", q):
                    return f"最初に教えてくれた好きな動物は{animal_values[0]}だよ！"
                if re.search(r"(最後|一番最後|最後に教えた|最近教えた)", q):
                    return f"最後に教えてくれた好きな動物は{animal_values[-1]}だよ！"
                if len(animal_values) == 1:
                    return f"今まで教えてくれた好きな動物は{animal_values[0]}だよ！"
                return "今まで教えてくれた好きな動物は、" + "、".join(animal_values) + "だよ！"

        # 色も複数保存ではなく、最新の値をプロフィールとして返す。
        if re.search(r"(好きな色|好きなカラー|何色が好き|何色好き|カラーは何色)", q):
            for fact in reversed(facts):
                if fact.get("label") == "好きな色":
                    return f"前に教えてくれた好きな色は{fact.get('value')}だよ！"

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
                        return f"前に教えてくれた{display}は{fact.get('value')}だよ！"

        if re.search(r"(好きなもの|何が好き|何がすき)", q):
            values = []
            for fact in facts:
                if fact.get("label") in ("好きなもの", "好きな動物", "好きな色"):
                    value = str(fact.get("value", "")).strip()
                    if value and value not in values:
                        values.append(value)
            if values:
                if re.search(r"(全部|すべて|今まで|これまで|まとめて)", q) and len(values) > 1:
                    return "今まで教えてくれた好きなものは、" + "、".join(values) + "だよ！"
                return f"前に教えてくれた好きなものは{values[-1]}だよ！"

        # 会話履歴を尋ねている場合、直近の一件だけでなく履歴を確認する。
        if turns and re.search(r"(さっき|前に|前の会話|何て言った|何と言った|何を話した|今まで言った|今まで教えた|言ったやつ)", q):
            animal_turns = []
            for turn in turns:
                user_text = str(turn.get("user", ""))
                detected = self._detect_fact(user_text)
                if detected and detected[0] == "好きな動物":
                    value = detected[1]
                    if value and value not in animal_turns:
                        animal_turns.append(value)
            if animal_turns and re.search(r"(動物|好き|全部|今まで|言ったやつ)", q):
                return "今まで教えてくれた好きな動物は、" + "、".join(animal_turns) + "だよ！"
            latest = turns[-1]
            return f"直近の会話では、あなたは「{latest.get('user', '')}」と話していたよ。"

        # 汎用的な記憶検索は、ユーザーが「以前の会話・記憶」を明示的に
        # 尋ねた場合にだけ実行する。単語がプロフィール値と一致しただけで、
        # 「チワワの性格は？」などの通常の知識質問を記憶回答で横取りしない。
        memory_recall_intent = bool(re.search(
            r"(前に|さっき|以前|これまで|今まで|覚えてる|覚えている|記憶してる|記憶している|"
            r"教えてくれた|言った内容|話した内容|何て言った|何と言った|前の会話|さっきの話|"
            r"前に話した|前に教えた|私について|僕について|俺について)",
            q,
        ))
        if not memory_recall_intent:
            return None

        # 明示的な記憶の質問なので、ここで初めて内容の類似度を確認する。
        query_tokens = self._tokens(q)
        candidates = []
        for fact in facts:
            if not isinstance(fact, dict):
                continue
            value = str(fact.get("value", ""))
            label = str(fact.get("label", ""))
            if label == "好きな動物" and not self._is_valid_animal_value(value):
                continue
            tokens = self._tokens(label + value)
            overlap = len(query_tokens & tokens) / max(1, len(query_tokens))
            if overlap >= 0.25:
                candidates.append((overlap, fact))
        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1].get("updated", "")), reverse=True)
            fact = candidates[0][1]
            return f"前に教えてくれた内容だと、「{fact.get('value', '')}」だよ。"
        return None

    @classmethod
    def _extract_known_topic(cls, text: str):
        """「○○って知ってる？」「○○って知ってるの？」などから話題を抽出する。"""
        raw = str(text or "").strip()
        raw = re.sub(r"[。！!？?]+$", "", raw).strip()
        # 質問の主語にAIへの呼びかけが入っていても、対象の話題だけを取り出す。
        raw = re.sub(r"^(?:ねえ|ねぇ|ところで|そういえば)", "", raw).strip()
        raw = re.sub(r"^(?:君|あなた|お前|AngelWhite|アンジェルホワイト)(?:は|が)", "", raw, flags=re.IGNORECASE).strip()
        endings = r"(?:知ってる|知ってるの|知っている|知っていますか|知ってますか?|聞いたことある(?:の)?|分かる|わかる)(?:の)?"
        patterns = (
            rf"^(.+?)(?:って|とは|について|のこと(?:を|は)?)\s*{endings}$",
            rf"^(.+?)\s*{endings}$",
        )
        topic = None
        for pattern in patterns:
            match = re.fullmatch(pattern, raw, flags=re.IGNORECASE)
            if match:
                topic = match.group(1).strip()
                break
        if not topic:
            return None
        topic = topic.strip(" \t　、。？！?!『』「」")
        if not topic or len(topic) > 70 or topic in {"何か", "なにか", "誰か", "だれか", "何", "なに", "君", "あなた"}:
            return None
        if re.search(r"(返し方|教えて|調べて|答えて|何について)", topic):
            return None
        return topic

    @classmethod
    def _extract_standalone_topic(cls, text: str):
        """単語・短い名詞句だけの入力を、説明してほしい話題として判定する。

        例: 「チワワ」「橋」「量子コンピュータ」「チワワの飼い方」
        文として成立している雑談、挨拶、ユーザーへの依頼・質問は除外する。
        """
        raw = str(text or "").strip()
        if not raw or len(raw) > 48:
            return None
        if re.search(r"[\s\u3000\r\n\t。！？?!,，、:：;；]", raw):
            return None
        # 「犬が好き」「ゲームが苦手」など、短くても文として成立する
        # 好み・感情の共有を、単語だけの検索トピックとして扱わない。
        # メモリ機能がOFFでも雑談ルートへ進めるよう、ここで先に除外する。
        if re.search(
                r"(?:が|は)(?:大好き|好き|嫌い|苦手|得意|かわいい|可愛い|かっこいい|カッコいい|"
                r"面白い|おもしろい|楽しい|楽しかった|うれしい|嬉しい|最高|怖い|こわい|"
                r"好きじゃない|嫌いじゃない)(?:です|だよ|だね|だ|なんだ|なんです)?$",
                raw,
        ):
            return None
        normalized = raw.lower()
        ignored = {
            "おはよう", "こんにちは", "こんばんは", "やあ", "お疲れ", "お疲れ様",
            "ありがとう", "ありがと", "どういたしまして", "よろしく", "よろしくね",
            "またね", "じゃあね", "おやすみ", "うん", "ううん", "はい", "いいえ",
            "そうなんだ", "そうだね", "なるほど", "いいね", "すごい", "了解", "ok",
            "ありがとうね", "たしかに", "ごめん", "ごめんなさい", "すみません",
            "ねえ", "ねぇ", "へえ", "へぇ", "ふむ", "あの", "まあ", "わあ", "うわ",
            "今日", "きょう", "明日", "あした", "昨日", "きのう", "今", "さっき",
            "これ", "それ", "あれ", "ここ", "そこ", "あそこ", "なに", "何", "誰", "どこ", "いつ",
            "1", "2", "3", "１", "２", "３",
        }
        if normalized in ignored:
            return None
        # 会話文・動作・質問・検索命令らしい末尾は単語検索にしない。
        if re.search(
                r"(?:って|とは|について|知ってる|知っている|聞いたことある|分かる|わかる|"
                r"だよね|だね|だよ|だな|なんだ|なんだよ|です|ます|でした|だった|だっけ|"
                r"してる|してた|やってる|やってた|遊んでる|遊んでた|作ってる|作ってた|"
                r"している|した|する|して|疲れた|つかれた|眠い|ねむい|楽しい|楽しかった|"
                r"かわいい|可愛い|かっこいい|すごい|面白い|おもしろい|悲しい|うれしい|嬉しい|"
                r"してほしい|ほしい|欲しい|ください|下さい|教えて|説明して|調べて|検索して|"
                r"知りたい|思う|感じる|よね|よな|かな|かも|でしょう|"
                r"は|が|を|に|で|と|の|へ|も)$",
                raw,
        ):
            return None
        if re.match(r"^(?:俺|僕|私|わたし|自分|君|あなた|お前)の?", raw):
            return None
        # 短い相づちは上の ignored に列挙して除外する。ひらがな・カタカナだけの
        # 単語でも、チワワ・アイス・アニメなどの話題語は検索対象にする。
        return raw

    @classmethod
    def _extract_focused_question(cls, text: str):
        """「チワワの被毛にはどんな種類がある？」のような対象+焦点の質問を抽出する。

        戻り値は (対象, 焦点ラベル, 焦点に関係する語句)。雑談や単なる相づちは対象外。
        """
        raw = str(text or "").strip()
        if not raw or len(raw) > 120:
            return None
        # 疑問符や「どんな」が省略された短い質問も拾う。
        # 例: 「チワワの性格は」「橋の仕組み」「Pythonの使い方」
        bare_focus_question = bool(re.search(
            r"(?:被毛|毛の種類|毛質|毛の長さ|性格|気質|飼い方|飼育方法|健康管理|育て方|"
            r"平均寿命|寿命|体重|体高|大きさ|サイズ|原産地|原産国|歴史|由来|起源|"
            r"構造|仕組み|原理|特徴|特長|種類|タイプ|分類|使い方|利用方法|違い|"
            r"原因|理由|値段|価格|費用|料金|用途|役割|仕組み)"
            r"(?:は|って|とは|について|のは)?$",
            raw,
            re.IGNORECASE,
        ))
        question_like = bool(
            re.search(r"[?？]$", raw)
            or re.search(r"(教えて|知りたい|説明して|調べて|どんな|どのような|何種類|どう違う|どれくらい)", raw)
            or bare_focus_question
        )
        if not question_like:
            return None

        focus_patterns = (
            ("被毛・毛の種類", r"被毛|毛の種類|毛質|毛の長さ|被毛の種類"),
            ("性格", r"性格|気質| temperament"),
            ("飼い方・健康管理", r"飼い方|飼育方法|健康管理|育て方"),
            ("平均寿命", r"平均寿命|寿命"),
            ("体重・大きさ", r"体重|体高|大きさ|サイズ"),
            ("原産地・歴史", r"原産地|原産国|歴史|由来|起源"),
            ("構造・仕組み", r"構造|仕組み|原理|どう動く|どう機能"),
            ("特徴", r"特徴|特長"),
            ("種類", r"種類|タイプ|分類"),
            ("使い方", r"使い方|利用方法|どう使う"),
            ("違い", r"違い|どう違う|比較"),
            ("原因・理由", r"原因|理由|なぜ|どうして"),
            ("値段・費用", r"値段|価格|費用|料金"),
            ("用途・役割", r"用途|役割|何に使う|何に使われ"),
        )
        # 最も具体的な焦点を先に探す。
        for label, pattern in focus_patterns:
            match = re.search(pattern, raw, re.IGNORECASE)
            if not match:
                continue
            prefix = raw[:match.start()].strip()
            prefix = re.sub(r"(?:には|では|とは|の|は|が|を|に|で|と|について|って|、|,|：|:)\s*$", "", prefix).strip()
            prefix = re.sub(r"^(?:ねえ|ねぇ|ところで|ちなみに|教えて|知りたいんだけど)\s*", "", prefix)
            prefix = re.sub(r"[「」『』（）()\s　]+", "", prefix)
            if not prefix or len(prefix) > 48:
                continue
            # 「チワワの被毛にはどんな種類がある？」のような質問だけ通し、
            # 「性格がつらい」など焦点語を含む通常の会話は通さない。
            remainder = raw[match.end():]
            if not re.search(r"(どんな|どのような|何種類|何が|何ですか|何です|ある|います|ですか|でしょうか|教えて|知りたい|どう違う|どれくらい|？|\?)", remainder):
                # 「チワワの性格？」「チワワの性格は」のような省略質問も受け入れる。
                bare_remainder = re.fullmatch(r"(?:は|って|とは|について|のは)?", remainder.strip())
                if not re.search(r"[?？]$", raw) and not bare_remainder:
                    continue
            return prefix, label, match.group(0)

        # 数量・意味など焦点の一般パターン。回答用の分類だけを足し、雑談を検索扱いにしない。
        generic_match = re.search(r"(何に使う|どういう意味|どういう仕組み|どんな仕組み|どんな種類|どのような種類|何種類|何が特徴|何でできている|何からできている)", raw)
        if generic_match:
            prefix = raw[:generic_match.start()].strip()
            prefix = re.sub(r"(?:には|では|とは|の|は|が|を|に|で|と|について|って|、|,|：|:)\s*$", "", prefix).strip()
            if prefix:
                label = generic_match.group(0)
                return prefix, label, label
        return None

    @staticmethod
    def _fetch_page_paragraphs(url: str, max_bytes: int = 800_000):
        """検索結果のリンク先本文から、見出し・段落・箇条書きを抽出する。

        ページ取得に失敗した場合は空配列を返し、検索スニペットだけで回答を続ける。
        """
        if not url:
            return []
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/126.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "ja-JP,ja;q=0.9,en;q=0.7",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=2.5) as response:
                content_type = response.headers.get("Content-Type", "")
                if content_type and not any(kind in content_type.lower() for kind in ("text/html", "application/xhtml+xml")):
                    return []
                raw = response.read(max_bytes).decode("utf-8", "replace")
        except Exception:
            return []
        # スクリプト等を段落解析前に除去する。
        raw = re.sub(r"(?is)<(script|style|noscript|svg|nav|footer|header|aside|form)\b[^>]*>.*?</\1\s*>", " ", raw)
        blocks = re.findall(
            r"(?is)<(?:h1|h2|h3|h4|p|li|dt|dd)\b[^>]*>(.*?)</(?:h1|h2|h3|h4|p|li|dt|dd)\s*>",
            raw,
        )
        paragraphs = []
        for block in blocks:
            value = html.unescape(re.sub(r"(?s)<[^>]+>", " ", block))
            value = re.sub(r"\s+", " ", value).strip()
            if len(value) < 24:
                continue
            if re.search(r"(cookieを|プライバシーポリシー|ログイン|会員登録|関連記事|あわせて読みたい|広告|スポンサーリンク)", value, re.IGNORECASE):
                continue
            if value not in paragraphs:
                paragraphs.append(value[:900])
            if len(paragraphs) >= 55:
                break
        return paragraphs

    @classmethod
    def _build_topic_overview(cls, topic: str, sources, fetched_paragraphs=None):
        """検索本文から、単独でも意味が通る説明文を選び、項目ごとに整理する。

        生成モデルではなく抽出型要約。前後関係がないと意味が通らない断片、
        宣伝文、誤解を招く断定、重複文は回答本文から除外する。
        """
        fetched_paragraphs = fetched_paragraphs or {}
        topic = str(topic or "").strip()
        topic_tokens = cls._tokens(topic)
        candidates = []
        seen = set()

        # 同じ表記の別話題を混ぜないため、入力トピックの分野を軽量に推定する。
        # 例: 「チワワ」では犬種が対象なので、同名の楽曲・歌手の記事内記述を除外する。
        animal_topic = bool(re.search(
            r"(チワワ|犬|猫|ネコ|ねこ|柴犬|トイプードル|ポメラニアン|ダックス|"
            r"ゴールデンレトリバー|ラブラドール|コーギー|パグ|ビーグル|ボーダーコリー|"
            r"ハムスター|うさぎ|ウサギ|フェレット|インコ|文鳥|ペット|犬種|猫種)",
            topic,
            re.IGNORECASE,
        ))
        animal_context_pattern = re.compile(
            r"(犬種|小型犬|超小型犬|大型犬|犬|飼い主|飼育|被毛|毛色|体重|体高|"
            r"散歩|しつけ|犬小屋|ペット|寿命|関節|動物病院|原産地|毛並み|"
            r"運動量|食事|餌|エサ|健康管理|犬の|チワワは|チワワが|チワワの)",
            re.IGNORECASE,
        )
        same_name_drift_pattern = re.compile(
            r"(DJ\s*BOBO|CMソング|邦題|歌詞|楽曲|シングル|音楽チャート|"
            r"ミュージックビデオ|歌手|曲として|曲が.{0,15}(?:ヒット|使われ)|"
            r"(?:スペイン|ヨーロッパ).{0,20}(?:ヒット|CMソング|大ヒット)|"
            r"(?:大ヒット|ヒット曲).{0,20}(?:曲|歌|ソング))",
            re.IGNORECASE,
        )
        weak_editorial_pattern = re.compile(
            r"(大きな瞳|コンパクトなボディ|誰よりも飼い主|愛くるしい表情|"
            r"世界中にファン|小柄な見た目で愛くるしい|人気犬種.{0,20}(?:魅力|人気))"
        )
        # サイトの宣伝・販売案内を、対象そのものの説明として誤採用しない。
        # 「チワワ」の検索で、チワワ専門販売サイトのロゴ説明やブリーダー紹介が
        # 犬種の基本情報として混ざる問題を防ぐ。
        promotional_patterns = re.compile(
            r"(チワワnowa|ブリーダーとして|ブリーダー専門サイト|ブリーダーさんから購入|"
            r"購入するメリット|子犬なら|子犬一覧|ブリーダーさんを.{0,20}(?:紹介|取材)|"
            r"実際に取材し|ロゴは|ロゴ.{0,20}表現|サイトでできること|できることや|"
            r"ご紹介します|ご紹介しています|お問い合わせ|無料相談|販売中|購入はこちら|"
            r"当サイト|このサイトでは|サービスのご案内|資料請求)",
            re.IGNORECASE,
        )

        def is_promotional(value):
            value = str(value or "")
            return bool(promotional_patterns.search(value))

        # 文頭に来ると意味が欠けやすい接続表現・代名詞的表現。
        context_dependent_start = re.compile(
            r"^(?:そのため|その結果|このため|その理由から|こうした理由から|"
            r"それにより|それによって|その一方で|一方で|しかし|ところが|"
            r"そこで|だから|つまり|また|さらに|なお|このように|このことから|"
            r"その名のとおり|その名は|そのために|このために|上記の|前述の|"
            r"これらの|このような|そのような|その中でも|その後|同じく)[、，]?"
        )
        # 宣伝・煽り・根拠の薄い動物行動論などは本文説明の候補にしない。
        low_quality_patterns = re.compile(
            r"(主従関係|自分は大型犬だ|誰よりも飼い主|世界中にファン|"
            r"驚くほど繊細|見た目とは裏腹に|絶対に.{0,10}(?:大丈夫|安全)|"
            r"最強の|最高の|必ず.{0,8}(?:勝|治|成功)|あなたも.{0,8}博士|"
            r"今すぐ.{0,10}(?:クリック|購入)|ランキング第|おすすめ.{0,8}商品|"
            r"もはや.{0,20}将来の技術|現在進行形の産業革命|世界に告げるものだった|"
            r"未来の技術.{0,20}現在|世紀の大発見)"
        )
        navigation_patterns = re.compile(
            r"(この記事では|続きはこちら|詳細はこちら|クリックして|スポンサーリンク|"
            r"ログイン|会員登録|関連記事|あわせて読みたい|広告|目次|"
            r"アイキャッチ画像出典|ENGLISH DESCRIPTION|この章では|この講座では|"
            r"本講座では|本記事では|この記事の目的|このページでは|このニュースは|"
            r"このコラムでは|ここでは.{0,20}(?:解説|紹介|見ていき)|"
            r"詳しく見ていきましょう|を詳しく見ていきましょう|"
            r"エンジニアになるにはどんなスキルが必要ですか|最大の違いは何ですか|"
            r"この講座では.{0,60}(?:目標|最低限な操作|チュートリアル)|"
            r"プログラミング未経験者を対象に|初心者を対象に.{0,40}(?:講座|チュートリアル)|"
            r"(?:Pythonの)?チュートリアルでは.{0,60}(?:学べ|説明|解説)|"
            r"この入門講座では|この講座で.{0,40}(?:学べ|取り組)|"
            r"一般的なPythonの入門書に取り組むと)"
        )
        editorial_patterns = re.compile(
            r"(誰よりも|世界中にファン|そんな魅力で|愛らしさで|かわいさから|"
            r"魅力で知られる|大人気の|驚くほど|最高にかわいい|"
            r"この章では|この講座では|このページでは|本記事では|このニュースは|"
            r"この講座では.{0,30}(?:目標|解説|紹介)|講座形式で解説|"
            r"チュートリアル形式で解説|対象としています)"
        )
        # 目次・導入メニュー・文の途中で切れた断片を説明文として採用しない。
        # 例: 「Pythonとは・Pythonの導入方法・利用開始」のような見出し列。
        outline_fragment_pattern = re.compile(
            r"(?:・|•|●|○).{0,55}(?:導入方法|利用開始|目次|チュートリアル|使い方).{0,55}(?:・|•|●|○)"
            r"|(?:「[^」]{1,40}」\s*(?:とは|の導入方法|の利用開始)\s*){2,}"
        )
        incomplete_context_pattern = re.compile(
            r"(上の.{0,25}で述べたように|次の例は|以下の例|以下のように記述|"
            r"次のように記述|しばしば以下のように|詳細は以下|下記の例|"
            r"次に示す|次の項目|これらの情報|のデフォルトエンコーディングは複雑|"
            r"こうした(?:書き方|方法|仕組み|ルール|特徴)|このような(?:書き方|方法|仕組み)|"
            r"このように書くこと|こういう書き方|上記の(?:方法|例|説明)|"
            r"次のような(?:コード|書き方|例)|以下のような(?:コード|書き方|例))"
        )
        # 指示語で前文を参照する文は、抽出後に意味が欠けるため除外する。
        dangling_reference_pattern = re.compile(
            r"(?:こうした書き方|こうした方法|このような書き方|こういう書き方|"
            r"この書き方|その方法|このルール|そのようなやり方).{0,35}"
            r"(?:不可能|できません|できない|となります|となる|です|である)"
        )
        python_governance_pattern = re.compile(
            r"(?:Pythonソフトウェア財団|Python Software Foundation|"
            r"CPython.{0,35}(?:開発|保守|管理)|"
            r"(?:グローバルコミュニティ|開発者コミュニティ).{0,35}(?:開発|保守|管理))",
            re.IGNORECASE,
        )
        python_implementation_pattern = re.compile(
            r"(?:通常.{0,15}Python.{0,15}CPython|CPython.{0,45}(?:C言語|実装|バージョン)|"
            r"作者によって.{0,12}C言語で書かれたバージョン)",
            re.IGNORECASE,
        )
        legacy_python_pattern = re.compile(
            r"(?:Python\s*2(?:\.x|\.\d+)?|Python 2\.4|Python 2系|"
            r"Python 2\.xではASCII|デフォルトエンコーディング以外の文字コード.{0,100}(?:Emacs|Vim))",
            re.IGNORECASE,
        )
        python_topic = bool(re.fullmatch(r"(?:python|パイソン)(?:\s*とは)?", topic, re.IGNORECASE))
        # Pythonの一般概要では「機能の一つ」だけで概要を埋めない。
        # 言語そのものの定義・分類を優先し、個別機能は特徴欄に回す。
        python_core_definition = re.compile(
            r"(?i)(?:Python\s*(?:（パイソン）)?\s*(?:は|とは|は、|とは、)).{0,220}"
            r"(?:高水準.{0,30})?(?:汎用.{0,20})?(?:プログラミング言語|スクリプト言語|動的型付け言語)"
            r"|Python.{0,100}(?:プログラミング言語|汎用プログラミング言語)である"
        )
        python_property_only = re.compile(
            r"(ガベージコレクション|参照されなくなったオブジェクト|標準ライブラリ|"
            r"インデント|字下げ|引数に型を指定|ランタイム|実行環境|"
            r"オブジェクト指向|CPython|Pythonソフトウェア財団|"
            r"プログラマーのグローバルコミュニティ)",
            re.IGNORECASE,
        )
        generic_editorial = re.compile(
            r"(この講座|この章|このページ|本記事|この記事|この入門|"
            r"講座では|チュートリアルでは|学習がスムーズ|"
            r"要点を整理|詳しく見ていきましょう|ご紹介します|"
            r"初心者を対象|講座を目標|見ていきましょう|"
            r"結論から言うと|万能ツール|最も学びやすい言語|"
            r"初心者にとって.{0,35}(?:最も|最高に).{0,20}(?:言語|ツール)|"
            r"幅広く活用できる万能|Pythonは多方面で活用されている言語です)"
        )
        # 見出し・目次・章タイトルが文の断片として混ざるのを防ぐ。
        # 説明文ではなく「インストール方法」「使い方」のタイトルだけを除外する。
        title_fragment_pattern = re.compile(
            r"^(?:\d+(?:\.\d+)*\s*)?(?:Pythonの)?(?:インストール方法|使い方|"
            r"実行環境|基本操作|環境構築)(?:｜|\||:|：|\s).{0,120}$|"
            r"^(?:\d+(?:\.\d+)*\s*)?Pythonのインストールと実行環境.*インストール方法$",
            re.IGNORECASE,
        )
        truncated_sentence = re.compile(
            r"(?:\bEm|\bVim|\bimport|\bdef|\bclass)\s*$|"
            r"(?:のように|次の例は|次のように|以下の例|以下のように|"
            r"デフォルトエンコーディング以外の文字コード.{0,40})$"
        )

        for source_index, source in enumerate(sources[:5]):
            url = str(source.get("url", ""))
            title = cls._clean_search_text(source.get("title", ""))
            # 広告・販売サイトのページは本文・スニペットとも要約候補から外す。
            if is_promotional(title) or is_promotional(source.get("snippet", "")):
                continue
            raw_blocks = fetched_paragraphs.get(url, [])
            # 各ページ単位で本文に使える情報があるかを確認する。
            # ほかのページ本文を取得できたことを理由に、このページのスニペットを
            # 一律で無視すると、有用な検索結果が欠けるためページごとに判断する。
            blocks = [
                cls._clean_search_text(value)
                for value in raw_blocks
                if value and not is_promotional(value)
            ]
            if not blocks:
                snippet = cls._clean_search_text(source.get("snippet", ""))
                blocks = [snippet] if snippet and not is_promotional(snippet) else []
            if not blocks:
                continue
            for block in blocks:
                block = cls._clean_search_text(block)
                if not block or is_promotional(block):
                    continue
                # 段落から文を切り出す。句点がないスニペットは後段の条件で厳しめに扱う。
                parts = re.split(r"(?<=[。！？!?])\s*", block)
                if len(parts) <= 1 and len(block) > 360:
                    parts = re.split(r"(?<=[、，,])\s*", block)
                for part in parts:
                    part = re.sub(r"^[-・•●○\s]+", "", part).strip()
                    part = re.sub(r"[\t\r\n]+", " ", part)
                    # 中国語の検索結果が回答に混ざるのを防ぐ。
                    if not cls._is_usable_japanese_candidate(part):
                        continue
                    # 本文に付いた [5][6] などの脚注記号を除去する。
                    part = re.sub(r"\[\s*\d+(?:\s*[,;、]\s*\d+)*\s*\]", "", part)
                    # Wiki系の脚注記号 [ † 1 ] / [‡2] も除去する。
                    part = re.sub(r"\[\s*[†‡]\s*\d+\s*\]", "", part)
                    part = re.sub(r"\s*\[\s*[†‡]\s*\d+\s*\]", "", part)
                    part = re.sub(r"\s{2,}", " ", part).strip()
                    # HTML抽出で混ざる句読点・括弧前後の不自然な空白を整理する。
                    part = re.sub(r"\s+([、。，．。！？!?）」』】〕〉》])", r"\1", part)
                    part = re.sub(r"([（「『【〔〈《])\s+", r"\1", part)
                    # 日本語語句の途中にスクレイピング由来で入った空白を除去する。
                    part = re.sub(r"(?<=[\u3040-\u30ff\u4e00-\u9fff])\s+(?=[\u3040-\u30ff\u4e00-\u9fff])", "", part)
                    part = re.sub(r"(?<=[A-Za-z0-9])\s+(?=[（(])", "", part)
                    part = re.sub(r"\s{2,}", " ", part).strip()
                    # 検索スニペットに混ざる「概要」「基本情報」などの見出しラベルを外す。
                    part = re.sub(r"^(?:概要(?:・基本情報)?|基本情報|定義|特徴|歴史|説明)\s*[:：]?\s*", "", part, flags=re.IGNORECASE)
                    # 「注意：」など記事側のラベルを本文にそのまま出さない。
                    part = re.sub(r"^(?:注意|補足|ポイント)\s*[:：]\s*", "", part).strip()
                    # 抽出時に先頭へ混ざった語を、意味を変えず自然な語順に整える。
                    part = re.sub(
                        r"^無料で使える\s*Pythonはオープンソースという形で開発されており、誰でも無料でダウンロードして使うことができます。$",
                        "Pythonはオープンソースのプログラミング言語で、無料でダウンロードして利用できます。",
                        part,
                    )
                    # FAQの質問文・見出し案内・記事のメタ文は説明内容として採用しない。
                    if part.endswith(("?", "？")) or re.search(r"(?:何ですか|ありますか|必要ですか|どんな.{0,20}ですか|最大の違いは何)[。！？!?]*$", part):
                        continue
                    # 短くても定義として完結している文は残す。
                    # 例:「橋とは川や道路などを越えるための構造物です。」
                    short_topic_definition = bool(
                        re.match(
                            rf"^\s*[「『]?{re.escape(topic)}[」』]?(?:は|とは|は、|とは、)",
                            part,
                            re.IGNORECASE,
                        )
                        and re.search(
                            r"(犬種|猫種|構造物|プログラミング言語|言語|コンピュータ|装置|技術|現象|一種|種類|を指す|の総称|を越えるための構造)",
                            part,
                        )
                        and len(part) >= 14
                    )
                    if len(part) < 24 and not short_topic_definition:
                        continue
                    if generic_editorial.search(part) or truncated_sentence.search(part):
                        continue
                    if title_fragment_pattern.search(part):
                        continue
                    # Pythonの一般概要では、導入操作やOSごとのコマンドなど
                    # 「始め方」に属する断片を主な特徴として混ぜない。
                    if python_topic and re.search(
                            r"(ターミナルで.{0,30}(?:入力|起動)|REPL|対話型シェル|"
                            r"Windowsでは\s*python|macOS.{0,15}Linuxでは\s*python3|"
                            r"インストール方法|環境構築から基本操作)",
                            part,
                            re.IGNORECASE,
                    ):
                        continue
                    # 文末まで取れていても、記事タイトルをつないだような断片は除外。
                    if re.search(r"(?:^|\s)\d+(?:\.\d+)+\s*Pythonの.{0,70}(?:｜|\||　|\s)\d+(?:\.\d+)*", part):
                        continue
                    # コロン区切りの目次や、複数の見出しを並べただけの断片を外す。
                    if re.fullmatch(r".{0,45}(?:とは|導入方法|利用開始|目次).{0,45}(?:・|｜|\||/).{0,70}", part):
                        continue
                    # 目次風の項目列、見出しだけの文章、例を前提とした断片を除外。
                    if outline_fragment_pattern.search(part):
                        continue
                    if incomplete_context_pattern.search(part) or dangling_reference_pattern.search(part):
                        continue
                    # 「こうした書き方は不可能」のように、参照先の例が本文から欠けた文を除外。
                    if re.search(r"(?:こうした|このような|そのような).{0,18}(?:書き方|方法|例).{0,18}(?:不可能|できない|できません)", part):
                        continue
                    # 文中に途中切れを示す省略記号や未完のコード例があれば採用しない。
                    if re.search(r"(?:…|\.\.\.)\s*$", part) or re.search(r"(?:例は|例：|例:|以下のように).{0,15}(?:Em|Vim|def |import )$", part):
                        continue
                    # 単語「Python」の一般的な説明では、Python 2.x固有の旧情報を主項目に混ぜない。
                    # 歴史やPython 2との比較を明示的に尋ねた場合は別扱い。
                    if python_topic and legacy_python_pattern.search(part):
                        continue
                    # 接続句で始まる断片は、無理に接続句だけ削って再利用せず除外する。
                    if context_dependent_start.match(part):
                        continue
                    if low_quality_patterns.search(part) or navigation_patterns.search(part):
                        continue
                    if editorial_patterns.search(part) or weak_editorial_pattern.search(part):
                        continue
                    # Wikipedia等の対象ページに含まれる「同名の曲」などの脱線を除外。
                    # 対象が犬種の場合、音楽情報そのものを犬の特徴として採用しない。
                    if animal_topic and same_name_drift_pattern.search(part) and not animal_context_pattern.search(part):
                        continue
                    # 対象語が偶然含まれるだけの関連情報を避けるため、分野の手がかりが
                    # 全くない一文は「関連する情報」候補にもしない。
                    if animal_topic and not (animal_context_pattern.search(part) or re.search(r"チワワ", part)):
                        continue
                    # 不完全な終端や省略記号は、文の欠落・途中切れである可能性が高い。
                    if part.endswith(("…", "...", "、", ",", "，", "：", ":", "や", "などの")):
                        continue
                    has_terminal = bool(re.search(r"[。！？!?]$", part))
                    has_fact_marker = bool(re.search(
                        r"(とは|である|です|原産|を指す|と呼ばれる|分類される|"
                        r"特徴がある|備える|存在する|含まれる|使われる|利用される|"
                        r"位置する|に属する|年に|世紀に|体高|体重|寿命|被毛|"
                        r"成分|構造|仕組み|役割|原因|方法|必要になる|注意が必要|"
                        r"バージョン|リリース|プリリリース|安定版|ベータ版|公開された|公開中)", part
                    ))
                    # 句点のない検索スニペットは定義・具体情報が含まれる場合のみ許可。
                    if not has_terminal and not has_fact_marker:
                        continue
                    compact = re.sub(r"[\s、。！？!?,，:：]", "", part).lower()
                    if not compact or compact in seen:
                        continue
                    seen.add(compact)

                    tokens = cls._tokens(part)
                    overlap = len(topic_tokens & tokens) / max(1, len(topic_tokens))
                    # 定義・具体的特徴を優先。topicを明確に主語にした文は高く評価する。
                    topic_pattern = re.escape(topic)
                    names_topic = bool(topic_pattern and re.search(
                        rf"^\s*[「『]?{topic_pattern}(?:（[^）]+）)?[」』]?(?:は|とは|という|[、：:])", part, re.IGNORECASE
                    ))
                    definition_bonus = 0.28 if has_fact_marker else 0.0
                    topic_bonus = 0.40 if names_topic else (0.10 if topic and topic in part else 0.0)
                    length_bonus = min(len(part), 180) / 1800.0
                    editorial_penalty = 0.15 if re.search(
                        r"(かわいい|可愛い|魅力的|人気を誇る|ファンを持つ|愛されている)", part
                    ) else 0.0
                    score = overlap * 0.5 + definition_bonus + topic_bonus + length_bonus - editorial_penalty
                    score += 0.025 if source_index == 0 else 0.0
                    candidates.append((score, source_index, part))

        candidates.sort(key=lambda item: item[0], reverse=True)
        selected_items = []
        selected_keys = set()
        selected_token_sets = []
        for score, source_index, part in candidates:
            key = re.sub(r"[\s、。！？!?,，:：]", "", part).lower()
            if not key or key in selected_keys:
                continue
            tokens = cls._tokens(part)
            # 文章間の重複はJaccard係数で判定する。短い文に共通の話題語が
            # 含まれるだけで、運動・健康など別の情報を落としてしまわないようにする。
            if any(
                    (len(tokens & previous) / max(1, len(tokens | previous))) >= 0.72
                    for previous in selected_token_sets
            ):
                continue
            if len(part) > 300:
                part = part[:297].rstrip(" 、，。") + "…"
            selected_items.append((score, source_index, part))
            selected_keys.add(key)
            selected_token_sets.append(tokens)
            # 先に全体から少数だけ選ぶと、飼い方・健康の文が上位を占めて
            # 「主な特徴」が空になることがある。分類前は多めに候補を保持し、
            # 後段で概要・特徴・注意点ごとに上限をかけて情報の偏りを防ぐ。
            if len(selected_items) >= 35:
                break

        if not selected_items:
            return (
                f"「{topic}」について検索結果は見つかったけれど、"
                "説明として単独で意味が通る情報を十分に抽出できなかったよ。\n\n"
                "別のキーワードで検索してみてね。"
            )

        # 内容を「概要」「特徴」「扱い方・注意点」に振り分ける。
        # これは取得した文を分類するだけで、新しい事実を作り出さない。
        topic_escaped = re.escape(topic)
        definition_pattern = re.compile(
            rf"^\s*[「『]?{topic_escaped}(?:（[^）]+）)?[」』]?(?:は|とは|という|として|を指す|で知られる|に分類される)",
            re.IGNORECASE,
        )
        definition_terms = re.compile(
            r"(原産|何の一種|犬種|分類される|を指す|と定義される|とは.{0,45}(?:こと|もの|一種|種類|構造物|装置|技術|言語|現象)|"
            r"として知られる|に位置する|に属する|が使われる|の総称|構造物です|乗り物です|"
            r"プログラミング言語|コンピュータの一種|計算をするコンピュータ|を越えるための構造)",
        )
        feature_terms = re.compile(
            r"(特徴|体高|体重|寿命|被毛|毛色|毛質|短毛|長毛|スムースコート|ロングコート|"
            r"大きさ|サイズ|性格|種類|立ち耳|顔立ち|頭部|瞳|耳|体形|体型|原産地|"
            r"構造|仕組み|形状|材質|成分|役割|性能|速度|長さ|高さ|面積|人口|代表例|主な|備える|持つ|存在する|"
            r"桁橋|トラス橋|アーチ橋|吊橋|吊り橋|斜張橋|ラーメン橋|橋梁|スパン|支点|橋脚|橋台|"
            r"荷重|圧縮力|引張力|曲げモーメント|軸力|部材|量子ビット|量子状態|重ね合わせ|干渉|"
            r"プログラミング言語|インタプリタ|文法|ライブラリ|フレームワーク|"
            r"読みやすい|シンプルな構文|動的型付け|オープンソース|無料で使える|"
            r"型を宣言する必要がなく|代入した値によって|インデント|字下げ|"
            r"マルチプラットフォーム|クロスプラットフォーム|拡張性|可読性|自動メモリ管理|ガベージコレクション|メモリから破棄|参照カウント|"
            r"オブジェクト指向|対話型|実行環境|実行速度|コンパイル|JIT|標準ライブラリ)",
        )
        use_case_terms = re.compile(
            r"(AI・機械学習|人工知能|機械学習|データ分析|データ解析|統計処理|"
            r"業務自動化|自動化|Webアプリ|ウェブアプリ|Web開発|ウェブ開発|"
            r"画像処理|科学技術計算|数値計算|シミュレーション|スクレイピング|"
            r"データ収集|グラフ化|売上予測|画像認識|自然言語処理|バックエンド|"
            r"利用分野|用途|活用例|活用され|使われて|使われる|利用される|利用できる|"
            r"に使われ|で使われ|分野で使|開発に使|分析に使|処理に使|作成でき|構築でき)",
            re.IGNORECASE,
        )
        care_terms = re.compile(
            r"(飼い方|運動|散歩|健康|病気|予防|治療|食事|餌|エサ|しつけ|管理|注意|安全|"
            r"必要|対策|費用|手入れ|使い方|利用方法|保守|デメリット|メリット|課題|リスク|"
            r"骨|関節|寒さ|暑さ|温度管理|熱中症|歯|目の病気|ストレス)",
        )
        overview_candidates = []
        feature_candidates = []
        use_case_candidates = []
        history_candidates = []
        version_candidates = []
        care_candidates = []
        governance_candidates = []
        other_candidates = []

        adjacent_tool_pattern = re.compile(
            r"^(?:Colab|Google Colaboratory|Jupyter(?: Notebook)?|PyCharm|VS Code|Visual Studio Code|Anaconda)\s*(?:は|とは|：|:)",
            re.IGNORECASE,
        )
        for score, source_index, part in selected_items:
            named_definition = bool(definition_pattern.search(part))
            definition_context = bool(definition_terms.search(part))
            mentions_topic = bool(topic and topic.casefold() in part.casefold())
            # 対象自体の明確な定義文を、用途語などの出現より優先して概要へ置く。
            # これで「Pythonは汎用プログラミング言語」のような文が、用途欄に流れない。
            if python_topic and python_governance_pattern.search(part):
                # 財団・CPythonコミュニティなどは、言語の特徴ではなく開発体制に分類する。
                governance_candidates.append((score + 0.25, part))
            elif python_topic and python_implementation_pattern.search(part):
                # CPythonはPythonの実装に関する補足。言語の主要な特徴と混ぜない。
                other_candidates.append((score + 0.12, part))
            elif (
                    python_topic
                    and python_property_only.search(part)
                    and not python_core_definition.search(part)
            ):
                # 「Pythonはガベージコレクションを内蔵」のような文は、
                # Python全体の定義ではなく、個別機能として特徴欄に入れる。
                feature_candidates.append((score + 0.18, part))
            elif named_definition and definition_context and not adjacent_tool_pattern.search(part):
                overview_candidates.append((score + 0.75, part))
            # 関連するサービス・開発ツールは、言語自体の機能ではなく周辺情報に分類する。
            elif adjacent_tool_pattern.search(part):
                other_candidates.append((score, part))
            # ソフトウェア等のバージョン情報は、機能説明と混ぜず独立させる。
            elif re.search(r"(バージョン|リリース|プリリリース|安定版|ベータ版|リリース候補)", part, re.IGNORECASE) and re.search(r"(?:\d{4}|\d+\.\d+)", part):
                version_candidates.append((score, part))
            # 開発者・公開年・由来などは、特徴と混ぜず背景情報として整理する。
            elif re.search(r"(開発者|によって開発|開発した|\d{4}年|\d{4}年ごろ|\d{4}年頃|初めて公開|初版|発祥|原産地|由来|歴史|祖先)", part, re.IGNORECASE):
                history_candidates.append((score, part))
            # 「何に使えるか」は定義や機能と分けて紹介し、回答の情報量を増やす。
            elif use_case_terms.search(part):
                use_case_candidates.append((score, part))
            # 具体的な飼育・健康情報や特徴を、単に「話題名は…」で始まるという
            # 理由だけで概要扱いしない。先に内容の種類を判定する。
            elif care_terms.search(part):
                care_candidates.append((score, part))
            elif feature_terms.search(part):
                # 明確な基本定義が含まれる文だけは概要に優先配置する。
                if named_definition and definition_context:
                    overview_candidates.append((score + 0.55, part))
                else:
                    feature_candidates.append((score, part))
            elif named_definition and definition_context:
                overview_candidates.append((score + 0.55, part))
            elif named_definition and mentions_topic:
                overview_candidates.append((score + 0.2, part))
            else:
                other_candidates.append((score, part))

        # 概要は「対象そのものを主語にした定義文」だけを採用する。
        # Pythonでは個別の実装機能（GC等）ではなく、言語そのものの定義があれば必ず優先。
        overview_candidates.sort(key=lambda item: item[0], reverse=True)
        if python_topic:
            core_defs = [
                (score + 2.0, sentence)
                for score, _source_index, sentence in selected_items
                if python_core_definition.search(sentence)
                   and not python_property_only.search(sentence)
                   and not generic_editorial.search(sentence)
            ]
            if core_defs:
                core_defs.sort(key=lambda item: item[0], reverse=True)
                overview = core_defs[0][1]
            else:
                # コア定義を取得できない場合、GC等の単一機能を概要とは呼ばない。
                non_property = [
                    item for item in overview_candidates
                    if not python_property_only.search(item[1])
                ]
                overview = non_property[0][1] if non_property else None
        else:
            overview = overview_candidates[0][1] if overview_candidates else None

        def unique_sentence_list(items, excluded, limit):
            result = []
            existing = {re.sub(r"\s+", "", value).casefold() for value in excluded if value}
            for _score, sentence in sorted(items, key=lambda item: item[0], reverse=True):
                key = re.sub(r"\s+", "", sentence).casefold()
                if not key or key in existing:
                    continue
                if len(sentence) > 270:
                    sentence = sentence[:267].rstrip(" 、，。") + "…"
                result.append(sentence)
                existing.add(key)
                if len(result) >= limit:
                    break
            return result

        if python_topic:
            # 開発財団・実装コミュニティの情報が特徴欄に混入しないよう二重に除外。
            feature_candidates = [
                item for item in feature_candidates
                if not python_governance_pattern.search(item[1])
            ]
            # 「Pythonは〜である」の定義文を特徴欄にも重ねない。
            feature_candidates = [
                item for item in feature_candidates
                if not python_core_definition.search(item[1])
            ]
            # 複数の似た定義・説明文から同一内容を繰り返すのを抑える。
            python_feature_seen = set()
            filtered_python_features = []
            for item in sorted(feature_candidates, key=lambda entry: entry[0], reverse=True):
                sentence = item[1]
                if python_property_only.search(sentence) and re.search(r"(ガベージコレクション|参照されなくなったオブジェクト)", sentence):
                    # メモリ管理は有用な特徴なので、概要ではなく特徴欄に1度だけ残す。
                    canonical = "memory_management"
                elif re.search(r"(標準ライブラリ|バッテリーを含む)", sentence):
                    canonical = "standard_library"
                elif re.search(r"(インデント|字下げ|可読性|読みやすい|ホワイトスペース)", sentence):
                    canonical = "indent_readability"
                elif re.search(r"(動的型付け|引数に型を指定|静的な型チェック)", sentence):
                    canonical = "typing"
                elif re.search(r"(インタプリタ|インタープリタ|実行環境|ランタイム)", sentence):
                    canonical = "runtime"
                else:
                    canonical = None
                if canonical and canonical in python_feature_seen:
                    continue
                if canonical:
                    python_feature_seen.add(canonical)
                filtered_python_features.append(item)
            feature_candidates = filtered_python_features
            # 概要にインタープリタ方式がすでに含まれる場合、同じ説明の再掲を避ける。
            if overview and re.search(r"インタプリタ|インタープリタ", overview):
                feature_candidates = [
                    item for item in feature_candidates
                    if not re.search(r"インタプリタ|インタープリタ", item[1])
                ]

        feature_list = unique_sentence_list(feature_candidates, [overview], 6)
        use_case_list = unique_sentence_list(
            use_case_candidates, [overview, *feature_list], 5
        )
        if python_topic:
            # 単語だけの「Python」説明では、Amoebaなど細かい開発環境の歴史より
            # 開発者・初期公開時期を優先する。歴史を明示的に尋ねた場合は焦点回答側で扱う。
            history_candidates = [
                item for item in history_candidates
                if not re.search(r"\bAmoeba\b|最初のバージョン.{0,35}開発", item[1], re.IGNORECASE)
            ]
        history_list = unique_sentence_list(
            history_candidates, [overview, *feature_list, *use_case_list], 3
        )
        version_list = unique_sentence_list(
            version_candidates, [overview, *feature_list, *use_case_list, *history_list], 3
        )
        care_list = unique_sentence_list(
            care_candidates, [overview, *feature_list, *use_case_list, *history_list, *version_list], 4
        )
        governance_list = unique_sentence_list(
            governance_candidates,
            [overview, *feature_list, *use_case_list, *history_list, *version_list, *care_list],
            3,
        )
        # 「関連する情報」は対象の本題から派生する情報だけに限定する。
        # 単語が一致しただけの別分野の事柄は、本文・関連情報のどちらにも出さない。
        if animal_topic:
            other_candidates = [
                item for item in other_candidates
                if animal_context_pattern.search(item[1]) and not (
                        same_name_drift_pattern.search(item[1])
                        and not re.search(r"犬|犬種|飼育|飼い主|ペット|チワワは|チワワが|チワワの", item[1])
                )
            ]
        if python_topic:
            # Python一般の説明では、講座宣伝・古いエンコーディング解説などを
            # 周辺情報として拾わず、Colabや開発環境など実用的な関連情報を優先する。
            other_candidates = [
                item for item in other_candidates
                if re.search(r"(Colab|Google Colaboratory|Jupyter|PyCharm|VS Code|Visual Studio Code|Anaconda|Python Software Foundation|Pythonソフトウェア財団|pip|仮想環境)", item[1], re.IGNORECASE)
                   and not re.search(r"(講座|チュートリアル|入門書|デフォルトエンコーディング|Python 2|Python 2\.x)", item[1], re.IGNORECASE)
            ]
        if python_topic:
            # CPythonの説明が複数ソースから重複する場合、関連技術欄には1件だけ残す。
            selected_cpython = False
            filtered_other = []
            for item in sorted(other_candidates, key=lambda entry: entry[0], reverse=True):
                sentence = item[1]
                if python_implementation_pattern.search(sentence):
                    if selected_cpython:
                        continue
                    selected_cpython = True
                filtered_other.append(item)
            other_candidates = filtered_other
        other_list = unique_sentence_list(
            other_candidates,
            [overview, *feature_list, *use_case_list, *history_list, *version_list, *care_list, *governance_list],
            3,
        )

        has_source_body = any(
            bool(fetched_paragraphs.get(str(source.get("url", "")), []))
            for source in sources[:5]
        )
        intro = (
            f"「{topic}」について、取得できたページ本文から要点を整理したよ。"
            if has_source_body
            else f"「{topic}」について、検索結果の説明文から確認できた情報を整理したよ。"
        )
        sections = [intro]
        if overview:
            sections.extend(["---", "**概要・基本情報**", overview])
        else:
            sections.extend([
                "**概要・基本情報**",
                f"取得したページの中から、「{topic}」自体を定義する十分に明確な文章は確認できなかったよ。"
                "宣伝文や話題の一部だけの文章で補わず、確認できた特徴を下にまとめるね。",
            ])
        if feature_list:
            sections.extend(["---", "**主な特徴・ポイント**"])
            sections.extend(f"・{item}" for item in feature_list)
        if use_case_list:
            sections.extend(["---", "**主な用途・活用例**"])
            sections.extend(f"・{item}" for item in use_case_list)
        if history_list:
            sections.extend(["---", "**歴史・開発の背景**"])
            sections.extend(f"・{item}" for item in history_list)
        if version_list:
            sections.extend(["---", "**バージョン・公開情報**"])
            sections.extend(f"・{item}" for item in version_list)
        if care_list:
            sections.extend(["---", "**注意点・補足**"])
            sections.extend(f"・{item}" for item in care_list)
        if governance_list:
            sections.extend(["---", "**開発体制**"])
            sections.extend(f"・{item}" for item in governance_list)
        if other_list:
            sections.extend(["---", "**関連する技術・周辺情報"])
            sections.extend(f"・{item}" for item in other_list)

        if not (feature_list or use_case_list or history_list or version_list or care_list or governance_list or other_list):
            sections.append("取得できた本文には詳しい説明が少なかったため、確認できた情報だけを載せているよ。別の検索語でも調べられるよ。")

        # 汎用的な候補だけを毎回出すのではなく、話題の種類に応じた次の探索先を示す。
        if re.search(r"(python|java|javascript|プログラミング|ソフト|アプリ|コンピュータ|ai|人工知能)", topic, re.IGNORECASE):
            sections.append("**次に調べられること**\n・始め方と環境構築\n・短いコード例\n・ライブラリや実用例")
        elif re.search(r"(犬|猫|チワワ|柴犬|トイプードル|ペット|動物)", topic):
            sections.append("**次に調べられること**\n・性格や種類の違い\n・飼い方と健康管理\n・費用や準備")
        else:
            sections.append("**次に調べられること**\n・仕組みの詳しい説明\n・種類や具体例\n・メリットと注意点")

        # 「参考にしたページ」の一覧は回答末尾に作らない。
        # 検索中のサイト名・取得件数はUI上の進捗表示で確認できる。
        sections.append("※検索で取得できた本文から要点を整理した説明だよ。ページ本文が取得できない場合は、説明に使える情報が限られることがあるよ。")
        return "\n\n".join(sections)

    @staticmethod
    def _known_topic_comment(topic: str) -> str:
        """一般的に説明しやすい話題には、短い無難なコメントを返す。"""
        q = re.sub(r"[\s　]+", "", topic).lower()
        if re.search(r"(橋|はし)$", q):
            return "川や道路、谷を渡るためのものだよね。アーチ橋や吊り橋みたいに形にもいろいろあって面白いね。"
        if re.search(r"(猫|ねこ|ネコ)$", q):
            return "自由気ままなしぐさや表情が魅力的だよね。見ているだけでも癒やされるね。"
        if re.search(r"(犬|いぬ|イヌ)$", q):
            return "人になついたり、一緒に遊んだりするところが魅力的だよね。"
        if re.search(r"(マイクラ|マインクラフト|minecraft)$", q):
            return "建築したり冒険したり、自分の好きな遊び方ができるところが面白いよね。"
        if re.search(r"(ゲーム)$", q):
            return "ジャンルによって遊び方も楽しさも違うところが面白いよね。"
        if re.search(r"(電車|列車|電車)$", q):
            return "人や荷物を運ぶ大事な乗り物だし、路線や車両ごとに特徴があるのも面白いね。"
        if re.search(r"(宇宙|星)$", q):
            return "まだ分かっていないことも多くて、考えるほど不思議で興味深い話題だよね。"
        if re.search(r"(ai|人工知能|人工知能)$", q):
            return "人の言葉を扱ったり、いろいろな作業を支援したりできる技術だね。できることと苦手なことの両方を知るのが大切だと思うよ。"
        return f"「{topic}」のことだね！どんなところが面白いのか、特徴や仕組みを一緒に見ていけそうだね。"

    @classmethod
    def _build_topic_menu(cls, topic: str, learned_answer: str = "") -> str:
        """最初に会話らしい一言を返し、ユーザーに次の行動を選んでもらう。"""
        first_line = str(learned_answer or "").strip()
        # Web検索要約のような長い回答は会話の導入として流用しない。
        if not first_line or len(first_line) > 350 or "出典:" in first_line or "検索結果" in first_line:
            first_line = f"ああ、{topic}ね！{cls._known_topic_comment(topic)}"
        # 学習回答を利用するときにも、選択肢は必ず提示する。
        return (
            f"{first_line}\n\n"
            f"よかったら「{topic}」について調べようか？それとも、次から選んでね。\n"
            f"1. {topic}の話をもっとする\n"
            f"2. Web検索をする\n"
            f"3. これで話を終わりにする\n"
            f"番号の1・2・3で選べるよ。"
        )

    @staticmethod
    def _context_messages(context):
        """main.pyから渡される現在のチャット履歴をテキスト一覧にする。"""
        messages = []
        if isinstance(context, str):
            return [context.strip()] if context.strip() else []
        if isinstance(context, (list, tuple)):
            for item in context:
                if isinstance(item, str):
                    value = item.strip()
                elif isinstance(item, dict):
                    value = str(item.get("text", item.get("content", item.get("user", "")))).strip()
                else:
                    value = str(item or "").strip()
                if value:
                    messages.append(value)
        elif isinstance(context, dict):
            raw_messages = context.get("messages", context.get("turns", []))
            if isinstance(raw_messages, (list, tuple)):
                for item in raw_messages:
                    if isinstance(item, dict):
                        value = str(item.get("text", item.get("content", item.get("user", "")))).strip()
                    else:
                        value = str(item or "").strip()
                    if value:
                        messages.append(value)
        return messages

    def _handle_topic_choice(self, text: str, context=None, progress_callback=None):
        """直前に表示した話題メニューへの数字入力を処理する。"""
        choice = str(text or "").strip().translate(str.maketrans("０１２３", "0123"))
        if choice not in {"1", "2", "3"}:
            return None

        messages = self._context_messages(context)
        # contextに今回のユーザー入力まで含まれる場合があるため、末尾だけでなく
        # 履歴内の直近のメニューを探す。
        menu_text = ""
        menu_index = -1
        for i in range(len(messages) - 1, -1, -1):
            candidate = messages[i]
            has_number_menu = (
                    "1." in candidate
                    and ("2. そのまま雑談を続ける" in candidate or "2. Web検索をする" in candidate)
                    and ("3. 終了する" in candidate or "3. これで話を終わりにする" in candidate)
            )
            if has_number_menu:
                menu_text = candidate
                menu_index = i
                break

        # 履歴が渡されない呼び出しでは、内部の会話履歴を使う。
        if not menu_text and not messages:
            turns = self.context_memory.get("turns", [])
            if turns and isinstance(turns[-1], dict):
                candidate = str(turns[-1].get("assistant", ""))
                if ("1." in candidate and
                        ("2. そのまま雑談を続ける" in candidate or "2. Web検索をする" in candidate) and
                        ("3. 終了する" in candidate or "3. これで話を終わりにする" in candidate)):
                    menu_text = candidate

        if not menu_text:
            return None

        # 話題メニューなら表示文から、好み共有メニューなら直前のユーザー発言から話題を復元。
        topic = None
        match = re.search(r"よかったら「(.+?)」について調べようか", menu_text)
        if match:
            topic = match.group(1).strip()
        if not topic:
            # 会話履歴の末尾から「チワワが好き」などの発言を探す。
            candidates = messages[:menu_index] if menu_index >= 0 else messages
            for prior in reversed(candidates):
                fact = self._detect_fact(prior)
                if fact and fact[0] in ("好きな動物", "好きな色", "好きなもの"):
                    topic = fact[1]
                    break
        if not topic:
            # contextが今回の会話全体を含まないケース向けに、保存済みターンも確認。
            for turn in reversed(self.context_memory.get("turns", [])):
                if not isinstance(turn, dict):
                    continue
                fact = self._detect_fact(str(turn.get("user", "")))
                if fact and fact[0] in ("好きな動物", "好きな色", "好きなもの"):
                    topic = fact[1]
                    break
        if not topic:
            return None

        is_preference_menu = "2. Web検索をする" in menu_text and "3. 終了する" in menu_text
        if choice == "1":
            if is_preference_menu:
                answer = self._reply_variant((
                    f"そうなんだね。君は「{topic}」のどんなところが好き？",
                    f"へえ、「{topic}」が好きなんだね！君はそれについてどう感じた？",
                    f"そうなんだね！「{topic}」のどんなところに惹かれるの？",
                ))
            else:
                answer = f"「{topic}」の話をもっとしよう！どんなところが気になっている？特徴や仕組みなど、一緒に話せるよ。"
        elif choice == "2":
            query = f"{topic}について"
            researched, confidence, sources, error = self._research_answer(query, progress_callback=progress_callback)
            if researched:
                answer = researched
                self._save_research_candidate(query, answer, confidence, sources)
            else:
                answer = (error or f"「{topic}」について検索できなかったよ。") + "\n\n少し時間をおいてもう一度試してね。"
                self._save_research_candidate(query, answer, confidence, sources)
        else:
            answer = f"了解！「{topic}」の話はここで終わりにしよう。またいつでも話しかけてね。"

        self._remember_turn(choice, answer)
        return answer

    def _teach(self, text):
        """明示的な教え方と「質問の返し方→回答」の両方を学習する。

        例:
          覚えて: 橋って知ってる？ => ああ、川にあるやつだよね
          橋って知ってるの返し方→ああ、川にあるやつだよね
        保存先は既存の ai_knowledge.json。次回は通常の質問と同じ正規化キーで照合する。
        """
        source_text = text.strip()
        match = re.match(
            r"^覚えて[：:]\s*(.+?)\s*(?:=>|＝|->|→)\s*(.+)$",
            source_text,
        )
        natural_style = False
        if not match:
            # 質問文に「？」「。」を付けても学習対象の質問キーを同じにする。
            match = re.match(
                r"^(.+?)(?:の返し方|への返し方|に対する返し方|の返答|への返答)"
                r"\s*(?:=>|＝|->|→|：|:)\s*(.+)$",
                source_text,
            )
            natural_style = bool(match)

        if not match:
            return None

        question = match.group(1).strip().rstrip(" \t\r\n。.!！?？、，")
        answer = match.group(2).strip()
        if (
                not question
                or not answer
                or len(question) > 200
                or len(answer) > 2000
                or len(self._normalize(question)) < 2
        ):
            return "その内容は保存できなかったよ。質問と返答を短くしてね。"

        key = self._normalize(question)
        now = datetime.now().isoformat(timespec="seconds")
        self.knowledge["pairs"][key] = {
            "question": question,
            "answer": answer,
            "confidence": 1.0,
            "source": "user_taught",
            "sources": [],
            "updated": now,
        }
        self.knowledge["counts"][key] = self.knowledge["counts"].get(key, 0) + 1
        self._save_knowledge()

        # 過去にこの質問を未学習として記録していた場合も、学習済みと分かる状態に更新する。
        for item in self.unlearned:
            if not isinstance(item, dict):
                continue
            item_key = str(item.get("key", ""))
            item_text = str(item.get("text", ""))
            if item_key == key or self._normalize(item_text) == key:
                item["status"] = "user_taught"
                item["learned_answer"] = answer
                item["learned_at"] = now
        self._save_unlearned()

        if natural_style:
            return f'覚えたよ！ 次に「{question}？」と聞かれたら、「{answer}」と返すね。次からこの返し方を優先するよ。'
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

    @staticmethod
    def _clean_search_text(value):
        """HTML由来の文字列を画面表示向けに整える。"""
        value = html.unescape(str(value or ""))
        value = re.sub(r"<[^>]+>", " ", value)
        value = re.sub(r"\s+", " ", value).strip()
        return value

    @staticmethod
    def _unwrap_search_url(link):
        """DuckDuckGoのリダイレクトURLを展開し、HTTP(S)だけを受け入れる。"""
        link = html.unescape(str(link or "")).strip()
        if link.startswith("//"):
            link = "https:" + link
        if link.startswith("/"):
            link = urllib.parse.urljoin("https://duckduckgo.com", link)

        try:
            parsed = urllib.parse.urlparse(link)
            if parsed.scheme not in ("http", "https"):
                return ""
            host = (parsed.hostname or "").lower()
            if host.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
                params = urllib.parse.parse_qs(parsed.query)
                destination = params.get("uddg", [""])[0]
                if destination:
                    link = urllib.parse.unquote(destination)
                    parsed = urllib.parse.urlparse(link)
            if parsed.scheme not in ("http", "https") or not parsed.hostname:
                return ""
            # トラッキング用パラメーターだけを落として同じページの重複を減らす。
            tracking_names = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "fbclid", "gclid"}
            clean_query = urllib.parse.urlencode([
                (key, val)
                for key, values in urllib.parse.parse_qs(parsed.query, keep_blank_values=True).items()
                if key.lower() not in tracking_names
                for val in values
            ])
            path = parsed.path or "/"
            return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, path, "", clean_query, ""))
        except (ValueError, TypeError):
            return ""

    @classmethod
    def _search_query_variants(cls, query):
        """会話語を除いた検索語も用意する。意味を変える過剰な書き換えはしない。"""
        original = re.sub(r"\s+", " ", str(query or "")).strip()
        if not original:
            return []
        variants = [original]
        simplified = original
        simplified = re.sub(r"^(?:ねえ|ちょっと|ところで|ちなみに|教えて|知りたいんだけど)[、,\s　]*", "", simplified)
        simplified = re.sub(r"(?:を|について)?(?:検索して|調べて|教えて|説明して|知りたい)[くださいね！!。\s　]*$", "", simplified)
        simplified = re.sub(r"[？?！!。]+$", "", simplified).strip()
        if simplified and simplified not in variants and len(simplified) >= 2:
            variants.append(simplified)
        # 「橋って知ってる？」のような認識確認は、話題の名詞自体も検索語にする。
        # 生の質問だけでは「知ってる？」の部分に引っ張られるため、主題語の検索も行う。
        known_match = re.match(
            r"^(.+?)(?:って|は)?(?:知ってる|知ってますか|聞いたことある|分かる|わかる)[？?！!。]*$",
            simplified,
            re.IGNORECASE,
        )
        if known_match:
            subject = known_match.group(1).strip().rstrip("っては？?！!。 ")
            if subject:
                # 主題語を優先候補にする。「橋」のような1文字の日本語名詞も許可。
                variants = [original]
                for variant in (subject + "とは", subject, simplified):
                    if variant and variant not in variants:
                        variants.append(variant)

        # 単語だけの入力では、同じ語の検索結果だけでなく、基本説明・特徴・用途を
        # 含む検索語も使う。これにより、名前だけ一致する記事や用語辞典に偏りにくくする。
        standalone = bool(
            re.fullmatch(r"[\wぁ-んァ-ン一-龯.+#-]{1,32}", original)
            and cls._extract_standalone_topic(original) is not None
        )
        if standalone and not re.search(
                r"(とは|について|使い方|やり方|方法|歴史|性格|特徴|仕組み|原理|由来|起源|違い|比較|費用|価格|寿命|体重|被毛|毛の種類|のは)$",
                simplified,
                re.IGNORECASE,
        ):
            for expanded in (
                    f"{simplified} とは 特徴 基本",
                    f"{simplified} 用途 できること 使い道",
            ):
                if expanded not in variants:
                    variants.append(expanded)
        # DuckDuckGo側で元の文がうまく処理されない場合の日本語語句検索。
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9._+-]*|[\u3040-\u30ff\u4e00-\u9fff]{2,}", simplified)
        if len(words) >= 3:
            compact = " ".join(words[:8])
            if compact not in variants:
                variants.append(compact)
        return variants[:3]

    def _web_search(self, query):
        """複数の検索元を順番に試し、検索HTMLの変更で結果が空になる問題を減らす。"""
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ja-JP,ja;q=0.9,en;q=0.7",
            "Cache-Control": "no-cache",
        }
        variants = self._search_query_variants(query) or [str(query or "").strip()]
        variants = [item for item in variants if item]
        if not variants:
            return []

        last_error = None

        def normalize_items(items, limit=12):
            cleaned = []
            seen = set()
            blocked_hosts = (
                "duckduckgo.com", "bing.com", "google.com", "microsoft.com",
                "facebook.com", "twitter.com", "x.com", "instagram.com",
            )
            blocked_path_parts = (
                "/privacy", "/terms", "/about", "/settings", "/login",
                "/signin", "/preferences", "/account", "/help/feedback",
            )
            blocked_titles = {
                "menu", "settings", "privacy", "terms", "sign in", "ログイン",
                "設定", "プライバシー", "利用規約", "フィードバック",
            }
            for item in items:
                title = self._clean_search_text(item.get("title", ""))
                snippet = self._clean_search_text(item.get("snippet", ""))
                link = self._unwrap_search_url(item.get("url", ""))
                if not title or len(title) < 2 or not link or link in seen:
                    continue
                try:
                    parsed_link = urllib.parse.urlparse(link)
                except ValueError:
                    continue
                host = (parsed_link.hostname or "").lower().removeprefix("www.")
                path = (parsed_link.path or "/").lower()
                if not host or any(host == blocked or host.endswith("." + blocked) for blocked in blocked_hosts):
                    continue
                if any(part in path for part in blocked_path_parts) or title.lower() in blocked_titles:
                    continue
                if parsed_link.scheme not in ("http", "https"):
                    continue
                seen.add(link)
                cleaned.append({"title": title, "url": link, "snippet": snippet})
                if len(cleaned) >= limit:
                    break
            return cleaned

        # 1) DuckDuckGo HTML / Lite。既知の結果リンクを優先し、結果ゼロのときは
        # ページ内の予備リンクも試す。片方がページ構造変更されても即失敗しない。
        ddg_results = []
        for variant_index, search_query in enumerate(variants):
            params = urllib.parse.urlencode({"q": search_query, "kl": "jp-jp", "kp": "-2"})
            endpoints = ["https://html.duckduckgo.com/html/?"]
            if variant_index == 0:
                endpoints.append("https://lite.duckduckgo.com/lite/?")
            for endpoint in endpoints:
                request = urllib.request.Request(endpoint + params, headers=headers)
                try:
                    with urllib.request.urlopen(request, timeout=7) as response:
                        raw = response.read(2_500_000).decode("utf-8", "replace")
                except Exception as error:
                    last_error = error
                    continue
                parser = _SearchParser()
                try:
                    parser.feed(raw)
                except Exception:
                    continue
                # items と fallback_items を混同せず、通常結果を優先する。
                parsed = normalize_items(parser.items)
                if not parsed:
                    parsed = normalize_items(parser.fallback_items)
                if parsed:
                    for result in parsed:
                        if result["url"] not in {item["url"] for item in ddg_results}:
                            ddg_results.append(result)
                if len(ddg_results) >= 12:
                    break
            if len(ddg_results) >= 12:
                break
        # DuckDuckGoに結果があってもここでは終了しない。
        # 1つの検索エンジンだけに依存すると、Wikipediaなど特定ドメインに偏るため、
        # Bingの結果も集めてから重複を除いて統合する。

        # 2) Bing HTMLも検索し、DuckDuckGoの結果と統合する。
        # 検索結果ブロック (li.b_algo) 内のタイトル・URL・説明だけを読むため、
        # ナビゲーションやフッターのリンクを検索結果と誤認しにくい。
        for search_query in variants[:3]:
            try:
                params = urllib.parse.urlencode({"q": search_query, "setlang": "ja", "cc": "JP"})
                request = urllib.request.Request("https://www.bing.com/search?" + params, headers=headers)
                with urllib.request.urlopen(request, timeout=7) as response:
                    raw = response.read(3_000_000).decode("utf-8", "replace")
            except Exception as error:
                last_error = error
                continue

            bing_items = []
            blocks = re.findall(
                r'<li\b(?=[^>]*class=["\'][^"\']*\bb_algo\b[^"\']*["\'])[^>]*>(.*?)(?=<li\b[^>]*class=["\'][^"\']*\bb_algo\b|</ol>)',
                raw,
                flags=re.IGNORECASE | re.DOTALL,
            )
            for block in blocks:
                anchor = re.search(
                    r'<h2\b[^>]*>.*?<a\b[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
                    block,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                if not anchor:
                    # HTMLの構造差に備え、b_algoブロック内で最初の外部リンクを使う。
                    anchor = re.search(
                        r'<a\b[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
                        block,
                        flags=re.IGNORECASE | re.DOTALL,
                    )
                if not anchor:
                    continue
                snippet_match = re.search(
                    r'<p\b[^>]*>(.*?)</p>', block,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                bing_items.append({
                    "url": html.unescape(anchor.group(1)),
                    "title": self._clean_search_text(anchor.group(2)),
                    "snippet": self._clean_search_text(snippet_match.group(1) if snippet_match else ""),
                })
            bing_results = normalize_items(bing_items)
            if bing_results:
                known_urls = {item["url"] for item in ddg_results}
                for result in bing_results:
                    if result["url"] not in known_urls:
                        ddg_results.append(result)
                        known_urls.add(result["url"])
                # 十分な候補が集まっても、他の検索語の結果も少し取り込めるよう
                # 24件までは収集を続ける。
                if len(ddg_results) >= 24:
                    break

        # 2.5) Bing RSS検索を予備として試す。
        # HTML検索ページが403/429などで拒否されても、RSS形式が利用できる場合がある。
        # RSSはHTMLのクラス名変更に影響されにくいため、XMLの title/link/description を読む。
        bing_rss_results = []
        for search_query in variants[:3]:
            try:
                params = urllib.parse.urlencode({"q": search_query, "format": "rss", "setlang": "ja", "cc": "JP"})
                request = urllib.request.Request(
                    "https://www.bing.com/search?" + params,
                    headers={**headers, "Accept": "application/rss+xml,application/xml,text/xml,*/*;q=0.8"},
                    )
                with urllib.request.urlopen(request, timeout=8) as response:
                    raw = response.read(2_500_000).decode("utf-8", "replace")
                root = ET.fromstring(raw)
            except Exception as error:
                last_error = error
                continue

            rss_items = []
            for entry in root.iter():
                if entry.tag.rsplit("}", 1)[-1].lower() != "item":
                    continue
                fields = {}
                for child in list(entry):
                    fields[child.tag.rsplit("}", 1)[-1].lower()] = (child.text or "").strip()
                if fields.get("link") and fields.get("title"):
                    rss_items.append({
                        "url": fields.get("link", ""),
                        "title": fields.get("title", ""),
                        "snippet": fields.get("description", ""),
                    })
            for item in normalize_items(rss_items):
                if item["url"] not in {existing["url"] for existing in bing_rss_results}:
                    bing_rss_results.append(item)

        # 通常Web検索から得られた候補のうち、Wikipedia以外を優先する。
        # Wikipediaだけが返った場合は、そのまま回答に流さず別の検索元も試す。
        non_wikipedia = [
            item for item in ddg_results
            if not (urllib.parse.urlparse(item.get("url", "")).hostname or "").lower().endswith("wikipedia.org")
        ]
        combined_search_results = []
        seen_search_urls = set()
        for item in non_wikipedia + bing_rss_results:
            if item["url"] not in seen_search_urls:
                combined_search_results.append(item)
                seen_search_urls.add(item["url"])
        if len(combined_search_results) >= 5:
            return combined_search_results[:24]

        # 3) Yahoo検索も予備として試す。DuckDuckGo/BingのHTML構造が変わった場合に備え、
        # 検索結果ブロック内の見出しリンクだけを拾い、Wikipedia以外の候補を追加する。
        yahoo_results = []
        for search_query in variants[:3]:
            try:
                params = urllib.parse.urlencode({"p": search_query, "ei": "UTF-8", "fr": "uh3_search"})
                request = urllib.request.Request("https://search.yahoo.com/search?" + params, headers=headers)
                with urllib.request.urlopen(request, timeout=7) as response:
                    raw = response.read(3_000_000).decode("utf-8", "replace")
            except Exception as error:
                last_error = error
                continue

            yahoo_items = []
            # Yahooの結果カードは class に algo を含む div として配信されることが多い。
            blocks = re.findall(
                r'<div\b(?=[^>]*class=["\'][^"\']*\balgo\b[^"\']*["\'])[^>]*>(.*?)(?=<div\b[^>]*class=["\'][^"\']*\balgo\b[^"\']*["\']|</ol>|</main>)',
                raw,
                flags=re.IGNORECASE | re.DOTALL,
            )
            for block in blocks:
                anchor = re.search(
                    r'<h3\b[^>]*>.*?<a\b[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
                    block,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                if not anchor:
                    anchor = re.search(
                        r'<a\b[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
                        block,
                        flags=re.IGNORECASE | re.DOTALL,
                    )
                if not anchor:
                    continue
                snippet_match = re.search(
                    r'<(?:p|div)\b[^>]*class=["\'][^"\']*(?:compText|snippet|description|algo\s*\w*)[^"\']*["\'][^>]*>(.*?)</(?:p|div)>',
                    block,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                yahoo_items.append({
                    "url": html.unescape(anchor.group(1)),
                    "title": self._clean_search_text(anchor.group(2)),
                    "snippet": self._clean_search_text(snippet_match.group(1) if snippet_match else ""),
                })
            parsed_yahoo = normalize_items(yahoo_items)
            for item in parsed_yahoo:
                host = (urllib.parse.urlparse(item.get("url", "")).hostname or "").lower()
                if host.endswith("wikipedia.org"):
                    continue
                if item["url"] not in {existing["url"] for existing in yahoo_results}:
                    yahoo_results.append(item)

        combined_non_wikipedia = []
        seen_urls = set()
        for item in combined_search_results + yahoo_results:
            if item["url"] not in seen_urls:
                combined_non_wikipedia.append(item)
                seen_urls.add(item["url"])
        if combined_non_wikipedia:
            return combined_non_wikipedia[:24]

        # Wikipedia APIは、ユーザーがWikipediaを明示的に指定した場合に限って使う。
        # 一般的なWeb検索に失敗しただけでWikipediaの記事を検索結果として見せない。
        if re.search(r"(wikipedia|ウィキペディア|wikiで)", str(query or ""), re.IGNORECASE):
            api_headers = dict(headers)
            api_headers["Accept"] = "application/json,text/plain,*/*"
            for api_host in ("ja.wikipedia.org", "en.wikipedia.org"):
                for search_query in variants[:2]:
                    params = urllib.parse.urlencode({
                        "action": "query", "list": "search", "srsearch": search_query,
                        "format": "json", "utf8": "1", "srlimit": "8", "srprop": "snippet",
                    })
                    request = urllib.request.Request(
                        f"https://{api_host}/w/api.php?{params}", headers=api_headers
                    )
                    try:
                        with urllib.request.urlopen(request, timeout=7) as response:
                            payload = json.loads(response.read(1_500_000).decode("utf-8", "replace"))
                    except Exception as error:
                        last_error = error
                        continue
                    search_items = payload.get("query", {}).get("search", []) if isinstance(payload, dict) else []
                    wiki_items = []
                    for entry in search_items:
                        title = self._clean_search_text(entry.get("title", ""))
                        if not title:
                            continue
                        page_path = urllib.parse.quote(title.replace(" ", "_"), safe="()_,'-")
                        wiki_items.append({
                            "title": title,
                            "url": f"https://{api_host}/wiki/{page_path}",
                            "snippet": self._clean_search_text(entry.get("snippet", "")),
                        })
                    wiki_results = normalize_items(wiki_items)
                    if wiki_results:
                        return wiki_results

        # エラーと「ページは取得できたが検索結果を解析できなかった」を区別する。
        if last_error and not ddg_results:
            raise last_error
        return []

    @classmethod
    def _build_focused_answer(cls, topic, focus_label, focus_pattern, sources, fetched_paragraphs=None):
        """特定の項目を尋ねる質問に対し、焦点に直接関係する文章だけを抽出する。"""
        fetched_paragraphs = fetched_paragraphs or {}
        focus_re = re.compile(focus_pattern, re.IGNORECASE)
        # 見出し・目次・章タイトルが回答候補に混ざるのを防ぐ。
        # _build_topic_overview() 内だけで定義されていたため、こちらでも定義する。
        title_fragment_pattern = re.compile(
            r"^(?:\d+(?:\.\d+)*\s*)?(?:Pythonの)?(?:インストール方法|使い方|"
            r"実行環境|基本操作|環境構築)(?:｜|\||:|：|\s).{0,120}$|"
            r"^(?:\d+(?:\.\d+)*\s*)?Pythonのインストールと実行環境.*インストール方法$",
            re.IGNORECASE,
        )
        if focus_label == "性格":
            # 性格そのものの語がなくても、警戒心・勇敢さ・吠えやすさなどの
            # 性格特性を述べている文を抽出対象に含める。
            focus_re = re.compile(
                r"(性格|気質|警戒心|機敏|注意深|活発|勇敢|忠実|愛情深|甘えん坊|"
                r"人懐こ|社交的|慎重|好奇心|賢い|独立心|自信|臆病|おとなしい|"
                r"友好的|用心深|吠えやす|繊細|性格差|飼い主に.{0,15}(?:忠実|愛情|懐))",
                re.IGNORECASE,
            )
        elif focus_label == "使い方":
            # 「使い方」という語が本文にない実用的な説明も候補に含める。
            # 例: 「コードを入力して実行する」「print()で文字を表示する」。
            focus_re = re.compile(
                r"(使い方|利用方法|操作|手順|方法|コード|コマンド|プログラム|"
                r"入力|実行|起動|クリック|選択|保存|インストール|環境構築|"
                r"print\s*\(|input\s*\(|import\s+\w+|def\s+\w+|"
                r"\.py|Shift|Enter|セル|変数|条件分岐|if文|for文|繰り返し|関数|"
                r"計算|文字を表示|画面に表示|書きます|記述します|入力します)",
                re.IGNORECASE,
            )
        meta_re = re.compile(
            r"(この章では|この講座では|本記事では|この記事では|このページでは|"
            r"このニュースは|詳しく見ていきましょう|エンジニアになるには|"
            r"必要ですか|最大の違いは何ですか|クリックして|スポンサーリンク|"
            r"ブリーダー専門サイト|ご紹介します|ご紹介しています|ロゴは|CMソング|DJ\s*BOBO|"
            r"今回は.{0,80}(?:解説|紹介|見ていき)|本稿では|次のように解説|"
            r"以下のとおり紹介|この記事を読むと|詳しく解説します)",
            re.IGNORECASE,
        )
        # 被毛について聞かれた場合、毛の説明に直接関係しない人気・好み・性格論は
        # 被毛の特徴と同じ語（スムースコート等）を含んでいても混ぜない。
        coat_off_topic_re = re.compile(
            r"(人気(?:が高|を集|の犬種|ランキング)|飼育されているチワワは.{0,25}ほとんど|"
            r"チワワ本来の性格|性格が濃く表れ|同じ気質|気質を持つ|"
            r"写真映え|可愛らしさ|可愛らしい|かわいらしさ|"
            r"希望する方|魅力を感じる|見た目の.{0,20}魅力|好みで選|"
            r"今回は.{0,50}(?:解説|紹介)|この章では|詳しく見ていきましょう)",
            re.IGNORECASE,
        )
        # 性格の質問では、犬種の性格を直接説明する事実を優先し、
        # 記事の導入・読者への問いかけ・人気/見た目の宣伝文を除外する。
        personality_editorial_re = re.compile(
            r"(見た目の可愛さから|見た目のかわいさから|初めて犬を飼う人に選ばれる|"
            r"小さな体に.{0,35}(?:芯の強さ|繊細な性格)|驚くほど繊細|"
            r"チワワと暮らすなかで|うちの子だけの性質|こう感じる場面|"
            r"性格の犬種なのか|今回は.{0,60}(?:解説|紹介)|"
            r"詳しく見ていきましょう|この章では|このページでは|本記事では|"
            r"魅力を秘めています|魅力的な犬種|人気を誇ります|"
            r"しつけの難易度は意外と高め|選ばれることも多いチワワ|"
            r"可愛いだけの犬種|かわいいだけの犬種|自己主張が強く.{0,30}番犬気質|"
            r"気の強さを秘めた番犬気質|番犬気質の犬)",
            re.IGNORECASE,
        )
        personality_trait_re = re.compile(
            r"(機敏|注意深|活発|勇敢|警戒心|忠実|愛情深|甘えん坊|人懐こ|"
            r"社交的|慎重|好奇心|賢い|独立心|自信|臆病|おとなしい|"
            r"友好的|用心深|吠えやす|飼い主に.{0,15}(?:忠実|愛情|懐)|"
            r"性格差|気質)",
            re.IGNORECASE,
        )
        # 同じ公式説明を「要点: ...」の形で言い換えた重複文は、
        # 一般的な説明とまとめ文の両方が並ばないようにする。
        personality_redundant_summary_re = re.compile(
            r"(?:要点\s*[:：].{0,40}(?:JKC|犬種紹介).{0,140}(?:機敏|注意深).{0,80}(?:活発|勇敢)|"
            r"(?:これは|あくまで)犬種全体の傾向であり、?断定ではありません)",
            re.IGNORECASE,
        )
        personality_coat_tangent_re = re.compile(
            r"(ロングコート|スムースコート|毛の長さ|被毛|飾り毛|ブラッシング|"
            r"毛の種類).{0,45}(?:チワワ|性格)|(?:チワワ|性格).{0,45}"
            r"(ロングコート|スムースコート|毛の長さ|被毛|飾り毛|ブラッシング|毛の種類)",
            re.IGNORECASE,
        )
        # 歴史・由来の質問では、実際の歴史情報と記事の感想・宣伝・締め文を分ける。
        history_editorial_re = re.compile(
            r"(今日ではとても人気|長く深い歴史|知られざる歴史|ルーツは謎に包まれ|"
            r"歴史があることが分かりました|"
            r"今回は.{0,60}(?:歴史|ルーツ|由来).{0,30}(?:紹介|解説)|"
            r"興味深い歴史|魅力的な歴史|その歴史をご紹介|ぜひご覧ください|"
            r"信ぴょう性が高い|信憑性が高い|確かな情報といえます|間違いありません)",
            re.IGNORECASE,
        )
        history_wrapup_re = re.compile(
            r"(?:長く深い歴史|知られざる歴史|歴史があることが分かりました|"
            r"ルーツは謎に包まれています)",
            re.IGNORECASE,
        )
        # 性格質問の回答に、身体的・健康管理の話題が混ざるのを防ぐ。
        # （例：体温を保ちにくい／緊張で震える）は別の質問として扱うべき。
        personality_health_tangent_re = re.compile(
            r"(体温を保ちにくい|体温調節|緊張で震える|震えることも|"
            r"普通.{0,12}異常.{0,12}判断|健康面|健康管理|骨や関節|"
            r"寒さへの対策|寒さ対策|散歩不要|運動量|長距離の散歩)",
            re.IGNORECASE,
        )
        # JKCの性格定義を言い換えただけの文が、別の箇条書きとして重複しないようにする。
        personality_trait_paraphrase_re = re.compile(
            # 公式の一文を残し、それを「性格は機敏で勇敢とされる一方…」と
            # 言い換えただけの文だけを除外する。
            r"(?:性格は|性格について).{0,35}(?:機敏で勇敢|「機敏で勇敢」).{0,55}(?:一方|反面|繊細さもあわせ持つ)|"
            r"(?:機敏で勇敢|「機敏で勇敢」).{0,35}(?:一方|反面).{0,35}(?:繊細|あわせ持つ)",
            re.IGNORECASE,
        )
        candidates = []
        seen = set()
        for source_index, source in enumerate(sources[:5]):
            url = str(source.get("url", ""))
            title = cls._clean_search_text(source.get("title", ""))
            if meta_re.search(title):
                continue
            # 取得本文だけでなく検索結果のスニペットも補助候補に加える。
            # ページ本文が取得できても、抽出された段落に必要な操作説明がない場合がある。
            blocks = list(fetched_paragraphs.get(url, []) or [])
            snippet = str(source.get("snippet", "") or "").strip()
            if snippet and snippet not in blocks:
                blocks.append(snippet)
            if not blocks:
                blocks = [""]
            for block in blocks:
                cleaned = cls._clean_search_text(str(block or ""))
                if not cleaned:
                    continue
                parts = re.split(r"(?<=[。！？])\s*", cleaned)
                for part in parts:
                    part = re.sub(r"^[-・•●○\s]+", "", part).strip()
                    part = re.sub(r"\[\s*\d+(?:\s*[,;、]\s*\d+)*\s*\]", "", part)
                    part = re.sub(r"\s{2,}", " ", part).strip()
                    if not cls._is_usable_japanese_candidate(part):
                        continue
                    if len(part) < 18 or part.endswith(("?", "？")) or meta_re.search(part):
                        continue
                    # 見出し・記事タイトル断片は、説明文として再掲しない。
                    if re.search(r"(?:チワワの種類は(?:2|二)種類だけ|違いや特徴.{0,30}比較|今回は.{0,80}(?:解説|紹介))", part, re.IGNORECASE):
                        continue
                    if not focus_re.search(part):
                        continue
                    if focus_label == "使い方":
                        # 「使い方」質問では目次・講座案内・サービス紹介を避け、
                        # 実際の手順、操作、コード例につながる文章を優先する。
                        howto_noise_re = re.compile(
                            r"(目次|目次一覧|この記事では|本記事では|このページでは|"
                            r"この講座では|講座|Udemy|侍エンジニア|スクール|受講|"
                            r"おすすめの.*(?:教材|講座|サイト)|学習ロードマップ|"
                            r"章立て|カリキュラム|詳しく見ていきましょう|"
                            r"ご紹介します|ご紹介しています|無料相談|資料請求|"
                            r"エンジニアになるには|学習を始めましょう)",
                            re.IGNORECASE,
                        )
                        howto_outline_re = re.compile(
                            r"^(?:第?\d+[.．、:：)）\s]*)?(?:Python\s*)?"
                            r"(?:入門|基礎|基本|概要|目次|環境構築|インストール|"
                            r"変数|演算子|リスト|辞書|関数|クラス|応用|まとめ)"
                            r"(?:とは|の使い方|の基本|編|章)?[。.!！]?$",
                            re.IGNORECASE,
                        )
                        if howto_noise_re.search(part) or howto_outline_re.match(part):
                            continue
                        # 実行可能な例や具体的な操作説明がない宣伝的・抽象的な文を減点する。
                        if re.search(r"(?:```|print\s*\(|import\s+\w+|def\s+\w+|pip\s+install|"
                                     r"クリック|選択し|入力し|保存し|実行し|インストールし|"
                                     r"次のコード|以下のコード|例えば)", part, re.IGNORECASE):
                            pass
                        elif len(part) < 35 or re.search(r"(学べます|学習できます|便利です|人気があります)$", part):
                            continue
                    if focus_label == "被毛・毛の種類" and coat_off_topic_re.search(part):
                        continue
                    if focus_label == "原産地・歴史":
                        # 「信ぴょう性が高い」など記事側の評価を落とし、確認可能な事実部分を残す。
                        part = re.sub(
                            r"登録されており[、，,]?\s*(?:信ぴょう性|信憑性)が高い(?:といえます|です|と考えられます)?[。！!]?\s*$",
                            "登録されています。",
                            part,
                            flags=re.IGNORECASE,
                        ).strip()
                        part = re.sub(
                            r"[、，,。]?\s*(?:そのため、?\s*)?(?:信ぴょう性|信憑性)が高い(?:といえます|です|と考えられます)?[。！!]?\s*$",
                            "。" if re.search(r"[。]", part) else "",
                            part,
                            flags=re.IGNORECASE,
                        ).strip()
                        # 歴史の説明に関係するように見えても、記事の感想・結びや
                        # 根拠を添えていない確信表現は要点として採用しない。
                        if history_editorial_re.search(part) or history_wrapup_re.search(part):
                            continue
                        # 年号と頭数を含む具体的な起源説は、一般ブログ同士の転載・反復だけでは
                        # 裏付け済みと扱わない。犬種の歴史では、信頼できる犬種登録団体・公式団体
                        # などの独立した出典で同じ主張を確認できない限り、断定文から除外する。
                        if (re.search(r"(?:18|19|20)\d{2}年", part)
                                and re.search(r"\d+頭", part)
                                and re.search(r"(?:発見|見つか|歴史は始|歴史が始)", part)):
                            year_match = re.search(r"((?:18|19|20)\d{2})年", part)
                            count_match = re.search(r"(\d+)頭", part)
                            trusted_history_domains = (
                                "akc.org", "jkc.or.jp", "fci.be", "thekennelclub.org.uk",
                                "ckc.ca", "ankc.org.au", "royalkennelclub.com",
                                "britannica.com", "nationalgeographic.com",
                            )
                            def _trusted_history_host(candidate_url):
                                host = (urllib.parse.urlparse(candidate_url).hostname or "").lower().removeprefix("www.")
                                return any(host == domain or host.endswith("." + domain) for domain in trusted_history_domains)

                            current_source_trusted = _trusted_history_host(url)
                            independently_confirmed_by_trusted_source = False
                            for other_url, other_blocks in fetched_paragraphs.items():
                                if other_url == url or not _trusted_history_host(other_url):
                                    continue
                                joined_other = " ".join(
                                    cls._clean_search_text(str(v or ""))
                                    for v in (other_blocks or [])
                                )
                                if (year_match and year_match.group(1) + "年" in joined_other
                                        and count_match and count_match.group(1) + "頭" in joined_other
                                        and re.search(r"(?:発見|見つか|チワワ州)", joined_other)):
                                    independently_confirmed_by_trusted_source = True
                                    break
                            if not (current_source_trusted and independently_confirmed_by_trusted_source):
                                continue
                    if focus_label == "性格":
                        # 公式説明をほぼそのまま言い直した「要点」文や、同じ特徴を再要約した文を除外。
                        if personality_redundant_summary_re.search(part):
                            continue
                        if personality_trait_paraphrase_re.search(part):
                            continue
                        # 記事の導入・煽り・読者への問いかけは、性格の説明として採用しない。
                        if personality_editorial_re.search(part):
                            continue
                        # 性格の回答に身体・健康の説明が混ざると焦点がずれるので除外する。
                        if personality_health_tangent_re.search(part):
                            continue
                        # 「被毛による性格差」の関係だけは性格質問に役立つため、
                        # 長い混合文を短い要点に直して残す。
                        coat_personality_relation = bool(re.search(
                            r"(?:毛の種類|被毛|被毛タイプ|コートの種類).{0,24}性格差.{0,18}(?:大きくありません|大きくない|ほとんどありません|ありません|見られません)",
                            part,
                        ))
                        if personality_coat_tangent_re.search(part):
                            if not coat_personality_relation:
                                continue
                            part = "被毛タイプ（ロングコート・スムースコート）による性格差は大きくないとする説明もある。"
                        # 性格という語を含むだけの文章ではなく、具体的な性格特性がある文を残す。
                        if not personality_trait_re.search(part):
                            continue
                    # 文脈が必要な接続句・未完の断片を避ける。
                    if re.match(r"^(?:そのため|また|さらに|この章では|このように|しかし|その結果)[、，]?", part):
                        continue
                    if part.endswith(("、", ",", "：", ":", "などの", "など")):
                        continue
                    key = re.sub(r"\s+", "", part).casefold()
                    if key in seen:
                        continue
                    seen.add(key)
                    score = 0.25 * bool(topic and topic in part) + 0.25 * bool(re.search(r"(とは|は|には|がある|に分かれる|タイプ|種類|特徴|必要|注意)", part)) + min(len(part), 160) / 1000.0
                    if focus_label == "使い方":
                        # 手順・コード例を優先し、単なる説明見出しは選ばない。
                        if re.search(r"(?:```|print\s*\(|import\s+\w+|def\s+\w+|pip\s+install|"
                                     r"クリック|選択し|入力し|保存し|実行し|インストールし|"
                                     r"次のコード|以下のコード|例えば)", part, re.IGNORECASE):
                            score += 1.5
                        if re.search(r"(目次|講座|教材|スクール|学習ロードマップ)", part, re.IGNORECASE):
                            score -= 2.0
                    if focus_label == "性格":
                        # JKCなどの直接的な性格定義や、具体的な特性を含む説明を優先。
                        if re.search(r"(JKC|犬種紹介|犬種標準|機敏で注意深|活発であり|大変勇敢)", part, re.IGNORECASE):
                            score += 0.75
                        if personality_trait_re.search(part):
                            score += 0.35
                        if re.search(r"(傾向があります|とされます|と説明|性格は|気質は)", part):
                            score += 0.15
                        # まとめの前置きや断定回避だけが加わった重複文は優先度を下げる。
                        if re.search(r"(要点\s*[:：]|これは犬種全体の傾向|断定ではありません)", part):
                            score -= 0.30
                    if focus_label == "被毛・毛の種類":
                        # 最初にタイプと決定的な違いを説明し、その後に具体的特徴・手入れを置く。
                        if re.search(r"スムースコート", part) and re.search(r"ロングコート", part) and re.search(r"(?:2|二)種類|2タイプ|主な違いは毛の長さ|被毛タイプ|毛の長さ", part):
                            score += 0.75
                        if re.search(r"ロングコート", part) and re.search(r"(耳|首元|尾|しっぽ).{0,12}飾り毛|飾り毛.{0,12}(耳|首元|尾|しっぽ)", part):
                            score += 0.45
                        if re.search(r"スムースコート", part) and re.search(r"(毛が短|被毛が短|短い被毛|短毛|滑らか)", part):
                            score += 0.35
                        if re.search(r"(ブラッシング|お手入れ|毛玉|寒さ|防寒|被毛のケア)", part):
                            score += 0.18
                    candidates.append((score, source_index, part))

        # 目次の列挙や講座への誘導を、質問への説明文として返さない。
        toc_like_pattern = re.compile(
            r"(?:^|[・\s])(?:\d+(?:[-.]\d+)*\s*[^。]{0,45})(?:\s*[・|｜]\s*\d+(?:[-.]\d+)*){2,}"
            r"|(?:目次|この記事でわかること|この記事の内容|学習ロードマップ|カリキュラム)",
            re.IGNORECASE,
        )
        promotional_answer_pattern = re.compile(
            r"(?:参考になります|こちらも参考|受講|講座|スクール|無料相談|資料請求|"
            r"お問い合わせ|サービスをご紹介|おすすめの.*(?:講座|スクール)|"
            r"学習ガイド|入門解説も|ご覧ください|登録はこちら)"
        )
        practical_focus = focus_label in {"使い方", "構造・仕組み", "原因・理由"}
        filtered_candidates = []
        for score, source_index, part in candidates:
            compact = re.sub(r"\s+", " ", part).strip()
            if title_fragment_pattern.search(compact):
                continue
            if promotional_answer_pattern.search(compact) or toc_like_pattern.search(compact):
                continue
            if practical_focus:
                has_actionable_detail = bool(re.search(
                    r"(?:```|`[^`]+`|\bprint\s*\(|\binput\s*\(|コード|コマンド|クリック|"
                    r"選択|保存|実行|インストール|入力すると|書くと|作成し|設定し|手順|\.py)",
                    compact, re.IGNORECASE
                ))
                if has_actionable_detail:
                    score += 1.25
                elif len(compact) < 45:
                    score -= 0.8
            filtered_candidates.append((score, source_index, part))
        candidates = filtered_candidates
        candidates.sort(key=lambda item: item[0], reverse=True)
        chosen = []
        chosen_keys = set()
        chosen_source_indices = []
        chosen_token_sets = []
        coat_type_summary_already_chosen = False
        long_coat_feathering_already_chosen = False
        for score, source_index, part in candidates:
            key = re.sub(r"\s+", "", part).casefold()
            if key in chosen_keys:
                continue
            # 似た文の重複をゆるく除く。
            normalized = re.sub(r"[、。！？!?,：:\s]", "", part).casefold()
            if any(normalized in old or old in normalized for old in chosen_keys if min(len(normalized), len(old)) > 20):
                continue

            # 日本語の言い換えによる重複を、完全一致だけでなくトークンの重なりでも除く。
            # 特に「JKCの説明」と「要点: JKCの説明では…」が二重に出る問題に対応。
            current_tokens = cls._tokens(part)
            if focus_label == "性格" and len(current_tokens) >= 8:
                duplicate_semantics = False
                for old_tokens in chosen_token_sets:
                    smaller = min(len(current_tokens), len(old_tokens))
                    if smaller < 8:
                        continue
                    overlap_ratio = len(current_tokens & old_tokens) / smaller
                    if overlap_ratio >= 0.62:
                        duplicate_semantics = True
                        break
                if duplicate_semantics:
                    continue

            if focus_label == "被毛・毛の種類":
                mentions_smooth = bool(re.search(r"スムースコート", part))
                mentions_long = bool(re.search(r"ロングコート", part))
                is_two_type_summary = (
                        mentions_smooth and mentions_long
                        and bool(re.search(r"(?:2|二)種類|2タイプ|二つのタイプ|主な違いは毛の長さ|被毛タイプ", part))
                )
                # 「被毛は2種類」「タイプは2つ」「違いは毛の長さ」は同じ要点。
                # 最も説明的な1文だけ残して、同義の説明を繰り返さない。
                if is_two_type_summary and coat_type_summary_already_chosen:
                    continue
                if is_two_type_summary:
                    coat_type_summary_already_chosen = True

                # 「耳・首元・尾に飾り毛がある」という内容の言い換えは1つだけ残す。
                is_feathering_detail = bool(
                    mentions_long
                    and re.search(r"飾り毛", part)
                    and re.search(r"(耳|首元|尾|しっぽ)", part)
                )
                if is_feathering_detail and long_coat_feathering_already_chosen:
                    continue
                if is_feathering_detail:
                    long_coat_feathering_already_chosen = True

            if len(part) > 480:
                part = part[:477].rstrip(" 、，。") + "…"
            chosen.append(part)
            chosen_keys.add(key)
            chosen_source_indices.append(source_index)
            chosen_token_sets.append(cls._tokens(part))
            if len(chosen) >= 9:
                break

        if not chosen:
            return None
        if practical_focus and not any(re.search(
                r"(?:コード|コマンド|プログラム|クリック|選択|保存|実行|インストール|入力|"
                r"書くと|作成し|設定し|手順|\.py|\bprint\s*\(|\binput\s*\(|"
                r"\bimport\s+\w+|\bdef\s+\w+|変数|条件分岐|if文|for文|関数|"
                r"Shift|Enter|セル|計算|文字を表示|画面に表示)",
                sentence, re.IGNORECASE
        ) for sentence in chosen):
            return None
        # 「使い方」の質問は、関連文をただ並べず、情報の内容ごとに整理する。
        # 各説明は検索で取得した文章から選び、検索元にない手順や事実を作り足さない。
        if focus_label == "使い方":
            setup_re = re.compile(
                r"(インストール|環境構築|実行環境|準備|ダウンロード|Colab|Jupyter|"
                r"Visual\s*Studio\s*Code|VSCode|エディタ|開発環境)", re.IGNORECASE
            )
            practice_re = re.compile(
                r"(コード|コマンド|入力|記述|書き|実行|起動|クリック|選択|保存|"
                r"\.py|print\s*\(|import\s+\w+|Shift|Enter|操作)", re.IGNORECASE
            )
            next_step_re = re.compile(
                r"(変数|条件分岐|if文|for文|繰り返し|関数|リスト|辞書|ライブラリ|"
                r"文法|基本構文|データ型|演算|学習目標|学習方法)", re.IGNORECASE
            )
            sections = [
                ("始める前に", []),
                ("基本的な操作", []),
                ("次に覚えたいこと", []),
            ]
            for sentence in chosen:
                if setup_re.search(sentence):
                    bucket = sections[0][1]
                elif next_step_re.search(sentence):
                    bucket = sections[2][1]
                elif practice_re.search(sentence):
                    bucket = sections[1][1]
                else:
                    bucket = sections[0][1] if not sections[0][1] else sections[1][1]
                if sentence not in bucket:
                    bucket.append(sentence)

            lines = [
                f"「{topic}」の使い方について、取得できた本文から関係する説明を整理するね。",
                "",
            ]
            section_number = 1
            for heading, sentences in sections:
                if not sentences:
                    continue
                if section_number > 1:
                    lines.extend(["---", ""])
                lines.append(f"**{section_number}. {heading}**")
                lines.extend(f"・{sentence}" for sentence in sentences)
                lines.append("")
                section_number += 1

            # 回答末尾に「参考にしたページ」の一覧は表示しない。
            lines.append(
                "※取得できた本文をもとに要点を整理した要約だよ。サイトによって手順や対応環境が異なる場合があるので、実際に使う環境も確認してね。"
            )
            return "\n".join(lines).strip()

        lines = [f"「{topic}」の{focus_label}について、取得できた本文から関係する説明をまとめるね。", "", f"**{focus_label}**"]
        lines.extend(f"・{sentence}" for sentence in chosen)
        lines.append("※取得できた本文から関係する部分を整理した要約だよ。")
        return "\n".join(lines)

    def _research_answer(
            self,
            query,
            detailed_topic=False,
            topic_label=None,
            focus_label=None,
            focus_pattern=None,
            progress_callback=None,
    ):
        """検索結果を質問との関連性で並べて返す。

        detailed_topic=True の場合はページ本文も取得し、単語入力向けの説明を構成する。
        progress_callback が渡された場合、検索開始と取得したサイト数をUIへ通知する。
        """
        if callable(progress_callback):
            try:
                progress_callback("Web検索中...")
            except Exception:
                pass
        try:
            results = self._web_search(query)
        except Exception as error:
            detail = ""
            if isinstance(error, urllib.error.HTTPError):
                detail = f" HTTP {error.code}"
                if error.reason:
                    detail += f" ({error.reason})"
            elif isinstance(error, urllib.error.URLError):
                detail = f" ({error.reason})"
            return (
                None,
                0.0,
                [],
                "Web検索に接続できなかったよ"
                f"（{type(error).__name__}{detail}）。検索サービス側のアクセス制限や一時的な障害の可能性もあるよ。少し待って再試行してね。",
            )

        if not results:
            return (
                None,
                0.0,
                [],
                "検索結果を取得できなかったよ。キーワードを短くしたり、別の言い方で試したりしてね。",
            )

        # 検索結果を取得したら、最初のサイト名と残りの件数をUIへ表示する。
        # このあと本文取得・関連度評価・要約を行うため、検索中ステータスとして使える。
        if callable(progress_callback):
            try:
                first_host = ""
                for result in results:
                    host = (urllib.parse.urlparse(str(result.get("url", ""))).hostname or "")
                    host = host.lower().removeprefix("www.")
                    if host:
                        first_host = host
                        break
                if first_host:
                    other_count = max(0, len(results) - 1)
                    status = (
                        f"{first_host} 他{other_count}件を検索中..."
                        if other_count
                        else f"{first_host} を検索中..."
                    )
                else:
                    status = f"検索結果{len(results)}件を確認中..."
                progress_callback(status)
            except Exception:
                pass

        query_tokens = self._tokens(query)
        stop_tokens = {
            "これ", "それ", "こと", "もの", "について", "教えて", "調べて", "検索して",
            "ください", "知りたい", "何ですか", "どういう", "どうして", "なぜ", "ですか",
            "方法", "最新", "情報", "とは", "って", "です", "ます", "する", "したい",
        }
        query_tokens = {token for token in query_tokens if token not in stop_tokens and len(token) > 1}
        normalized_query = self._normalize(query)
        ranked = []
        for result in results:
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            title_tokens = self._tokens(title)
            snippet_tokens = self._tokens(snippet)
            title_overlap = len(query_tokens & title_tokens) / max(1, len(query_tokens))
            snippet_overlap = len(query_tokens & snippet_tokens) / max(1, len(query_tokens))
            combined = title + " " + snippet
            normalized_combined = self._normalize(combined)
            phrase_bonus = 0.18 if normalized_query and normalized_query in normalized_combined else 0.0
            title_bonus = 0.08 if title and any(t in title_tokens for t in query_tokens) else 0.0
            snippet_bonus = 0.04 if snippet else 0.0
            relevance = min(1.0, title_overlap * 0.62 + snippet_overlap * 0.38 + phrase_bonus + title_bonus + snippet_bonus)
            result["relevance"] = round(relevance, 3)
            result["_domain"] = (urllib.parse.urlparse(result.get("url", "")).hostname or "").lower().removeprefix("www.")
            ranked.append(result)

        if detailed_topic:
            # 単語検索では、対象の解説ページと販売・サービス紹介ページを区別する。
            # 例えば「チワワ」単独検索では、ブリーダー専門サイトのホームページより
            # 犬種の定義・特徴・飼い方を説明する記事を優先する。
            purchase_intent = bool(re.search(
                r"(購入|買いたい|販売|ブリーダー|子犬を迎え|お迎え|値段|費用|料金|里親)",
                query,
            ))
            educational_title_terms = re.compile(
                r"(とは|どんな|特徴|性格|飼い方|歴史|犬種|図鑑|基礎知識|基本情報|健康|種類|仕組み|定義|解説|ガイド)",
                re.IGNORECASE,
            )
            promotional_terms = re.compile(
                r"(ブリーダー専門サイト|ブリーダーさん|子犬なら|子犬販売|ペットショップ|"
                r"販売中|購入するメリット|お迎え|通販|オンラインショップ|ロゴ|当サイト|"
                r"このサイト|ご紹介します|ご紹介しています|お問い合わせ|無料相談|資料請求)",
                re.IGNORECASE,
            )
            for item in ranked:
                title = str(item.get("title", ""))
                snippet = str(item.get("snippet", ""))
                url = str(item.get("url", ""))
                host = (urllib.parse.urlparse(url).hostname or "").lower().removeprefix("www.")
                path = urllib.parse.urlparse(url).path.strip("/")
                promo_penalty = 0.0
                if not purchase_intent and promotional_terms.search(title + " " + snippet):
                    promo_penalty += 0.62
                    item["_promo_page"] = True
                if not purchase_intent and not path and re.search(r"(専門サイト|公式サイト|ブリーダー|ショップ|販売)", title, re.IGNORECASE):
                    promo_penalty += 0.28
                    item["_promo_page"] = True
                if educational_title_terms.search(title):
                    item["relevance"] += 0.13
                # Wikipediaだけを特別扱いしない。公的機関・教育機関の情報は
                # 信頼性の観点で軽く評価するが、Wikipediaへの順位ボーナスは付けない。
                if host.endswith(("jkc.or.jp", "go.jp", "ac.jp")):
                    item["relevance"] += 0.07
                item["relevance"] = round(max(0.0, min(1.0, item["relevance"] - promo_penalty)), 3)

        ranked.sort(
            key=lambda item: (item["relevance"], bool(item.get("snippet")), len(item.get("snippet", ""))),
            reverse=True,
        )

        # 関連性の低い結果で水増ししない。一致が弱い場合は不確実性を明示して候補を返す。
        if detailed_topic:
            # 関連度が低いページで5件を埋めると、同じ単語が含まれるだけの
            # 別話題が回答に混ざりやすい。候補が少ない場合は件数を減らす。
            relevant = [item for item in ranked if item["relevance"] >= 0.12 and not item.get("_promo_page")]
        else:
            relevant = [item for item in ranked if item["relevance"] >= 0.06]
            if not relevant:
                relevant = ranked[:3]

        # 上位だけで同じドメインに偏らないように選ぶ。
        useful = []
        used_domains = set()
        for item in relevant:
            domain = item.get("_domain", "")
            if domain and domain not in used_domains:
                useful.append(item)
                used_domains.add(domain)
            elif not domain:
                useful.append(item)
            if len(useful) >= 5:
                break
        if not useful:
            if detailed_topic:
                return (
                    None,
                    0.0,
                    [],
                    f"「{query}」について、対象自体を説明する信頼できそうなページを選べなかったよ。"
                    "別の検索語を試すか、検索をやり直してみてね。",
                )
            useful = ranked[:3]

        domains = {item.get("_domain", "") for item in useful if item.get("_domain")}
        diversity = len(domains)
        mean_relevance = sum(item["relevance"] for item in useful[:3]) / max(1, min(3, len(useful)))
        confidence = min(0.88, 0.22 + mean_relevance * 0.58 + min(diversity, 3) * 0.055 + (0.04 if len(useful) >= 3 else 0.0))
        confidence = round(confidence, 3)

        if detailed_topic:
            fetched_paragraphs = {}
            # 詳しい説明を作るため、関連度の高い最大5ページから本文を取得する。
            # 取得中もUIに進捗を送る。本文取得に失敗したページは検索スニペットで補う。
            pages_to_fetch = useful[:5]
            for page_index, source in enumerate(pages_to_fetch, start=1):
                url = str(source.get("url", ""))
                if not url:
                    continue
                if callable(progress_callback):
                    try:
                        host = (urllib.parse.urlparse(url).hostname or "サイト").lower().removeprefix("www.")
                        progress_callback(f"{host} の本文を確認中... ({page_index}/{len(pages_to_fetch)})")
                    except Exception:
                        pass
                paragraphs = self._fetch_page_paragraphs(url)
                if paragraphs:
                    fetched_paragraphs[url] = paragraphs
            # 質問対象と焦点を別々に扱う。例えば「チワワの被毛にはどんな種類がある？」
            # では、検索語全体ではなく「チワワ」の話題に対する「被毛」の回答を作る。
            if topic_label and focus_label and focus_pattern:
                focused_answer = self._build_focused_answer(
                    topic_label, focus_label, focus_pattern, useful, fetched_paragraphs
                )
                if focused_answer:
                    answer = focused_answer
                elif focus_label == "使い方":
                    # 焦点語との一致が厳しく、操作説明を十分に抽出できなかった場合は、
                    # 取得済み本文からトピック概要を作る経路に切り替える。
                    overview_answer = self._build_topic_overview(
                        topic_label or query, useful, fetched_paragraphs
                    )
                    if overview_answer and "説明として単独で意味が通る情報を十分に抽出できなかった" not in overview_answer:
                        answer = (
                                f"「{topic_label}」の使い方について、取得できた本文から関連情報を整理するね。\n\n"
                                + overview_answer
                        )
                    else:
                        answer = (
                            f"「{topic_label}」の使い方について検索したけれど、"
                            "具体的な操作手順を確認できる情報が少なかったよ。\n\n"
                            "「Python print 使い方」「Python コード 実行方法」のように、"
                            "知りたい操作を加えて検索してみてね。"
                        )
                else:
                    answer = (
                        f"「{topic_label}」の{focus_label}について検索したけれど、"
                        "取得した本文から直接説明している箇所を十分に確認できなかったよ。\n\n"
                        "別の言い方で検索してみてね。"
                    )
            else:
                answer = self._build_topic_overview(topic_label or query, useful, fetched_paragraphs)
            # 詳細モードでは、複数ページの本文と検索スニペットを統合して回答を構成する。
            return answer, confidence, useful, None

        lines = []
        for result in useful[:3]:
            detail = self._clean_search_text(result.get("snippet") or result.get("title", ""))
            if len(detail) > 420:
                detail = detail[:417].rstrip() + "…"
            if detail:
                title = self._clean_search_text(result.get("title", "出典ページ"))
                url = result.get("url", "")
                lines.append(f"・{detail}\n  出典: {title} — {url}")

        if not lines:
            return (
                None,
                confidence,
                useful,
                "検索結果は見つかったけれど、回答に使える説明文を取り出せなかったよ。別のキーワードでも試してみてね。",
            )

        if re.search(r"(最新|現在|今日|今週|今月|ニュース|価格|値段|株価|為替|天気|運行状況|在庫)", query):
            intro = "関連する検索結果を見つけたよ。最新情報は更新時刻がページによって違うことがあるので、出典も確認してね。"
        elif re.search(r"(違い|比較|メリット|デメリット|どっち|どちら)", query):
            intro = "比較に役立ちそうな情報を集めたよ。下の出典ごとに説明の観点が違うこともあるので、条件をそろえて見比べてね。"
        elif re.search(r"(使い方|やり方|手順|どうやって|方法)", query):
            intro = "やり方に関する情報を探したよ。手順や対応バージョンはページごとに違う場合があるので、利用している環境に合うか確認してね。"
        elif re.search(r"(なぜ|なんで|どうして|理由|仕組み|原理|どういう意味|原因)", query):
            intro = "質問に関係する説明を検索結果から整理するね。ページの記述をもとにした要約なので、違う説明がある場合は出典も比べてね。"
        else:
            intro = "検索結果から、質問に関連する情報をまとめるね。"
        answer = intro + "\n\n" + "\n\n".join(lines)
        if confidence < 0.42 or mean_relevance < 0.08:
            answer += "\n\n※検索結果と質問の一致度が低めなので、参考候補として見てね。キーワードを絞ると、より合う情報を見つけやすいよ。"
        elif diversity < 2:
            answer += "\n\n※確認できた情報源が限られているため、内容は未確認の候補として扱ってね。"
        else:
            answer += "\n\n※これは検索結果の要約だよ。重要な判断では、リンク先の本文や更新日も確認してね。"

        for result in useful:
            result.pop("_domain", None)
            result.pop("_promo_page", None)
        return answer, confidence, useful, None

    def _save_research_candidate(self, query, answer, confidence, sources, allow_auto_register=True):
        """検索結果を候補として保存する。単語説明の長文は自動で知識回答に固定しない。"""
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

        # 高い関連度・高い信頼度・3つ以上の独立ドメインがそろった時だけ自動登録する。
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

        if allow_auto_register and confidence >= 0.80 and len(domains) >= 3 and answer:
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
        """明示的な検索依頼や、鮮度・外部事実が必要な問いを検索へ振り分ける。"""
        query = str(query or "").strip()
        if not query:
            return False

        intent = self._classify_intent(query)
        explicit_patterns = (
            r"https?://|www\.",
            r"(検索して|検索してみて|ウェブ検索|web検索|ネットで調べて|ネットで検索|調べて|調査して|詳しく調査|情報を集めて|出典を探して|ソースを探して)",
            r"(最新|今日のニュース|最新ニュース|最近のニュース|速報|現在の.*(価格|値段|状況|バージョン|天気)|今の天気|今日の天気|今日の.*(試合|結果|ニュース))",
            r"(発売日|販売価格|現在価格|株価|為替レート|公式サイト|公式ページ|営業時間|運行状況|在庫|料金|レビュー|評判|比較して)",
        )
        explicit_request = any(
            re.search(pattern, query, re.IGNORECASE)
            for pattern in explicit_patterns
        )
        # 検索を明示されていない好み・感情の共有は、話題語が含まれていても検索しない。
        # 例: 「犬が好き」「ゲームが楽しい」。ただし「犬が好きな理由を調べて」など
        # 明示的な検索依頼は上の explicit_request を優先する。
        if not explicit_request and re.search(
                r"(?:が|は)(?:大好き|好き|嫌い|苦手|得意|かわいい|可愛い|かっこいい|カッコいい|"
                r"面白い|おもしろい|楽しい|楽しかった|うれしい|嬉しい|最高|怖い|こわい|"
                r"好きじゃない|嫌いじゃない)(?:です|だよ|だね|だ|なんだ|なんです)?[。！!]*$",
                query,
        ):
            return False
        # 「首都を教えて」「ゲームの違いを説明して」など、疑問符のない知識依頼も拾う。
        factual_topic = bool(re.search(
            r"(首都|人口|面積|定義|意味|仕組み|原理|原因|理由|違い|比較|メリット|デメリット|"
            r"使い方|やり方|手順|仕様|スペック|対応OS|対応機種|発売日|発売時期|バージョン|"
            r"価格|値段|料金|株価|為替|天気|気温|ニュース|歴史|由来|何年|何月|何日|何時|どこ|誰)",
            query,
            re.IGNORECASE,
        ))
        knowledge_request = factual_topic and bool(re.search(
            r"(教えて|知りたい|説明して|まとめて|どんな|何|なに|いつ|どこ|誰|なぜ|なんで|どうして|"
            r"どうやって|違い|比較|使い方|どういう意味|どういう仕組み|[?？])",
            query,
            re.IGNORECASE,
        ))
        # 自分への質問・感情の共有・相談・雑談は、検索が必要な知識依頼のみ例外扱いにする。
        conversational_intents = {
            "ask_ai_about_self", "ask_ai_activity", "share_feeling", "check_in",
            "conversation_starter", "joke_request", "reflection_question",
            "decision_support", "friend_issue", "advice_request", "opinion_request",
            "recommendation_request", "casual_statement",
        }
        if intent in conversational_intents:
            return explicit_request or bool(knowledge_request and intent != "ask_ai_about_self" and intent != "ask_ai_activity")
        if intent == "ask_if_known":
            # 「橋って知ってる？」など、AIの行動についてではなく一般的な対象を知っているか
            # 尋ねる発言は、知識確認として検索する。AI自身への質問は会話で返す。
            asks_about_ai = bool(re.search(
                r"(?:君|あなた|お前|AngelWhite|アンジェルホワイト).{0,18}(?:知ってる|知ってますか|聞いたことある|分かる|わかる)",
                query,
                re.IGNORECASE,
            ))
            return not asks_about_ai
        if explicit_request or knowledge_request:
            return True
        # 事実型の質問なら respond() 側の分類と合わせて検索する。
        return intent == "factual_question"

    @classmethod
    def _contextual_answer(cls, query: str, context=None):
        """直前の会話内容と、前の返答で尋ねたことを使って会話をつなぐ。"""
        query = str(query or "").strip()
        if not query:
            return None

        # 外部検索が必要な質問・AI自身への質問・助言依頼を、短い会話の続きとして誤処理しない。
        intent = cls._classify_intent(query)
        if intent in {
            "factual_question", "ask_ai_about_self", "ask_ai_activity",
            "advice_request", "opinion_request", "recommendation_request",
            "general_question", "reflection_question", "decision_support",
            "friend_issue", "conversation_starter", "joke_request", "ask_if_known",
        } or re.search(r"(検索して|web検索|ウェブ検索|ネットで調べて|調査して|最新ニュース|今日の天気|株価|為替レート)", query, re.IGNORECASE):
            return None

        messages = []
        if isinstance(context, str):
            messages = [context]
        elif isinstance(context, (list, tuple)):
            for item in context:
                if isinstance(item, str):
                    value = item.strip()
                elif isinstance(item, dict):
                    value = str(item.get("text", item.get("content", item.get("user", "")))).strip()
                else:
                    value = str(item or "").strip()
                if value:
                    messages.append(value)
        elif isinstance(context, dict):
            raw_messages = context.get("messages", context.get("turns", []))
            if isinstance(raw_messages, (list, tuple)):
                for item in raw_messages:
                    if isinstance(item, dict):
                        value = str(item.get("text", item.get("content", item.get("user", "")))).strip()
                    else:
                        value = str(item or "").strip()
                    if value:
                        messages.append(value)

        # 空の新規チャットでは文脈を推測しない。
        if not messages:
            return None

        recent = messages[-8:]
        recent_text = " ".join(recent)
        last_ai = recent[-1] if recent else ""
        q = re.sub(r"[\s　]+", "", query)
        # 今回の発言も、既知情報を判定するための話題メモに含める。
        # これにより「明日サッカーの大会なんだ」の「サッカー」を見落とさない。
        combined_text = recent_text + " " + query
        context_q = re.sub(r"[\s　]+", "", combined_text)

        has_game = bool(re.search(r"(マイクラ|マインクラフト|Minecraft|ゲーム|フォートナイト|スプラ|ポケモン|Apex|エーペックス)", combined_text, re.IGNORECASE))
        has_building = bool(re.search(r"(建築|家を作|家を建|家づくり|大きい家|おしゃれな家|ブロック)", context_q))
        has_school = bool(re.search(r"(学校|授業|テスト|試験|宿題|課題|先生|教科)", context_q))
        has_sports = bool(re.search(r"(部活|クラブ活動|サッカー|バスケ|バスケット|野球|バレー|バレーボール|テニス|陸上|水泳|ラグビー|卓球|剣道|柔道|大会|試合|コンクール)", context_q))
        has_tournament = bool(re.search(r"(大会|試合|勝ちたい|優勝|対戦|相手チーム|緊張)", context_q))
        sport_patterns = (
            (r"サッカー", "サッカー"),
            (r"バスケットボール|バスケ", "バスケットボール"),
            (r"野球", "野球"),
            (r"バレーボール|バレー", "バレーボール"),
            (r"テニス", "テニス"),
            (r"陸上", "陸上"),
            (r"水泳", "水泳"),
            (r"卓球", "卓球"),
            (r"バドミントン", "バドミントン"),
            (r"ラグビー", "ラグビー"),
            (r"剣道", "剣道"),
            (r"柔道", "柔道"),
        )
        known_sport = next((name for pattern, name in sport_patterns
                            if re.search(pattern, combined_text)), None)
        known_club = next((re.search(pattern + r"(?:部|クラブ)", combined_text).group(0)
                           for pattern, _name in sport_patterns
                           if re.search(pattern + r"(?:部|クラブ)", combined_text)), None)
        knows_goal = bool(re.search(r"(勝ちたい|優勝したい|優勝を目指|勝つつもり|絶対勝つ)", combined_text))

        # ------------------------------------------------------------
        # 以前のAIの質問への短い回答を先に処理する
        # ------------------------------------------------------------
        if re.search(r"(何部に入ってる|何部なの|部活では|何の部活)", last_ai):
            club = re.search(r"(サッカー|バスケットボール|バスケ|野球|バレーボール|バレー|テニス|陸上|水泳|卓球|バドミントン|剣道|柔道|吹奏楽|美術|科学|演劇|合唱|ダンス|ラグビー|弓道|空手)(?:部|クラブ)", query)
            if club:
                name = club.group(0)
                return cls._reply_variant((
                    f"{name}なんだね！どのポジションや役割を担当してるの？",
                    f"{name}なんだ、いいね！部活ではどんな練習をすることが多いの？",
                    f"{name}なんだね！大会や発表会に出ることもあるの？",
                ))
            if re.search(r"(部だよ|部です|クラブだよ|クラブです)", query):
                return cls._reply_variant((
                    f"{query.rstrip('。！!')}なんだね！普段はどんな活動をしてるの？",
                    "そうなんだ！その部活で一番好きなところは何？",
                ))

        # スポーツ・部活の話題：競技名がすでに分かっている場合は聞き直さない。
        if re.search(r"(大会|試合|コンクール|発表会|リーグ戦|予選|決勝)", query) and has_sports:
            if re.search(r"(ある|出る|出場|参加|なんだ|だよ|です|近い|控えて)", query):
                if known_sport:
                    return cls._reply_variant((
                        f"{known_sport}の大会なんだね！どんな目標を立ててる？",
                        f"{known_sport}の大会に向けて頑張ってるんだね。今、特に力を入れている練習はある？",
                        f"{known_sport}の大会なんだね！チームの調子はどう？",
                    ))
                return cls._reply_variant((
                    "大会があるんだね！何の競技なの？分かったら、その競技に合わせて一緒に考えるよ。",
                    "大事な大会なんだね！どんな競技の大会なの？",
                    "大会に向けて頑張ってるんだね。何の競技か教えてくれる？",
                ))

        # 緊張や不安は直前の大会・試合に結びつけて受け止める。
        if re.search(r"(緊張|不安|心配|怖い|こわい|ドキドキ)", query) and has_sports and has_tournament:
            return cls._reply_variant((
                "大会前は緊張するよね。練習してきたことを一つずつ出せば大丈夫。特にどんなところが不安？",
                "大事な試合だからこそ緊張するんだね。深呼吸して、まず最初のプレーに集中するのもいいと思うよ。",
                "緊張するのは、それだけ本気で頑張りたいってことかもね。今いちばん気になっていることは何？",
            ))

        # 勝ちたい・目標を伝えてくれたときは、先ほどの大会・部活に直接反応。
        if re.search(r"(勝ちたい|優勝したい|勝てるといい|絶対勝つ|勝つつもり|勝ちたいんだ)", query) and has_sports:
            return cls._reply_variant((
                "勝ちたいよね！チームで声をかけ合って、自分ができるプレーに集中するのが大事だね。どんなところでチームに貢献したい？",
                "その気持ちいいね！大会までに特に伸ばしたいプレーや、意識している作戦はある？",
                "本気で勝ちたいんだね。これまで練習してきたことを出せるように、準備できることを一緒に整理しよう！",
            ))

        # スポーツの具体的な返答（ポジション、練習内容など）を話題に結びつける。
        if has_sports and re.search(r"(ポジション|フォワード|ミッドフィールダー|ディフェンダー|キーパー|ゴールキーパー|練習|シュート|パス|ドリブル|守備|攻撃|走り込み|筋トレ)", query):
            return cls._reply_variant((
                "そうなんだ！それを上達させるために、普段どんな練習をしてるの？",
                "なるほど、その部分が大事なんだね。今は得意なところ？それともこれから伸ばしたいところ？",
                "いいね！大会ではそこを活かせるとよさそうだね。ほかに意識していることはある？",
            ))

        # 学校の話：前の質問に答えた教科・宿題・テストを明確に引き継ぐ。
        if has_school and re.search(r"(数学|国語|英語|理科|社会|歴史|地理|宿題|テスト|試験|先生|友達|クラス)", query):
            topic = re.search(r"(数学|国語|英語|理科|社会|歴史|地理|宿題|テスト|試験|先生|友達|クラス)", query).group(1)
            if re.search(r"(苦手|難しい|むずかしい|分からない|わからない|できない|大変|失敗)", query):
                return f"{topic}が大変なんだね。どの部分で困ってる？ひとつずつ整理してみよう。"
            return cls._reply_variant((
                f"{topic}なんだね！どんなことがあったの？",
                f"なるほど、{topic}の話なんだね。今はどんなところが気になってる？",
            ))

        # 直前の質問が「どう感じた？」なら、感情を返した短文に対して内容を聞く。
        if re.search(r"(どう感じた|どんな気持ち|どう思った)", last_ai):
            if re.search(r"(うれしい|嬉しい|楽しい|楽しかった|緊張|不安|悲しい|悔しい|くやしい|怖い|安心|残念|腹が立つ)", query):
                return cls._reply_variant((
                    "その気持ちになったんだね。何がきっかけだったの？",
                    "そう感じたんだね。よかったら、そう思った理由も聞かせて。",
                ))

        # ゲーム内で家づくりの話が続いているとき。
        if re.search(r"(大きい|大きな|でかい|広い).{0,8}家", query) and (has_game or has_building):
            return "大きい家いいね！どんな素材で作ってるの？木材？石？それとも別のブロック？"
        if re.search(r"(木造|木材|木で作|石造|石で作|コンクリート|レンガ|クォーツ|羊毛)", query) and (has_game or has_building):
            material = re.search(r"(木造|木材|木|石造|石|コンクリート|レンガ|クォーツ|羊毛)", query)
            material_name = material.group(1) if material else "その素材"
            return f"{material_name}を使ってるんだね！いいね。屋根や内装はどんな感じにする予定？"
        if re.search(r"(家を作ってた|家を作った|家を建ててた|家を建てた|建築してた|建築した|建物を作ってた)", query) and (has_game or has_building):
            return "建築してたんだね！どんな家を作ってたの？大きい家？それともおしゃれな家？"
        if re.search(r"(冒険してた|冒険した|探索してた|探索した|探検してた|探検した|洞窟に行った|洞窟を探検)", query) and (has_game or re.search(r"(冒険|探索|洞窟|バイオーム|探検)", context_q)):
            return "冒険してたんだね！どこを探索してたの？洞窟？それとも新しいバイオーム？"

        # 直前のAI返答に含まれる質問に短い返答をした場合の軽い受け止め。
        if re.search(r"(何について|何のこと|もう少し教えて|どんな予定|何をしてた|何をしている)", last_ai):
            if len(q) <= 36 and not re.search(r"(検索して|とは何|って何|何ですか|何？|何\?)", q):
                return cls._reply_variant((
                    f"{query.rstrip('。！!')}なんだね！もう少し聞かせて。",
                    f"なるほど、{query.rstrip('。！!')}なんだ。具体的にはどんな感じ？",
                ))

        return None

    @classmethod
    def _classify_intent(cls, query: str):
        """会話の大まかな意図を分類する軽量ルール。意味理解モデルではない。"""
        raw = str(query or "").strip()
        if not raw:
            return "empty"
        q = re.sub(r"[\s　]+", "", raw)
        q_no_punc = re.sub(r"[。！!？?]+$", "", q)

        # 対象+焦点が明確な質問（例: 「チワワの被毛にはどんな種類がある？」）は、
        # 汎用雑談や文脈確認より先に情報検索対象として扱う。
        if cls._extract_focused_question(raw):
            return "factual_question"

        # 哲学的・内省的な問いは、辞書的な事実質問や検索に押し込めず対話として扱う。
        if re.search(r"(人生って何|生きる意味|幸せって何|幸せとは|愛って何|愛とは何|何のために生きる|自分らしさって何)", q):
            return "reflection_question"

        # 会話の開始、退屈、冗談の依頼などを先に識別する。
        if re.search(r"(なんか話そう|何か話そう|雑談しよう|何話そう|何話す|話し相手になって|暇だな|暇なんだ|退屈だな|つまらないな)", q):
            return "conversation_starter"
        if re.search(r"(面白い話して|笑わせて|冗談言って|ダジャレ言って|ジョーク言って|何か笑える話)", q):
            return "joke_request"

        # 迷い・決断の相談は、単なる一般質問ではなく選択肢を整理する。
        if re.search(r"(やめようかな|辞めようかな|続けようか迷|やめるべき|辞めるべき|どうしよう|迷ってる|迷っている|決められない|選べない|どっちにするか|するか悩んで|行くか迷って|断ろうかな)", q):
            return "decision_support"

        # 人間関係のトラブルを学校一般の話として処理しない。
        if re.search(r"(友達|友人|親友|家族|兄弟|姉妹|クラスメイト|同僚).{0,12}(けんか|喧嘩|仲直り|無視された|嫌われた|気まずい|言い合い|トラブル|裏切られ)", q) or re.search(r"(けんか|喧嘩|仲直り|無視された|嫌われた|気まずい).{0,12}(友達|友人|親友|家族|兄弟|姉妹|クラスメイト)", q):
            return "friend_issue"

        # 「○○って知ってるの？」系は、一般質問や事実検索より先に会話の話題確認として扱う。
        if re.search(
                r"(?:って|とは|について|のこと(?:を|は)?)(?:知ってる|知ってるの|知っている|知っていますか|知ってますか?|聞いたことある(?:の)?|分かる|わかる)(?:の)?$",
                q_no_punc,
        ) or re.search(
            r"^(?:君|あなた|お前|AngelWhite|アンジェルホワイト)(?:は|が).+(?:知ってる|知ってるの|知っている|知っていますか|知ってますか?|聞いたことある(?:の)?|分かる|わかる)(?:の)?$",
            q_no_punc,
            re.IGNORECASE,
        ):
            return "ask_if_known"

        # 外部情報を必要とする質問は、雑談と取り違えない。
        factual_markers = (
            r"とは|って何|ってなに|何ですか|何なの|何年|何月|何日|何時|"
            r"いつ発売|どこで買|いくら|値段|価格|最新|ニュース|バージョン|"
            r"誰が作|何が原因|何の意味|首都|人口|面積|定義|何色|何個|何匹|"
            r"何種類|何円|何cm|何kg|何歳|どこの国|どこの県|どの国|"
            r"どんな人物|どんな場所|いつから|いつまで|誰の作品|天気|気温|降水確率|"
            r"仕組み|原理|理由|どういう意味|どういう仕組み|どうやって作|何が起き|違い|比較|メリット|デメリット|原因|影響|特徴|使い方|やり方|手順|必要なもの|何に使う|どう動く|どう機能|いつ更新|リリース|対応機種|対応OS|仕様|スペック|安全性|危険性|どんな種類|どのような種類|何種類|被毛|毛の種類|毛質|平均寿命|体重|体高|原産地|どんな性格|どんな特徴|種類がある|どんな用途"
        )
        is_question = bool(
            re.search(r"[?？]$", raw)
            or re.search(r"(?:ですか|ますか|なの|かな|でしょうか|じゃない|じゃね|かい|の？|の\?)$", q)
            or re.search(r"^(?:なぜ|なんで|どうして(?!る|た)|どうやって|何で|なにで)", q)
        )
        has_explanatory_why = bool(re.search(r"^(?:なぜ|なんで|どうして(?!る|た)|どうやって|何で|なにで).{1,50}(?:？|\?|$)", q))
        if (is_question and re.search(factual_markers, q)) or has_explanatory_why:
            return "factual_question"

        # AI自身について聞かれているのか、ユーザー自身の行動について話しているのかを分ける。
        # 「僕」は一人称としてユーザーも使うので、AIを指す語の判定には含めない。
        subject_ai = bool(re.search(r"(?:君|あなた|お前|AngelWhite|アンジェルホワイト)", raw, re.IGNORECASE))
        play_words = r"(?:ゲーム|マイクラ|マインクラフト|Minecraft|フォートナイト|スプラ|ポケモン|Apex|エーペックス|勉強|運動|仕事|スポーツ|動画|音楽|読書)"
        # 「ゲーム好き？」は行動ではなく、AIの好みを聞いている質問として先に分類する。
        if re.search(r"(?:ゲーム|マイクラ|マインクラフト|Minecraft|スポーツ|音楽|読書)(?:は)?(?:好き|得意|楽しい)(?:なの|なのですか|ですか)?[？?。！!]*$", q, re.IGNORECASE) or re.search(r"好きな(?:ゲーム|スポーツ|音楽)(?:は何|はなに|は)?[？?。！!]*$", q):
            return "ask_ai_about_self"
        play_verbs = r"(?:やってる|してる|遊んでる|プレイしてる|している|やっている|遊ぶの|プレイするの|やるの|するの|できる|楽しんでる|やったことある)"
        if re.search(play_words + r".*" + play_verbs, raw, re.IGNORECASE) and (subject_ai or re.search(r"[?？]$", raw) or re.search(r"(?:の|か)$", q_no_punc)):
            return "ask_ai_activity"
        if re.search(r"(?:今|いま)?(?:何してる|なにしてる|何をしている|何している|何してた|なにしてた)(?:の)?$", q_no_punc):
            return "ask_ai_activity"
        if re.search(r"(?:元気|調子)どう|元気$", q_no_punc):
            return "check_in"
        if re.search(r"(?:君|あなた|お前|AngelWhite).*(?:何歳|年齢|住んで|出身|好きなゲーム|趣味|好きなもの)", raw, re.IGNORECASE):
            return "ask_ai_about_self"
        if re.search(r"(?:どうすれば|どうしたら|何をすれば|何したらいい|方法を教えて|アドバイスして|助けて)", q):
            return "advice_request"
        if re.search(r"(?:どう思う|どうかな|意見を聞かせて|ありだと思う|いいと思う)", q):
            return "opinion_request"
        if re.search(r"(?:おすすめ|オススメ).*(?:ある|教えて|知りたい)|何かおすすめ", q):
            return "recommendation_request"
        if re.search(r"(?:知ってる|知ってますか|聞いたことある|分かる|わかる)[？?]?$", q):
            return "ask_if_known"

        # 気持ち・状態の共有は、内容を聞き返して会話を続ける。
        if re.search(r"(?:疲れた|つかれた|しんどい|眠い|ねむい|悲しい|かなしい|寂しい|さびしい|不安|怖い|こわい|落ち込んだ|イライラ|腹が立つ|うれしい|嬉しい|楽しかった|面白かった|おもしろかった|最高だった|最悪だった|つらかった|辛かった|悔しい|くやしい|失敗した|うまくいかなかった|嫌なことがあった)", raw):
            return "share_feeling"
        if re.search(r"(?:これ|それ|あれ).*(?:どう思う|どうかな|どう？|どう\?)", q):
            return "opinion_request"
        if is_question:
            return "general_question"

        # 残りは事実質問ではない発言として扱う。雑談の返事で打ち切らないための fallback。
        return "casual_statement"

    @staticmethod
    def _reply_variant(options):
        """同じ意図でも返答が毎回固定になりすぎないよう、自然な候補から選ぶ。"""
        choices = [item for item in options if isinstance(item, str) and item.strip()]
        return random.choice(choices) if choices else None

    @classmethod
    def _general_conversation_answer(cls, query: str):
        """意図・話題の大枠から返答。未知の雑談は会話として受け止める。"""
        raw = str(query or "").strip()
        if not raw:
            return None
        q = re.sub(r"[\s\u3000]+", "", raw)
        intent = cls._classify_intent(raw)

        # 現時点の知識で確実に答えられない事実質問は捏造せず、後段の検索/確認へ。
        if intent == "factual_question":
            return None

        if intent == "conversation_starter":
            return cls._reply_variant((
                "いいね、何か話そう！最近ハマっているゲームや趣味の話にする？それとも今日あったことを聞かせて？",
                "もちろん！ゲーム、学校、動物、好きなものの話でもいいし、今気になっていることから始めてもいいよ。",
                "雑談しよう！最近うれしかったこと、気になっていること、何でも話してみて。",
            ))

        if intent == "joke_request":
            return cls._reply_variant((
                "じゃあ一つ！アルミ缶の上にあるみかん……って、ちょっとベタだったかな？もう一つ聞く？",
                "ダジャレを一つ。布団が吹っ飛んだ！……うん、王道だね。もっと変わったのも考えてみようか？",
                "軽いやつを一つ。トマトが赤くなったのは、なぜ？……『日焼けしたから』！どうかな？",
            ))

        if intent == "reflection_question":
            return cls._reply_variant((
                "大きな問いだね。人によって答えは違うけど、自分にとって大切な人や夢中になれることが、意味につながる場合もあると思う。何かそう考えるきっかけがあった？",
                "簡単に一つの答えを出せない問いだね。人それぞれ違う答えを持っていていいと思う。最近、そんなことを考える出来事があったのかな？",
                "僕には人間のような人生経験はないけど、一緒に考えることはできるよ。君は今、人生のどんなところについて考えている？",
            ))

        if intent == "decision_support":
            if re.search(r"(部活|クラブ|チーム|サークル)", raw):
                return cls._reply_variant((
                    "部活を続けるか迷っているんだね。すぐに決めなくても大丈夫。楽しいところ、つらいところ、続けた場合と辞めた場合を一緒に整理してみよう。迷う一番の理由は何？",
                    "辞めようか考えるくらい、何か引っかかることがあるんだね。人間関係、練習の負担、時間のことなど、何が一番大きい？",
                ))
            return cls._reply_variant((
                "迷っているんだね。急いで結論を出す前に、選択肢ごとの良い点・困る点と、君が一番大切にしたいことを整理しよう。何と何で迷っているの？",
                "一緒に考えよう。今の選択肢と、それぞれ心配なことを教えてくれたら、決める材料を整理できるよ。",
            ))

        if intent == "friend_issue":
            return cls._reply_variant((
                "友達とのことで嫌なことがあったんだね。無理にすぐ仲直りしなくてもいいよ。何が起きたのか、話せる範囲で聞かせてくれる？",
                "それは気まずいし、つらいね。よかったら、けんかのきっかけと今どうしたいと思っているかを聞かせて。一緒に考えよう。",
                "友達との関係って難しいこともあるよね。君の気持ちも大事だよ。今は仲直りしたい？それとも少し距離を置きたい？",
            ))

        if intent == "ask_ai_activity":
            if re.search(r"(ゲーム|マイクラ|マインクラフト|Minecraft|フォートナイト|スプラ|ポケモン|Apex|エーペックス)", raw, re.IGNORECASE):
                return cls._reply_variant((
                    "僕自身がゲームを操作して遊ぶことはできないけど、ゲームの話ならできるよ！君は何を遊んでるの？",
                    "僕は実際にプレイはできないんだ。でもゲームの攻略や建築の話なら一緒に考えられるよ。何のゲームをしてる？",
                    "自分でゲームを遊ぶことはできないけど、ゲームの話は大歓迎だよ！最近は何を遊んでるの？",
                ))
            return cls._reply_variant((
                "今は君と会話しているよ！ほかにも質問やアイデアを一緒に考えられるよ。",
                "今は君の話を聞いて、返事を考えているところだよ。君は今何をしてるの？",
            ))

        if intent == "ask_ai_about_self":
            if re.search(r"(ゲーム|マイクラ|マインクラフト|Minecraft|スポーツ|音楽|読書|好きなもの|趣味|フォートナイト|スプラ|ポケモン)", q, re.IGNORECASE):
                return cls._reply_variant((
                    "僕には人間のような個人的な趣味はないけど、ゲームや音楽の話を一緒に楽しみながら考えられるよ！君は何が好き？",
                    "人間みたいに自分の趣味を持っているわけではないんだ。でも、好きなものについて話すのは得意だよ。君のお気に入りは何？",
                    "僕自身の好みはないけど、いろいろなゲームや音楽について話せるよ！最近ハマっているものはある？",
                ))
            if re.search(r"(何歳|年齢)", q):
                return "僕は人間ではないから、実際の年齢はないよ。ほかにも気になることがある？"
            return "僕はAngelWhiteだよ。人間のような出身地や自宅はないけど、いろいろな話を一緒に考えられるよ。"

        if intent == "check_in":
            return cls._reply_variant((
                "聞いてくれてありがとう！僕は人間のような体調や気分はないけど、いつでも話を聞くよ。君は元気？",
                "僕はAIだから体調はないけど、君と話す準備はできているよ！今日はどう？",
            ))

        if intent == "advice_request":
            return cls._reply_variant((
                "一緒に考えよう！今の状況と、どうなったらうまくいきそうか教えてくれたら、手順を整理するよ。",
                "もちろん。何に困っているのか教えてくれる？できることを順番に整理しよう。",
                "手伝うよ！まず、何をしたいのか、どこで困っているのか聞かせて。",
            ))

        if intent == "opinion_request":
            return cls._reply_variant((
                "一緒に考えるよ！何についての意見がほしいか、内容を送ってくれる？良い点と気になる点を整理するよ。",
                "もちろん！対象の文章やアイデアを見せてくれたら、良いところと改善できそうなところを考えるよ。",
                "意見を考えるね。何についてなのか、もう少し詳しく教えて！",
            ))

        if intent == "recommendation_request":
            return cls._reply_variant((
                "いいよ！ゲーム、映画、音楽、勉強など、どんなジャンルのおすすめがいい？",
                "おすすめを考えよう！今の気分や好きなジャンルを教えてくれる？",
            ))

        if intent == "ask_if_known":
            subject = re.sub(r"(?:って|は)?(?:知ってる|知ってますか|聞いたことある|分かる|わかる)[？?]*$", "", q)
            if subject:
                return cls._reply_variant((
                    f"「{subject}」のことだね。どの部分について知りたい？分かる範囲で説明するよ。",
                    f"{subject}についてだね！知りたいことをもう少し具体的に教えてくれたら、分かる範囲で答えるよ。",
                ))
            return "何について知りたいの？話題を教えてくれたら、分かる範囲で説明するよ。"

        if intent == "share_feeling":
            if re.search(r"(疲れた|つかれた|しんどい|眠い|ねむい)", raw):
                return cls._reply_variant((
                    "お疲れさま！少し休めるなら休んでね。何か大変なことがあった？",
                    "今日は大変だったんだね。無理しすぎないでね。話したいことがあれば聞くよ。",
                    "お疲れさま。ひと息つけそう？何があったか話しても大丈夫だよ。",
                ))
            if re.search(r"(悲しい|かなしい|寂しい|さびしい|不安|怖い|こわい|落ち込んだ|イライラ|腹が立つ|最悪だった|つらかった|辛かった|悔しい|くやしい|失敗した|うまくいかなかった|嫌なことがあった)", raw):
                return cls._reply_variant((
                    "それはつらいね。話せる範囲で大丈夫だから、何があったか聞かせてくれる？",
                    "大変だったね。無理に説明しなくてもいいけど、よかったら話を聞かせて。",
                ))
            if re.search(r"(うれしい|嬉しい|楽しかった|面白かった|おもしろかった|最高だった|楽しみ)", raw):
                return cls._reply_variant((
                    "それはよかった！何が一番うれしかった、または楽しかったの？",
                    "いいことがあったんだね！よかったら、何があったのか教えて！",
                    "それはいいね！どんなところが特にうれしかった？",
                ))

        if intent == "general_question":
            return cls._reply_variant((
                "その話、もう少し聞かせて！何を知りたいのか一緒に整理するよ。",
                "もう少し詳しく教えてくれる？状況が分かれば、できるだけ役に立つ返事をするよ。",
                "なるほど。どの部分が気になっているのか教えてくれたら、一緒に考えるよ。",
            ))

        # 同意を求める会話。話題に合わせて応答を変える。
        if re.search(r"(かわいい|可愛い|かっこいい|カッコいい|すごい|最高|面白い|おもしろい|いいよね|いいね)[よねだね！!。]*$", q):
            if re.search(r"(猫|ねこ|ネコ)", raw):
                return cls._reply_variant((
                    "猫ってかわいいよね！しぐさや表情も魅力的だよね。猫を飼ってるの？",
                    "分かる、猫の自由気ままなしぐさってかわいいよね。猫と暮らしてるの？",
                    "猫は見ているだけでも癒やされるよね！好きな種類はある？",
                ))
            if re.search(r"(犬|いぬ|イヌ|わんちゃん|ワンちゃん|動物|ペット)", raw):
                return cls._reply_variant((
                    "分かる！動物のしぐさって見ているだけで癒やされるよね。犬や猫と暮らしてる？",
                    "かわいいよね！どんなところが一番好き？しぐさ？表情？それとも一緒に過ごす時間かな？",
                ))
            if re.search(r"(マイクラ|マインクラフト|Minecraft|ゲーム)", raw, re.IGNORECASE):
                return cls._reply_variant((
                    "そうだね！ゲームのどんなところが特に好き？建築、冒険、それとも別のところ？",
                    "いいよね！どの部分が一番楽しいと思う？",
                ))
            return cls._reply_variant((
                "そうだね！そう感じた理由も気になるな。どの部分が特にいいと思った？",
                "たしかに、いいところだよね！特に気に入っている部分はどこ？",
            ))

        # 予定・イベント。予定の種類ごとに別の返答を選び、汎用的な返答に偏らないようにする。
        future_words = r"(明日|あした|今度|来週|週末|これから|今日のあと|明後日|あさって|再来週)"
        has_future = bool(re.search(future_words, raw))

        # テスト・試験：教科や範囲など、次の会話に役立つ情報を聞く。
        if has_future and re.search(r"(テスト|試験|小テスト|期末|中間|模試)", raw):
            return cls._reply_variant((
                "テストなんだね！何の教科？範囲が分かれば、勉強する順番も一緒に考えられるよ。",
                "そっか、テストがあるんだね。何の教科で、どのあたりが気になってる？",
                "テスト前なんだね！もう勉強を始めてる？必要なら短い勉強計画を一緒に作ろう。",
            ))

        # 試合・大会：発言内ですでに競技名が分かる場合は、競技を聞き直さない。
        if has_future and re.search(r"(試合|大会|コンクール|試合に出|試合が)", raw):
            sport_patterns = (
                (r"サッカー", "サッカー"),
                (r"バスケットボール|バスケ", "バスケットボール"),
                (r"野球", "野球"),
                (r"バレーボール|バレー", "バレーボール"),
                (r"テニス", "テニス"),
                (r"陸上", "陸上"),
                (r"水泳", "水泳"),
                (r"卓球", "卓球"),
                (r"バドミントン", "バドミントン"),
                (r"ラグビー", "ラグビー"),
                (r"剣道", "剣道"),
                (r"柔道", "柔道"),
            )
            sport = next((name for pattern, name in sport_patterns if re.search(pattern, raw)), None)
            if sport:
                return cls._reply_variant((
                    f"{sport}の大会なんだね！どんな目標を立ててる？",
                    f"明日は{sport}の大会なんだね！今、特に力を入れている練習はある？",
                    f"{sport}の大会、楽しみだね！緊張してる？それとも楽しみな気持ちのほうが大きい？",
                ))
            return cls._reply_variant((
                "試合があるんだね！何の競技なの？楽しみな感じ？それとも緊張してる？",
                "大事な日になりそうだね。何の大会なの？準備や作戦を一緒に整理することもできるよ.",
            ))

        # 発表・面接：テーマや準備状況を確認する。
        if has_future and re.search(r"(発表|プレゼン|面接|面談)", raw):
            if re.search(r"(面接|面談)", raw):
                return cls._reply_variant((
                    "面接があるんだね！どんな面接なの？よく聞かれそうな質問への答えを一緒に練習できるよ。",
                    "なるほど、面接の予定なんだね。何についての面接か教えてくれたら、準備を整理しよう。",
                ))
            return cls._reply_variant((
                "発表があるんだね！何について発表するの？構成や話す内容を一緒に考えられるよ。",
                "そうなんだ！発表の準備はどこまで進んでる？必要なら要点をまとめよう。",
            ))

        # 旅行・お出かけ：目的地や楽しみにしていることを聞く。
        if has_future and re.search(r"(旅行|お出かけ|出かける|遊びに行|泊まり|観光)", raw):
            return cls._reply_variant((
                "旅行に行くんだね！どこに行く予定なの？一番楽しみにしていることはある？",
                "いいね、旅行の予定があるんだ！もう行き先ややりたいことは決まってる？",
                "お出かけするんだね！誰と行く予定？準備や持ち物を一緒に考えることもできるよ。",
            ))

        # 部活・仕事・アルバイト。発言内に部活名があれば、所属を聞き直さない。
        if has_future and re.search(r"(部活|クラブ活動|サッカー部|バスケットボール部|バスケ部|野球部|バレーボール部|バレー部|テニス部|陸上部|水泳部|卓球部|バドミントン部|吹奏楽部|美術部|科学部|演劇部|合唱部|ダンス部|仕事|バイト|アルバイト)", raw):
            club_patterns = (
                (r"サッカー(?:部|クラブ)?", "サッカー"),
                (r"バスケットボール部|バスケ部", "バスケットボール"),
                (r"野球部", "野球"),
                (r"バレーボール部|バレー部", "バレーボール"),
                (r"テニス部", "テニス"),
                (r"陸上部", "陸上"),
                (r"水泳部", "水泳"),
                (r"卓球部", "卓球"),
                (r"バドミントン部", "バドミントン"),
                (r"吹奏楽部", "吹奏楽"),
                (r"美術部", "美術"),
                (r"科学部", "科学"),
                (r"演劇部", "演劇"),
                (r"合唱部", "合唱"),
                (r"ダンス部", "ダンス"),
            )
            club_name = next((name for pattern, name in club_patterns if re.search(pattern, raw)), None)
            if club_name:
                return cls._reply_variant((
                    f"明日は{club_name}の活動なんだね！どんな練習をする予定？",
                    f"{club_name}なんだね！明日は練習中心？それとも試合や発表があるの？",
                    f"{club_name}の活動なんだ、いいね！今、特に上達させたいことはある？",
                ))
            return cls._reply_variant((
                "明日は部活なんだね！何部に入ってるの？練習や試合の予定かな？" if re.search(r"(部活|クラブ活動)", raw) else "仕事の予定なんだね。どんなことをする日なの？準備で気になることがあれば一緒に整理しよう。",
                "そうなんだ！" + ("部活では明日どんなことをする予定？" if re.search(r"(部活|クラブ活動)", raw) else "仕事の前に準備しておきたいことはある？"),
            ))

        # 学校・授業は具体的に何があるのかを確認する。
        if has_future and re.search(r"(学校|授業|登校)", raw):
            return cls._reply_variant((
                "明日は学校なんだね！授業や行事で何か特別な予定があるの？",
                "学校があるんだね。明日は楽しみなことがある？それとも気になることがあるのかな？",
            ))

        # そのほかの予定・用事。
        event_words = r"(予定|用事|約束|行事|宿題|課題|準備|イベント)"
        if has_future and re.search(event_words, raw):
            return cls._reply_variant((
                "そうなんだ！どんな予定なの？よかったら聞かせて。",
                "予定があるんだね。楽しみな予定かな？それとも準備を手伝ったほうがよさそう？",
                "なるほど！いつ、どんなことをする予定なの？必要なら一緒に段取りを整理しよう。",
            ))

        # 話題共有を広く拾う。特定の語尾だけに依存しない。
        if re.search(r"(学校|授業|テスト|試験|宿題|課題|部活|先生|友達|クラス)", raw):
            if re.search(r"(忙しい|大変|疲れた|難しい|むずかしい|困った|失敗)", raw):
                return cls._reply_variant((
                    "学校のことで大変なんだね。どのあたりが一番大変？",
                    "忙しそうだね。勉強や活動のことで、何か手伝えることはある？",
                    "それは大変そう。よかったら、何があったか聞かせて。",
                ))
            return cls._reply_variant((
                "学校の話なんだね。今日はどんなことがあったの？",
                "なるほど、学校のことなんだね。何か印象に残ったことはあった？",
                "学校ではどんな感じだった？楽しかったことでも大変だったことでも聞くよ。",
            ))

        if re.search(r"(マイクラ|マインクラフト|Minecraft|フォートナイト|スプラ|ポケモン|ゲーム|Apex|エーペックス)", raw, re.IGNORECASE):
            if re.search(r"(作って|建築|家|建て|ブロック|クラフト)", raw):
                return cls._reply_variant((
                    "建築してたんだね！どんなものを作っているの？素材やデザインも聞きたいな。",
                    "いいね、建築は工夫できるところが多いよね。どんな家や建物を作ってるの？",
                ))
            if re.search(r"(冒険|探索|洞窟|バイオーム|探検|村を見つけ)", raw):
                return cls._reply_variant((
                    "冒険してたんだね！どこを探索したの？何か面白いものは見つかった？",
                    "いいね！どこまで探検したの？珍しい場所やアイテムは見つけた？",
                ))
            return cls._reply_variant((
                "そうなんだ！ゲームでは何をしてたの？建築、冒険、それとも別のこと？",
                "いいね！最近はどんな遊び方をしてるの？",
                "ゲームの話いいね！今は何にハマってるの？",
            ))

        # 過去/現在/未来の出来事を区別して質問する。
        if re.search(r"(明日|あした|今度|来週|週末|これから|楽しみ|予定)", raw):
            return cls._reply_variant((
                "そうなんだ！どんな予定なの？よかったら聞かせて。",
                "いいね、何があるの？楽しみなことなら詳しく聞きたいな。",
            ))
        if re.search(r"(今日|昨日|さっき|最近|今週)", raw):
            return cls._reply_variant((
                "そうなんだ！そのあとどうなったの？もう少し聞かせて。",
                "なるほど！その話、もう少し詳しく聞きたいな。",
                "そうだったんだね。いちばん印象に残ったことは何？",
            ))

        # 特定の話題を認識できない普通の発言でも、即座に知識不足の定型文で終わらせない。
        return cls._reply_variant((
            "そうなんだ！もう少し詳しく聞かせて。",
            "なるほど、そういうことがあったんだね。続きも聞きたいな。",
            "そうなんだね。君はそれについてどう感じた？",
            "へえ、興味深いね！もう少し教えてくれる？",
        ))

    def respond(self, text: str, mode: str = "一般モード", context=None, progress_callback=None) -> str:
        raw = text.strip()
        if not raw:
            return "何か話しかけてみてね。"

        # 数字選択はプロフィール検出や通常会話より先に処理する。
        selected = self._handle_topic_choice(raw, context, progress_callback=progress_callback)
        if selected is not None:
            return selected

        # 1. まず「情報を教える発言」かを判定する。
        # 「好きな動物は猫です」を質問と誤認して、古い記憶を返す不具合を防ぐ。
        detected_fact = self._detect_fact(raw)
        if detected_fact:
            label, value = detected_fact
            # 記憶設定が有効なら好みを保存するが、雑談の返答そのものは設定に依存させない。
            self._remember_turn(raw, f"{label}を{value}として記憶したよ。")
            if label in ("好きな動物", "好きな色", "好きなもの"):
                opening = self._reply_variant((
                    f"{value}が好きなんだね！",
                    f"へえ、{value}が好きなんだ！",
                    f"なるほど、{value}が好きなんだね。",
                    f"{value}が好きなんだね。いいね！",
                ))
                invitation = self._reply_variant((
                    "よろしければ、こんなこともできるよ。",
                    "よかったら、次の中から選んでね。",
                    "この話題で、こんなことができるよ！",
                    "続けるなら、次のどれかを選べるよ。",
                ))
                return (
                    f"{opening}\n\n{invitation}\n"
                    "1. そのまま雑談を続ける\n"
                    "2. Web検索をする\n"
                    "3. 終了する"
                )
            return self._reply_variant((
                f"教えてくれてありがとう！ {label}を記憶したよ。",
                f"なるほど！ {label}として覚えておくね。",
                f"了解！ {label}のことを記憶したよ。",
            ))

        # 2. 質問なら、保存済みのユーザー記憶を確認する。Web検索より優先。
        remembered = self._memory_answer(raw)
        if remembered is not None:
            self._remember_turn(raw, remembered)
            return remembered

        # 3. ユーザーが明示的に教えた知識を保存する。
        taught = self._teach(raw)
        if taught is not None:
            self._remember_turn(raw, taught)
            return taught

        # 4. まず入力が単語・短いトピック名かを判定する。
        # 以前のバージョンが保存した web_auto 回答が残っている場合、古い要約を
        # そのまま返すと要約改善が反映されないため、トピック入力では再検索を優先する。
        key = self._normalize(raw)
        standalone_topic = self._extract_standalone_topic(raw)
        pair = self.knowledge["pairs"].get(key)
        if standalone_topic and isinstance(pair, dict) and pair.get("source") != "user_taught":
            # ユーザー自身が教えた回答は保持し、検索由来のキャッシュだけを無効化。
            self.knowledge["pairs"].pop(key, None)
            self.knowledge["counts"].pop(key, None)
            self._save_knowledge()
            pair = None
        focused_question = self._extract_focused_question(raw)
        if focused_question and not (isinstance(pair, dict) and pair.get("source") == "user_taught"):
            topic_label, focus_label, focus_surface = focused_question
            # 焦点別に検索語を短く整理する。質問全文だけでなく、対象+焦点語で検索結果を選ぶ。
            focus_query_map = {
                "被毛・毛の種類": "被毛 毛の種類 ロングコート スムースコート",
                "性格": "性格 特徴",
                "飼い方・健康管理": "飼い方 健康管理 注意点",
                "平均寿命": "平均寿命 寿命",
                "体重・大きさ": "体重 体高 大きさ",
                "原産地・歴史": "原産地 歴史 由来",
                "構造・仕組み": "構造 仕組み 原理",
                "特徴": "特徴 基本情報",
                "種類": "種類 タイプ 分類",
                "使い方": "使い方 方法",
                "違い": "違い 比較",
                "原因・理由": "原因 理由 仕組み",
                "値段・費用": "値段 価格 費用",
                "用途・役割": "用途 役割 使い道",
            }
            focus_regex_map = {
                "被毛・毛の種類": r"(被毛|毛の長さ|毛の種類|毛質|スムースコート|ロングコート|短毛|長毛|被毛の種類)",
                "性格": r"(性格|気質|人懐こ|警戒心|活発|おとなしい|忠実)",
                "飼い方・健康管理": r"(飼い方|飼育|健康管理|運動|食事|しつけ|注意|病気|予防)",
                "平均寿命": r"(平均寿命|寿命|年生き|歳まで)",
                "体重・大きさ": r"(体重|体高|大きさ|サイズ|kg|センチ|cm)",
                "原産地・歴史": r"(原産地|原産国|歴史|由来|起源|発祥|年に誕生)",
                "構造・仕組み": r"(構造|仕組み|原理|働き|どう動く|作用)",
                "特徴": r"(特徴|特長|代表的|備える|持つ|見られる)",
                "種類": r"(種類|タイプ|分類|に分かれる|形式)",
                "使い方": r"(使い方|利用方法|操作|手順|方法)",
                "違い": r"(違い|一方|対して|比べ|異なる|差がある)",
                "原因・理由": r"(原因|理由|ために|によって|仕組み)",
                "値段・費用": r"(値段|価格|費用|料金|円程度|相場)",
                "用途・役割": r"(用途|役割|使い道|何に使|利用され)",
            }
            focus_query = focus_query_map.get(focus_label, focus_surface)
            search_query = f"{topic_label} {focus_query}"
            researched, confidence, sources, error = self._research_answer(
                search_query,
                detailed_topic=True,
                topic_label=topic_label,
                focus_label=focus_label,
                focus_pattern=focus_regex_map.get(focus_label, re.escape(focus_surface)),
                progress_callback=progress_callback,
            )
            if researched:
                answer = researched
                self._save_research_candidate(raw, answer, confidence, sources, allow_auto_register=False)
            else:
                answer = (error or f"「{topic_label}」の{focus_label}について検索できなかったよ。")
                self._save_research_candidate(raw, answer, confidence, sources, allow_auto_register=False)
            self._remember_turn(raw, answer)
            return answer

        known_topic = self._extract_known_topic(raw) if self._classify_intent(raw) == "ask_if_known" else None
        if known_topic:
            learned_answer = ""
            if pair and isinstance(pair, dict) and pair.get("source") == "user_taught":
                learned_answer = str(pair.get("answer", "")).strip()
            answer = self._build_topic_menu(known_topic, learned_answer)
            self._remember_turn(raw, answer)
            return answer
        if pair:
            self.knowledge["counts"][key] = (
                    self.knowledge["counts"].get(key, 0) + 1
            )
            self._save_knowledge()
            answer = pair.get("answer", "まだ返答が登録されていないよ。")
            self._remember_turn(raw, answer)
            return answer

        # 単語・短い名詞句だけの入力は「その話題を詳しく説明して」の意味として扱う。
        # 例: 「チワワ」「橋」「量子コンピュータ」。挨拶・雑談文・1/2/3は除外済み。
        if standalone_topic:
            researched, confidence, sources, error = self._research_answer(
                standalone_topic, detailed_topic=True, progress_callback=progress_callback
            )
            if researched:
                answer = researched
                self._save_research_candidate(raw, answer, confidence, sources, allow_auto_register=False)
            else:
                answer = (
                    f"「{standalone_topic}」について調べようとしたけど、今は検索結果を取得できなかったよ。\n"
                    "インターネット接続を確認するか、「チワワの特徴」のように少し詳しく入力してみてね。"
                )
                if error:
                    answer += f"\n\n検索エラー: {error}"
                self._save_research_candidate(raw, answer, confidence, sources, allow_auto_register=False)
            self._remember_turn(raw, answer)
            return answer

        now = datetime.now()
        known_topic = self._extract_known_topic(raw) if self._classify_intent(raw) == "ask_if_known" else None
        if known_topic:
            answer = self._build_topic_menu(known_topic)
            self._remember_turn(raw, answer)
            return answer
        contextual = self._contextual_answer(raw, context)

        if contextual is not None:
            answer = contextual
        elif self._is_greeting(raw):
            answer = self._reply_variant((
                "こんにちは！ AngelWhiteだよ。今日は何を話そうか？",
                "やあ！来てくれてありがとう。何か話したいことはある？",
                "こんにちは！雑談でも質問でも、気軽に話しかけてね。",
                "おっ、こんにちは！今日はどんな話をしようか？",
            ))
        elif re.search(r"(ありがとう|ありがと|助かった|感謝)", raw):
            answer = self._reply_variant((
                "どういたしまして！ またいつでも話しかけてね。",
                "役に立てたならうれしいよ！ほかにもあれば言ってね。",
                "こちらこそ、話してくれてありがとう！",
                "いつでもどうぞ！また気軽に聞いてね。",
            ))
        elif re.search(r"(さようなら|じゃあね|またね|おやすみ)", raw):
            answer = self._reply_variant((
                "またね！ いつでも話しかけてね。",
                "話せてよかったよ。またいつでも来てね！",
                "じゃあね！また話そう。",
                "おやすみ。ゆっくり休んでね！" if re.search(r"おやすみ", raw) else "またね！次に話せるのを待ってるよ。",
            ))
        elif re.search(r"(君の名前|あなたの名前|名前は何|誰なの|何者)", raw):
            answer = self._reply_variant((
                "僕はAngelWhite。独自の回答エンジンと知識記憶を少しずつ開発しているよ。",
                "AngelWhiteだよ！検索や会話ができる独自のAIを目指して開発中なんだ。",
                "僕の名前はAngelWhite。これから少しずつできることを増やしていくよ。",
            ))
        elif re.search(r"(何時|今の時間|現在時刻)", raw):
            answer = f'このPCの現在時刻は {now.strftime("%H:%M")} だよ。'
        elif re.search(r"(今日の日付|今日は何日|何月何日|今日の日にち)", raw):
            answer = f'今日は {now.strftime("%Y年%m月%d日")} だよ。'
        elif re.search(r"^覚えて[：:]?", raw):
            answer = "覚えたい内容は「覚えて: 質問 => 返答」の形で教えてね。たとえば「覚えて: 橋って知ってる？ => ああ、川にあるやつだよね」だよ。"
        elif re.search(
                r"(?:俺|僕|私|わたし|自分)は.{1,80}(?:が好き|が嫌い|が苦手|が得意)",
                raw,
        ) or re.search(r"^.{1,30}が好き(?:です|だよ|だ|！|!|。)?$", raw):
            # この返答をした後、_remember_turn() が発言内容からプロフィール情報を保存する。
            detected = self._detect_fact(raw)
            if detected:
                label, value = detected
                if label in ("好きな色", "好きなもの", "好きな動物"):
                    answer = self._reply_variant((
                        f"{value}が好きなんだね！教えてくれてありがとう。",
                        f"へえ、{value}が好きなんだ！いいね。",
                        f"なるほど、{value}がお気に入りなんだね。",
                        f"{value}が好きなんだね。どんなところが好きなの？",
                        f"いいね！{value}が好きなんだ。好きになったきっかけはある？",
                    ))
                else:
                    answer = self._reply_variant((
                        "そうなんだね！教えてくれてありがとう。",
                        "なるほど、そういう好みなんだね。",
                        "へえ、そうなんだ！もう少し聞いてみたいな。",
                    ))
            else:
                match = re.search(
                    r"(?:俺|僕|私|わたし|自分)は(.{1,60}?)(が好き|が嫌い|が苦手|が得意)",
                    raw,
                )
                if match:
                    subject = match.group(1).strip()
                    reaction_options = {
                        "が好き": (
                            f"{subject}が好きなんだね！どんなところが好き？",
                            f"へえ、{subject}が好きなんだ！",
                            f"なるほど、{subject}がお気に入りなんだね。",
                            f"いいね！{subject}が好きなんだね。",
                        ),
                        "が嫌い": (
                            f"{subject}は苦手なんだね。何か理由があるの？",
                            f"そっか、{subject}は好きじゃないんだね。",
                            f"なるほど、{subject}が苦手なんだね。",
                        ),
                        "が苦手": (
                            f"{subject}が苦手なんだね。どのあたりが難しい？",
                            f"そっか、{subject}はちょっと苦手なんだね。",
                            f"なるほど！話してくれてありがとう。",
                        ),
                        "が得意": (
                            f"{subject}が得意なんだ、すごいね！",
                            f"いいね！{subject}が得意なんだね。",
                            f"へえ、{subject}が得意なんだ！いつから得意なの？",
                        ),
                    }
                    answer = self._reply_variant(reaction_options[match.group(2)])
                else:
                    answer = "そうなんだね！ 教えてくれてありがとう。"
        elif re.fullmatch(
                r"(?:俺|僕|私|わたし|自分)は(?:犬|猫|ねこ|ネコ|狼|オオカミ|うさぎ|ウサギ|ライオン|パンダ)(?:だよ|です|だ|なんだよ|なんです)?[！!。]?",
                raw,
        ):
            # 自分を動物に例える発言は、プロフィールの「好きな動物」には保存しない。
            answer = self._reply_variant((
                "そうなんだ！犬の話でも、ほかの話でもできるよ。今日は何を話そうか？",
                "なるほど！動物の話をする？それとも別の話題にする？",
                "そうなんだね！ほかにも好きなものがあったら教えて。",
            ))
        elif re.search(r"(マイクラ|マインクラフト).*(やってる|遊んでる|プレイしてる|したことある)", raw):
            answer = self._general_conversation_answer(raw) or "僕自身がゲームを操作して遊ぶことはできないけど、マイクラの話ならできるよ！今日は何をしてたの？"
        elif re.search(r"(疲れた|つかれた|しんどい|眠い|ねむい)", raw):
            answer = self._general_conversation_answer(raw) or "お疲れさま！ 少し休めるなら休んでね。今日は何か大変なことがあった？"
        elif re.search(r"(今日は|さっき|昨日).*(マイクラ|マインクラフト|ゲーム).*(やってた|遊んでた|してた|作った)", raw):
            answer = self._general_conversation_answer(raw) or "いいね！ 今日はマイクラで何をしてたの？ 建築したり、冒険したりした？"
        elif self._should_web_search(raw) or self._classify_intent(raw) == "factual_question":
            # 明示依頼だけでなく、明確な事実・仕組み・理由を尋ねる質問も検索候補にする。
            # 雑談や相談は _classify_intent() で別扱いにして検索へ流さない。
            researched, confidence, sources, error = self._research_answer(raw, progress_callback=progress_callback)
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
            # 共通の会話ルーターを使い、質問文ごとの個別パターン追加だけに依存しない。
            answer = self._general_conversation_answer(raw)
            if answer is None:
                answer = (
                    "今の内蔵知識だけでは、事実を正確に答えられないみたい。\n"
                    "最新情報や具体的な事実を調べたいときは「〇〇を検索して」と頼んでね。"
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