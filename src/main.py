import sys
import json
import mimetypes
import math
from pathlib import Path

from PySide6.QtCore import Qt, QPoint, QPointF, QEvent, QThread, Signal
from PySide6.QtGui import QFont, QPainter, QColor, QPixmap, QPolygonF
from ai_engine import JapaneseAI

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QFrame,
    QMenu,
    QSizePolicy,
    QFileDialog,
    QGraphicsDropShadowEffect,
)


# =========================================================
# パス
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHAT_FILE = DATA_DIR / "chats.json"
MEMORY_SETTINGS_FILE = DATA_DIR / "memory_settings.json"


# =========================================================
# 設定アイコン（絵文字・フォントに依存しない独自の線画）
# =========================================================

class SettingsIconButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__("", parent)
        self.setToolTip("設定")
        self.setFixedSize(42, 42)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                border-radius: 9px;
            }
            QPushButton:hover { background-color: #292929; }
            QPushButton:pressed { background-color: #333333; }
        """)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor("#eeeeee" if self.underMouse() or self.isDown() else "#a0a0a0")
        # 塗りつぶしではなく細い線で描き、文字アイコンのように軽い印象にする
        pen = painter.pen()
        pen.setColor(color)
        pen.setWidthF(1.7)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        # 12枚の歯を持つギアの輪郭を描画
        cx, cy = self.width() / 2, self.height() / 2
        points = []
        teeth = 10
        for i in range(teeth * 4):
            angle = -math.pi / 2 + (2 * math.pi * i / (teeth * 4))
            phase = i % 4
            radius = (13.0, 13.0, 10.2, 10.2)[phase]
            points.append((cx + math.cos(angle) * radius,
                           cy + math.sin(angle) * radius))
        painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in points]))
        # 中央の穴も細い輪郭線で表現
        painter.drawEllipse(QPointF(cx, cy), 4.0, 4.0)
        painter.end()


# =========================================================
# ゼムクリップアイコン（絵文字ではなく、UI用の線画）
# =========================================================

class PaperclipIcon(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(20, 24)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = painter.pen()
        pen.setColor(QColor("#eeeeee"))
        pen.setWidthF(1.8)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        # 細い線で描く、シンプルなゼムクリップ形状
        from PySide6.QtGui import QPainterPath
        path = QPainterPath()
        path.moveTo(13.8, 7.5)
        path.lineTo(13.8, 16.0)
        path.cubicTo(13.8, 19.0, 11.8, 21.0, 9.3, 21.0)
        path.cubicTo(6.8, 21.0, 5.0, 19.0, 5.0, 16.5)
        path.lineTo(5.0, 6.7)
        path.cubicTo(5.0, 4.6, 6.4, 3.2, 8.2, 3.2)
        path.cubicTo(10.0, 3.2, 11.3, 4.6, 11.3, 6.5)
        path.lineTo(11.3, 16.0)
        path.cubicTo(11.3, 17.2, 10.6, 18.0, 9.5, 18.0)
        path.cubicTo(8.4, 18.0, 7.7, 17.2, 7.7, 16.0)
        path.lineTo(7.7, 7.8)
        painter.drawPath(path)
        painter.end()


# =========================================================
# AI回答のコピーアイコン（Android風の独自線画）
# =========================================================

class CopyAnswerButton(QPushButton):
    """重なった縦長の紙を線画で描く、AI回答用のコピーアイコン。"""
    def __init__(self, parent=None):
        super().__init__("", parent)
        self.setObjectName("copyAnswerButton")
        self.setToolTip("回答をコピー")
        self.setAccessibleName("回答をコピー")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(30, 28)
        self.setStyleSheet("""
            QPushButton#copyAnswerButton {
                background-color: transparent;
                border: none;
                border-radius: 6px;
                padding: 0px;
            }
            QPushButton#copyAnswerButton:hover {
                background-color: #262626;
            }
            QPushButton#copyAnswerButton:pressed {
                background-color: #333333;
            }
        """)

    def paintEvent(self, event):
        # QPushButton側にはホバー時などの背景だけを描かせる。
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor("#eeeeee" if self.underMouse() or self.isDown() else "#8b8b8b")
        pen = painter.pen()
        pen.setColor(color)
        pen.setWidthF(1.65)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        from PySide6.QtGui import QPainterPath
        # 奥側の紙：縦長で少し左上に配置。
        back = QPainterPath()
        back.moveTo(8.5, 4.5)
        back.lineTo(16.0, 4.5)
        back.quadTo(17.7, 4.5, 17.7, 6.2)
        back.lineTo(17.7, 19.0)
        back.quadTo(17.7, 20.5, 16.0, 20.5)
        back.lineTo(8.5, 20.5)
        back.quadTo(6.9, 20.5, 6.9, 18.8)
        back.lineTo(6.9, 6.2)
        back.quadTo(6.9, 4.5, 8.5, 4.5)
        painter.drawPath(back)

        # 手前側の紙：奥側と間隔を詰め、右下に少しだけ重ねる。
        front = QPainterPath()
        front.moveTo(12.0, 8.0)
        front.lineTo(20.0, 8.0)
        front.quadTo(21.7, 8.0, 21.7, 9.7)
        front.lineTo(21.7, 22.0)
        front.quadTo(21.7, 23.5, 20.0, 23.5)
        front.lineTo(12.0, 23.5)
        front.quadTo(10.4, 23.5, 10.4, 21.8)
        front.lineTo(10.4, 9.7)
        front.quadTo(10.4, 8.0, 12.0, 8.0)
        painter.drawPath(front)
        painter.end()


# =========================================================
# メッセージバブル
# =========================================================

class MessageBubble(QFrame):
    def __init__(self, text):
        super().__init__()

        self.setAttribute(
            Qt.WidgetAttribute.WA_TranslucentBackground
        )

        self.setMaximumWidth(650)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 13, 20, 13)
        layout.setSpacing(0)

        label = QLabel(text)

        label.setWordWrap(True)

        label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        label.setStyleSheet("""
            QLabel {
                color: white;
                background: transparent;
                border: none;
                font-family: "Segoe UI";
                font-size: 13px;
            }
        """)

        layout.addWidget(label)

    def paintEvent(self, event):
        painter = QPainter(self)

        painter.setRenderHint(
            QPainter.RenderHint.Antialiasing
        )

        painter.setPen(Qt.PenStyle.NoPen)

        painter.setBrush(
            QColor("#7C3AED")
        )

        painter.drawRoundedRect(
            self.rect(),
            22,
            22
        )

        painter.end()


class SearchStatusWidget(QFrame):
    """Web検索中の進捗を、AI回答カラムの上に読みやすい大きさで表示する。"""
    def __init__(self, text="Web検索中..."):
        super().__init__()
        self.setObjectName("searchStatusWidget")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setMinimumHeight(44)
        self.setStyleSheet("QFrame#searchStatusWidget { background: transparent; border: none; }")

        outer = QHBoxLayout(self)
        outer.setContentsMargins(10, 7, 10, 7)
        outer.setSpacing(0)
        outer.addStretch(1)

        content = QWidget(self)
        content.setMinimumWidth(560)
        content.setMaximumWidth(820)
        row = QHBoxLayout(content)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)

        self.icon_label = QLabel("⌕")
        self.icon_label.setStyleSheet("color: #b3b3b3; background: transparent; border: none; font-size: 21px;")
        self.icon_label.setFixedWidth(24)
        row.addWidget(self.icon_label, 0, Qt.AlignmentFlag.AlignVCenter)

        self.status_label = QLabel(str(text))
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #b8b8b8; background: transparent; border: none; font-family: \"Segoe UI\"; font-size: 14px; font-weight: 500;")
        row.addWidget(self.status_label, 1, Qt.AlignmentFlag.AlignVCenter)

        outer.addWidget(content, 0, Qt.AlignmentFlag.AlignHCenter)
        outer.addStretch(1)

    def set_status_text(self, text):
        self.status_label.setText(str(text))


class ImagePreviewOverlay(QWidget):
    """新しいウィンドウを作らず、アプリ内に画像プレビューを重ねて表示する。"""

    def __init__(self, image_path, parent):
        super().__init__(parent)
        self.image_path = image_path
        self.original_pixmap = QPixmap(image_path)
        self.setGeometry(parent.rect())
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("background: rgba(0, 0, 0, 190);")
        self.setCursor(Qt.CursorShape.ArrowCursor)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.addStretch(1)

        self.panel = QFrame(self)
        self.panel.setObjectName("previewPanel")
        # 拡大プレビューを見やすくするため、パネルを大きめにする。
        # アプリのウィンドウが小さい場合でも画面内に収まるよう上限も設定する。
        self.panel.setMinimumSize(640, 440)
        self.panel.setMaximumSize(1400, 1000)
        self.panel.setStyleSheet("""
            QFrame#previewPanel { background: #171717; border: 1px solid #353535; border-radius: 16px; }
            QLabel { color: #eeeeee; background: transparent; border: none; }
        """)
        panel_layout = QVBoxLayout(self.panel)
        panel_layout.setContentsMargins(14, 12, 14, 14)
        panel_layout.setSpacing(10)

        top_row = QHBoxLayout()
        name_label = QLabel(Path(image_path).name)
        name_label.setStyleSheet("font-size: 12px;")
        top_row.addWidget(name_label)
        top_row.addStretch(1)
        close_button = QPushButton("×")
        close_button.setFixedSize(28, 28)
        close_button.setCursor(Qt.CursorShape.PointingHandCursor)
        close_button.setStyleSheet("QPushButton { color: #eeeeee; background: #292929; border: none; border-radius: 16px; font-size: 20px; } QPushButton:hover { background: #444444; }")
        close_button.clicked.connect(self.close_overlay)
        top_row.addWidget(close_button)
        panel_layout.addLayout(top_row)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(560, 340)
        self.image_label.setStyleSheet("background: #101010; border-radius: 10px;")
        panel_layout.addWidget(self.image_label, 1)

        outer.addWidget(self.panel, 8, Qt.AlignmentFlag.AlignCenter)
        outer.addStretch(1)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.show()
        self.raise_()
        self._scale_image()

    def _scale_image(self):
        if not self.original_pixmap.isNull() and self.image_label.width() > 0 and self.image_label.height() > 0:
            self.image_label.setPixmap(self.original_pixmap.scaled(
                self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        self.setGeometry(self.parentWidget().rect())
        self._scale_image()
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        if not self.panel.geometry().contains(event.position().toPoint()):
            self.close_overlay()
            event.accept()
            return
        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close_overlay()
            return
        super().keyPressEvent(event)

    def close_overlay(self):
        self.hide()
        self.deleteLater()


class ClickableImageLabel(QLabel):
    """クリックするとアプリ内オーバーレイで画像を拡大するサムネイル。"""

    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("クリックして画像を拡大")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            ImagePreviewOverlay(self.image_path, self.window())
            event.accept()
            return
        super().mousePressEvent(event)


# =========================================================
# チャットメッセージ
# =========================================================

class ChatMessage(QWidget):
    """ユーザーは右寄せバブル、AIは中央の会話カラムに表示する。"""
    def __init__(self, text, attachments=None, role="user"):
        super().__init__()

        self.message_text = text
        self.attachments = attachments or []
        self.role = role
        is_assistant = role == "assistant"

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(0)

        message_column = QVBoxLayout()
        message_column.setContentsMargins(0, 0, 0, 0)
        message_column.setSpacing(8)

        # QLayoutにはsetMinimumWidth/setMinimumSizeがないため、
        # AI回答用のQWidgetにレイアウトを入れて、そのウィジェットの幅を設定する。
        assistant_container = None
        if is_assistant:
            assistant_container = QWidget()
            assistant_container.setMinimumWidth(600)
            assistant_container.setMaximumWidth(780)
            assistant_container.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Preferred
            )
            assistant_container.setLayout(message_column)
            message_column.setAlignment(Qt.AlignmentFlag.AlignLeft)
        else:
            message_column.setAlignment(Qt.AlignmentFlag.AlignRight)

        if self.attachments:
            attachments_frame = QFrame()
            attachments_frame.setMaximumWidth(680)
            attachments_frame.setStyleSheet("""
                QFrame {
                    background-color: #202020;
                    border: 1px solid #393939;
                    border-radius: 12px;
                }
            """)
            attachments_layout = QVBoxLayout(attachments_frame)
            attachments_layout.setContentsMargins(10, 10, 10, 10)
            attachments_layout.setSpacing(8)

            for attachment in self.attachments:
                if isinstance(attachment, str):
                    file_path = attachment
                    name = Path(file_path).name
                    mime_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
                else:
                    file_path = attachment.get("path", "")
                    name = attachment.get("name") or Path(file_path).name
                    mime_type = attachment.get("type") or mimetypes.guess_type(name)[0] or "application/octet-stream"

                if mime_type.startswith("image/") and file_path and Path(file_path).is_file():
                    image_label = ClickableImageLabel(file_path)
                    pixmap = QPixmap(file_path)
                    if not pixmap.isNull():
                        image_label.setPixmap(pixmap.scaled(
                            320, 220,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation
                        ))
                        image_label.setStyleSheet("background: transparent; border: none;")
                        image_label.setToolTip(name)
                        attachments_layout.addWidget(image_label, alignment=Qt.AlignmentFlag.AlignRight)
                    else:
                        self._add_file_label(attachments_layout, name, mime_type)
                else:
                    self._add_file_label(attachments_layout, name, mime_type)

            message_column.addWidget(
                attachments_frame,
                alignment=Qt.AlignmentFlag.AlignLeft if is_assistant else Qt.AlignmentFlag.AlignRight
            )

        if text:
            if is_assistant:
                answer_label = QLabel(text)
                answer_label.setWordWrap(True)
                answer_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                answer_label.setMinimumWidth(560)
                answer_label.setMaximumWidth(780)
                answer_label.setSizePolicy(
                    QSizePolicy.Policy.Expanding,
                    QSizePolicy.Policy.Preferred
                )
                answer_label.setStyleSheet("""
                    QLabel {
                        color: #f2f2f2;
                        background: transparent;
                        border: none;
                        font-family: "Segoe UI";
                        font-size: 14px;
                        padding: 2px 0;
                    }
                """)
                message_column.addWidget(answer_label, alignment=Qt.AlignmentFlag.AlignLeft)

                # AI回答の下に、縦長の紙を詰めて重ねた独自コピーアイコンを表示。
                action_row = QHBoxLayout()
                action_row.setContentsMargins(0, 0, 0, 0)
                action_row.setSpacing(2)

                copy_button = CopyAnswerButton()
                copy_button.clicked.connect(self.copy_message_to_clipboard)
                action_row.addWidget(copy_button, alignment=Qt.AlignmentFlag.AlignLeft)
                action_row.addStretch(1)
                message_column.addLayout(action_row)
            else:
                bubble = MessageBubble(text)
                message_column.addWidget(bubble, alignment=Qt.AlignmentFlag.AlignRight)

                # ユーザーメッセージの下にも同じ独自コピーアイコンを表示。
                # バブルと同じ右端にそろえ、コピー対象はユーザー発言の全文。
                action_row = QHBoxLayout()
                action_row.setContentsMargins(0, 0, 0, 0)
                action_row.setSpacing(2)
                action_row.addStretch(1)

                copy_button = CopyAnswerButton()
                copy_button.setToolTip("メッセージをコピー")
                copy_button.setAccessibleName("ユーザーメッセージをコピー")
                copy_button.clicked.connect(self.copy_message_to_clipboard)
                action_row.addWidget(copy_button, alignment=Qt.AlignmentFlag.AlignRight)
                message_column.addLayout(action_row)

        if is_assistant:
            layout.addStretch(1)
            layout.addWidget(assistant_container, 0, Qt.AlignmentFlag.AlignHCenter)
            layout.addStretch(1)
        else:
            layout.addStretch(1)
            layout.addLayout(message_column)

    def copy_message_to_clipboard(self):
        """このメッセージ全文をOSのクリップボードへコピーする。"""
        QApplication.clipboard().setText(self.message_text)


# =========================================================
# チャット保存
# =========================================================

class ChatStorage:
    def __init__(self):
        DATA_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

    def load(self):
        if not CHAT_FILE.exists():
            return []

        try:
            with open(
                    CHAT_FILE,
                    "r",
                    encoding="utf-8"
            ) as f:
                data = json.load(f)

            if isinstance(data, list):
                return data

        except Exception:
            pass

        return []

    def save(self, chats):
        DATA_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        temp_file = DATA_DIR / "chats.tmp"

        try:
            with open(
                    temp_file,
                    "w",
                    encoding="utf-8"
            ) as f:
                json.dump(
                    chats,
                    f,
                    ensure_ascii=False,
                    indent=4
                )

            temp_file.replace(CHAT_FILE)

        except Exception as e:
            print(
                "チャット保存エラー:",
                e
            )


# =========================================================
# チャット履歴ボタン
# =========================================================

class ChatHistoryButton(QPushButton):
    def __init__(
            self,
            text,
            chat_id,
            parent_window
    ):
        super().__init__(text)

        self.chat_id = chat_id
        self.parent_window = parent_window

        self.setFixedHeight(38)

        self.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding-left: 12px;
                padding-right: 12px;
                color: #dddddd;
                background-color: transparent;
                border: none;
                border-radius: 7px;
                font-size: 13px;
            }

            QPushButton:hover {
                background-color: #242424;
            }
        """)

        self.clicked.connect(
            self.open_chat
        )

    def open_chat(self):
        self.parent_window.load_chat(
            self.chat_id
        )

    def contextMenuEvent(self, event):
        menu = QMenu(self)

        menu.setStyleSheet("""
            QMenu {
                background-color: #171717;
                color: white;
                border: 1px solid #303030;
                padding: 5px;
            }

            QMenu::item {
                padding: 8px 25px;
                border-radius: 5px;
            }

            QMenu::item:selected {
                background-color: #2a2a2a;
            }
        """)

        delete_action = menu.addAction(
            "メッセージを削除"
        )

        action = menu.exec(
            event.globalPos()
        )

        if action == delete_action:
            self.parent_window.show_delete_confirmation(
                self.chat_id
            )


# =========================================================
# 削除確認オーバーレイ
# =========================================================

class DeleteConfirmationOverlay(QWidget):
    def __init__(
            self,
            parent_window,
            chat_title,
            on_delete
    ):
        super().__init__(parent_window)

        self.parent_window = parent_window
        self.on_delete = on_delete

        self.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground,
            True
        )

        # 半透明背景
        self.setStyleSheet("""
            QWidget {
                background-color: rgba(0, 0, 0, 170);
            }
        """)

        # =================================================
        # 確認ダイアログ
        # =================================================

        self.dialog = QFrame(self)

        self.dialog.setFixedWidth(420)

        self.dialog.setStyleSheet("""
            QFrame {
                background-color: #050505;
                border: 1px solid #252525;
                border-radius: 14px;
            }

            QLabel {
                color: white;
                background: transparent;
                border: none;
            }

            QPushButton {
                border: none;
                border-radius: 8px;
                padding: 9px 18px;
                font-size: 13px;
            }
        """)

        layout = QVBoxLayout(
            self.dialog
        )

        layout.setContentsMargins(
            24,
            22,
            24,
            22
        )

        layout.setSpacing(14)

        # =================================================
        # タイトル
        # =================================================

        title = QLabel(
            "メッセージを削除"
        )

        title.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 17px;
                font-weight: bold;
            }
        """)

        # =================================================
        # 説明
        # =================================================

        description = QLabel(
            "このチャットのすべてのメッセージを削除しますか？"
        )

        description.setWordWrap(True)

        description.setStyleSheet("""
            QLabel {
                color: #aaaaaa;
                font-size: 13px;
            }
        """)

        # =================================================
        # チャット名
        # =================================================

        chat_label = QLabel(
            chat_title
        )

        chat_label.setStyleSheet("""
            QLabel {
                color: #dddddd;
                font-size: 13px;
                padding: 8px 10px;
                background-color: #111111;
                border-radius: 7px;
            }
        """)

        # =================================================
        # ボタン
        # =================================================

        button_layout = QHBoxLayout()

        button_layout.addStretch()

        # キャンセル
        cancel_button = QPushButton(
            "キャンセル"
        )

        cancel_button.setStyleSheet("""
            QPushButton {
                color: #dddddd;
                background-color: #202020;
            }

            QPushButton:hover {
                background-color: #2b2b2b;
            }
        """)

        cancel_button.clicked.connect(
            self.close_overlay
        )

        # 削除
        delete_button = QPushButton(
            "削除"
        )

        delete_button.setStyleSheet("""
            QPushButton {
                color: white;
                background-color: #7C3AED;
            }

            QPushButton:hover {
                background-color: #8B5CF6;
            }
        """)

        delete_button.clicked.connect(
            self.delete_messages
        )

        button_layout.addWidget(
            cancel_button
        )

        button_layout.addWidget(
            delete_button
        )

        layout.addWidget(title)

        layout.addWidget(
            description
        )

        layout.addWidget(
            chat_label
        )

        layout.addLayout(
            button_layout
        )

    # =====================================================
    # 表示時
    # =====================================================

    def showEvent(self, event):
        super().showEvent(event)

        self.raise_()

        self.center_dialog()

    # =====================================================
    # リサイズ
    # =====================================================

    def resizeEvent(self, event):
        super().resizeEvent(event)

        self.center_dialog()

    # =====================================================
    # ダイアログ中央配置
    # =====================================================

    def center_dialog(self):
        x = (
                    self.width()
                    - self.dialog.width()
            ) // 2

        y = (
                    self.height()
                    - self.dialog.height()
            ) // 2

        self.dialog.move(
            x,
            y
        )

    # =====================================================
    # 半透明部分をクリックしたら閉じる
    # =====================================================

    def mousePressEvent(self, event):
        click_position = (
            event.position().toPoint()
        )

        # ダイアログの外側
        if not self.dialog.geometry().contains(
                click_position
        ):
            self.close_overlay()

            event.accept()

            return

        # ダイアログ内部
        event.ignore()

    # =====================================================
    # 閉じる
    # =====================================================

    def close_overlay(self):
        self.hide()

        self.deleteLater()

        self.parent_window.delete_overlay = None

    # =====================================================
    # 削除
    # =====================================================

    def delete_messages(self):
        self.on_delete()

        self.close_overlay()


# =========================================================
# チャット一括削除の確認メニュー
# =========================================================
class BulkDeleteConfirmOverlay(QWidget):
    def __init__(self, parent_window):
        super().__init__(parent_window)
        self.parent_window = parent_window
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("QWidget { background-color: rgba(0, 0, 0, 175); }")

        self.dialog = QFrame(self)
        self.dialog.setFixedWidth(430)
        self.dialog.setStyleSheet("""
            QFrame { background: #111111; border: 1px solid #303030; border-radius: 15px; }
            QLabel { background: transparent; border: none; color: #eeeeee; }
            QPushButton { border: none; border-radius: 8px; padding: 10px 17px; font-size: 13px; }
        """)
        layout = QVBoxLayout(self.dialog)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(14)

        title = QLabel("すべてのチャットを削除しますか？")
        title.setStyleSheet("font-size: 17px; font-weight: 700; color: #ffffff;")
        title.setWordWrap(True)
        warning = QLabel("この操作は取り消せません。")
        warning.setStyleSheet("font-size: 13px; font-weight: 700; color: #ff7777;")
        description = QLabel("保存されているすべてのチャットとメッセージが削除されます。")
        description.setWordWrap(True)
        description.setStyleSheet("font-size: 13px; color: #aaaaaa;")

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel = QPushButton("キャンセル")
        cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        cancel.setStyleSheet("QPushButton { color: #dddddd; background: #292929; } QPushButton:hover { background: #383838; }")
        cancel.clicked.connect(self.close_overlay)
        delete = QPushButton("削除")
        delete.setCursor(Qt.CursorShape.PointingHandCursor)
        delete.setStyleSheet("QPushButton { color: white; background: #b42332; } QPushButton:hover { background: #d12f40; }")
        delete.clicked.connect(self.continue_to_code)
        buttons.addWidget(cancel)
        buttons.addWidget(delete)

        layout.addWidget(title)
        layout.addWidget(description)
        layout.addWidget(warning)
        layout.addLayout(buttons)

    def showEvent(self, event):
        super().showEvent(event)
        self.setGeometry(self.parent_window.rect())
        self.center_dialog()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.center_dialog()

    def center_dialog(self):
        if hasattr(self, "dialog"):
            self.dialog.adjustSize()
            self.dialog.move((self.width()-self.dialog.width())//2,
                             (self.height()-self.dialog.height())//2)

    def continue_to_code(self):
        self.close_overlay()
        self.parent_window.show_bulk_delete_code_dialog()

    def close_overlay(self):
        self.hide()
        self.deleteLater()
        if self.parent_window.bulk_delete_overlay is self:
            self.parent_window.bulk_delete_overlay = None


class BulkDeleteCodeOverlay(QWidget):
    REQUIRED_CODE = "AB1C23"

    def __init__(self, parent_window):
        super().__init__(parent_window)
        self.parent_window = parent_window
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("QWidget { background-color: rgba(0, 0, 0, 185); }")

        self.dialog = QFrame(self)
        self.dialog.setFixedWidth(430)
        self.dialog.setStyleSheet("""
            QFrame { background: #111111; border: 1px solid #303030; border-radius: 15px; }
            QLabel { background: transparent; border: none; color: #eeeeee; }
            QPushButton { border: none; border-radius: 8px; padding: 10px 17px; font-size: 13px; }
            QLineEdit { color: #f3f3f3; background: #1d1d1d; border: 1px solid #414141; border-radius: 8px; padding: 11px 12px; selection-background-color: #555555; }
            QLineEdit:focus { border: 1px solid #777777; }
        """)
        layout = QVBoxLayout(self.dialog)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(13)

        title = QLabel("一括削除を確認")
        title.setStyleSheet("font-size: 17px; font-weight: 700; color: #ffffff;")
        instructions = QLabel("この操作を完了させるには AB1C23 と入力してください")
        instructions.setWordWrap(True)
        instructions.setStyleSheet("font-size: 13px; color: #c5c5c5;")

        self.code_input = QLineEdit()
        self.code_input.setPlaceholderText("確認コードを入力")
        self.code_input.setMaxLength(32)
        self.code_input.returnPressed.connect(self.try_delete)

        self.error_label = QLabel("")
        self.error_label.setStyleSheet("font-size: 12px; color: #ff7777;")
        self.error_label.hide()

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel = QPushButton("キャンセル")
        cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        cancel.setStyleSheet("QPushButton { color: #dddddd; background: #292929; } QPushButton:hover { background: #383838; }")
        cancel.clicked.connect(self.close_overlay)
        delete = QPushButton("削除")
        delete.setCursor(Qt.CursorShape.PointingHandCursor)
        delete.setStyleSheet("QPushButton { color: white; background: #b42332; } QPushButton:hover { background: #d12f40; }")
        delete.clicked.connect(self.try_delete)
        buttons.addWidget(cancel)
        buttons.addWidget(delete)

        layout.addWidget(title)
        layout.addWidget(instructions)
        layout.addWidget(self.code_input)
        layout.addWidget(self.error_label)
        layout.addLayout(buttons)

    def showEvent(self, event):
        super().showEvent(event)
        self.setGeometry(self.parent_window.rect())
        self.center_dialog()
        self.code_input.setFocus()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.center_dialog()

    def center_dialog(self):
        if hasattr(self, "dialog"):
            self.dialog.adjustSize()
            self.dialog.move((self.width()-self.dialog.width())//2,
                             (self.height()-self.dialog.height())//2)

    def try_delete(self):
        if self.code_input.text() != self.REQUIRED_CODE:
            self.error_label.setText("コードが違います")
            self.error_label.show()
            self.dialog.adjustSize()
            self.center_dialog()
            return
        self.parent_window.delete_all_chats()
        self.close_overlay()

    def close_overlay(self):
        self.hide()
        self.deleteLater()
        if self.parent_window.bulk_delete_overlay is self:
            self.parent_window.bulk_delete_overlay = None


# =========================================================
# 記憶削除の確認画面
# =========================================================
class MemoryDeleteConfirmOverlay(QWidget):
    """記憶削除の確認を行い、続行するとコード入力画面へ進む。"""
    def __init__(self, parent_window):
        super().__init__(parent_window)
        self.parent_window = parent_window
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("QWidget { background-color: rgba(0, 0, 0, 175); }")

        self.dialog = QFrame(self)
        self.dialog.setFixedWidth(430)
        self.dialog.setStyleSheet("""
            QFrame { background: #111111; border: 1px solid #303030; border-radius: 15px; }
            QLabel { background: transparent; border: none; color: #eeeeee; }
            QPushButton { border: none; border-radius: 8px; padding: 10px 17px; font-size: 13px; }
        """)
        layout = QVBoxLayout(self.dialog)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(14)

        title = QLabel("記憶を削除しますか？")
        title.setStyleSheet("font-size: 17px; font-weight: 700; color: #ffffff;")
        title.setWordWrap(True)
        description = QLabel("保存されている会話の記憶とプロフィール情報を削除します。この操作は取り消せません。チャット履歴や学習済み知識は削除されません。")
        description.setWordWrap(True)
        description.setStyleSheet("font-size: 13px; color: #b8b8b8;")
        warning = QLabel("この操作を続けるには、次の画面で確認コードが必要です。")
        warning.setWordWrap(True)
        warning.setStyleSheet("font-size: 12px; color: #ff7777;")

        buttons = QHBoxLayout()
        buttons.addStretch()
        no_button = QPushButton("いいえ")
        no_button.setCursor(Qt.CursorShape.PointingHandCursor)
        no_button.setStyleSheet("QPushButton { color: #dddddd; background: #292929; } QPushButton:hover { background: #383838; }")
        no_button.clicked.connect(self.close_overlay)
        delete_button = QPushButton("削除")
        delete_button.setCursor(Qt.CursorShape.PointingHandCursor)
        delete_button.setStyleSheet("QPushButton { color: white; background: #b42332; } QPushButton:hover { background: #d12f40; }")
        delete_button.clicked.connect(self.continue_to_code)
        buttons.addWidget(no_button)
        buttons.addWidget(delete_button)

        layout.addWidget(title)
        layout.addWidget(description)
        layout.addWidget(warning)
        layout.addLayout(buttons)

    def showEvent(self, event):
        super().showEvent(event)
        self.setGeometry(self.parent_window.rect())
        self.center_dialog()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.center_dialog()

    def center_dialog(self):
        if hasattr(self, "dialog"):
            self.dialog.adjustSize()
            self.dialog.move((self.width() - self.dialog.width()) // 2,
                             (self.height() - self.dialog.height()) // 2)

    def continue_to_code(self):
        self.close_overlay()
        self.parent_window.show_memory_delete_code_dialog()

    def mousePressEvent(self, event):
        if not self.dialog.geometry().contains(event.position().toPoint()):
            self.close_overlay()
            event.accept()
            return
        super().mousePressEvent(event)

    def close_overlay(self):
        self.hide()
        self.deleteLater()
        if self.parent_window.memory_delete_overlay is self:
            self.parent_window.memory_delete_overlay = None


# =========================================================
# 記憶削除の確認コード入力画面
# =========================================================
class MemoryDeleteCodeOverlay(QWidget):
    REQUIRED_CODE = "AB1C23"

    def __init__(self, parent_window):
        super().__init__(parent_window)
        self.parent_window = parent_window
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("QWidget { background-color: rgba(0, 0, 0, 185); }")

        self.dialog = QFrame(self)
        self.dialog.setFixedWidth(430)
        self.dialog.setStyleSheet("""
            QFrame { background: #111111; border: 1px solid #303030; border-radius: 15px; }
            QLabel { background: transparent; border: none; color: #eeeeee; }
            QPushButton { border: none; border-radius: 8px; padding: 10px 17px; font-size: 13px; }
            QLineEdit { color: #f3f3f3; background: #1d1d1d; border: 1px solid #414141; border-radius: 8px; padding: 11px 12px; selection-background-color: #555555; }
            QLineEdit:focus { border: 1px solid #777777; }
        """)
        layout = QVBoxLayout(self.dialog)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(13)

        title = QLabel("記憶の削除を確認")
        title.setStyleSheet("font-size: 17px; font-weight: 700; color: #ffffff;")
        instructions = QLabel("記憶を削除するには、確認コード AB1C23 を入力してください。")
        instructions.setWordWrap(True)
        instructions.setStyleSheet("font-size: 13px; color: #c5c5c5;")

        self.code_input = QLineEdit()
        self.code_input.setPlaceholderText("確認コードを入力")
        self.code_input.setMaxLength(32)
        self.code_input.returnPressed.connect(self.try_delete)

        self.error_label = QLabel("")
        self.error_label.setStyleSheet("font-size: 12px; color: #ff7777;")
        self.error_label.hide()

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel = QPushButton("いいえ")
        cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        cancel.setStyleSheet("QPushButton { color: #dddddd; background: #292929; } QPushButton:hover { background: #383838; }")
        cancel.clicked.connect(self.close_overlay)
        delete = QPushButton("削除")
        delete.setCursor(Qt.CursorShape.PointingHandCursor)
        delete.setStyleSheet("QPushButton { color: white; background: #b42332; } QPushButton:hover { background: #d12f40; }")
        delete.clicked.connect(self.try_delete)
        buttons.addWidget(cancel)
        buttons.addWidget(delete)

        layout.addWidget(title)
        layout.addWidget(instructions)
        layout.addWidget(self.code_input)
        layout.addWidget(self.error_label)
        layout.addLayout(buttons)

    def showEvent(self, event):
        super().showEvent(event)
        self.setGeometry(self.parent_window.rect())
        self.center_dialog()
        self.code_input.setFocus()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.center_dialog()

    def center_dialog(self):
        if hasattr(self, "dialog"):
            self.dialog.adjustSize()
            self.dialog.move((self.width() - self.dialog.width()) // 2,
                             (self.height() - self.dialog.height()) // 2)

    def try_delete(self):
        if self.code_input.text().strip() != self.REQUIRED_CODE:
            self.error_label.setText("コードが違います")
            self.error_label.show()
            self.dialog.adjustSize()
            self.center_dialog()
            self.code_input.setFocus()
            self.code_input.selectAll()
            return

        try:
            self.parent_window.delete_ai_memory()
        except Exception as exc:
            self.error_label.setText("記憶を削除できませんでした。もう一度試してください。")
            self.error_label.show()
            self.dialog.adjustSize()
            self.center_dialog()
            print("AngelWhite: 記憶削除エラー:", repr(exc), flush=True)
            return

        self.close_overlay()

    def mousePressEvent(self, event):
        if not self.dialog.geometry().contains(event.position().toPoint()):
            self.close_overlay()
            event.accept()
            return
        super().mousePressEvent(event)

    def close_overlay(self):
        self.hide()
        self.deleteLater()
        if self.parent_window.memory_delete_overlay is self:
            self.parent_window.memory_delete_overlay = None


# =========================================================
# AI回答をバックグラウンドで実行（Web検索中もUIを固めない）
# =========================================================

class AIResponseThread(QThread):
    response_ready = Signal(str, str, str)  # chat_id, original question, answer
    search_progress = Signal(str, str)      # chat_id, progress/status text

    def __init__(self, engine, chat_id, question, mode, context):
        super().__init__()
        self.engine = engine
        self.chat_id = str(chat_id)
        self.question = question
        self.mode = mode
        self.context = context

    def run(self):
        # どんな場合でもUIへ結果を返す。例外の詳細はIntelliJのRunへ出力する。
        try:
            answer = self.engine.respond(
                self.question,
                mode=self.mode,
                context=self.context,
                progress_callback=lambda message: self.search_progress.emit(self.chat_id, str(message))
            )

            if not isinstance(answer, str) or not answer.strip():
                answer = (
                    "うまく返答を作れなかったよ。"
                    "質問を少し変えて、もう一度送ってみてね。"
                )

        except Exception as exc:
            import traceback

            print("\n===== AngelWhite AIエラー =====", flush=True)
            traceback.print_exc()
            print("エラーの詳細:", repr(exc), flush=True)
            print("================================\n", flush=True)

            answer = (
                "回答の作成中に問題が発生したよ。\n"
                f"エラーの種類: {type(exc).__name__}\n"
                f"詳細: {exc}\n"
                "詳しい原因はIntelliJのRunを確認してね。"
            )

        try:
            self.response_ready.emit(
                self.chat_id,
                self.question,
                answer
            )
        except Exception as exc:
            import traceback
            print(
                "AngelWhite: 回答通知の送信に失敗:",
                repr(exc),
                flush=True
            )
            traceback.print_exc()


# =========================================================
# 記憶設定用オン・オフスイッチ
# =========================================================

class MemoryToggle(QWidget):
    """つまみが移動する、クリック可能な独自トグルスイッチ。"""
    toggled = Signal(bool)

    def __init__(self, checked=False, parent=None):
        super().__init__(parent)
        self._checked = bool(checked)
        self.setFixedSize(46, 27)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("会話を記憶する")

    def isChecked(self):
        return self._checked

    def setChecked(self, checked):
        checked = bool(checked)
        if self._checked != checked:
            self._checked = checked
            self.update()
            self.toggled.emit(self._checked)
        else:
            self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setChecked(not self._checked)
            event.accept()
            return
        super().mousePressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        track = self.rect().adjusted(1, 1, -1, -1)
        track_color = QColor("#8b5cf6" if self._checked else "#454545")
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(track_color)
        painter.drawRoundedRect(track, 15, 15)
        knob_d = 20
        knob_x = self.width() - knob_d - 4 if self._checked else 4
        knob_y = (self.height() - knob_d) // 2
        painter.setBrush(QColor("#ffffff"))
        painter.drawEllipse(knob_x, knob_y, knob_d, knob_d)


# =========================================================
# 記憶機能ポップアップ（土台）
# =========================================================

class MemorySettingsOverlay(QWidget):
    """記憶機能の独自UI。今後ここに記憶設定を追加する。"""
    def __init__(self, parent_window):
        super().__init__(parent_window)
        self.parent_window = parent_window
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("QWidget#memoryOverlay { background-color: transparent; }")
        self.setObjectName("memoryOverlay")

        self.dialog = QFrame(self)
        self.dialog.setObjectName("memoryDialog")
        self.dialog.setFixedWidth(360)
        self.dialog.setStyleSheet("""
            QFrame#memoryDialog {
                background-color: #171717;
                border: 1px solid #353535;
                border-radius: 18px;
            }
            QLabel { color: #f1f1f1; background: transparent; border: none; }
            QLabel#memorySubtitle { color: #a3a3a3; font-size: 11px; }
            QFrame#memoryDivider { background-color: #303030; border: none; max-height: 1px; }
            QPushButton {
                color: #f2f2f2; background-color: #292929; border: 1px solid #3a3a3a;
                border-radius: 9px; padding: 9px 16px; font-size: 13px;
            }
            QPushButton:hover { background-color: #383838; }
            QPushButton#memoryClose { background-color: #8b5cf6; border: none; font-weight: 600; }
            QPushButton#memoryClose:hover { background-color: #9b75fa; }
        """)
        layout = QVBoxLayout(self.dialog)
        layout.setContentsMargins(17, 15, 17, 14)
        layout.setSpacing(9)

        heading_row = QHBoxLayout()
        heading = QLabel("記憶機能")
        heading.setStyleSheet("font-size: 18px; font-weight: 650;")
        heading_row.addWidget(heading)
        heading_row.addStretch(1)
        close_button = QPushButton("×")
        close_button.setFixedSize(32, 32)
        close_button.setStyleSheet("QPushButton { font-size: 20px; padding: 0; border-radius: 9px; }")
        close_button.setCursor(Qt.CursorShape.PointingHandCursor)
        close_button.clicked.connect(self.close_overlay)
        heading_row.addWidget(close_button)
        layout.addLayout(heading_row)

        subtitle = QLabel("AngelWhite の記憶を管理する画面")
        subtitle.setObjectName("memorySubtitle")
        layout.addWidget(subtitle)

        divider = QFrame()
        divider.setObjectName("memoryDivider")
        divider.setFixedHeight(1)
        layout.addWidget(divider)

        setting_card = QFrame()
        setting_card.setStyleSheet("""
            QFrame {
                background-color: #202020;
                border: 1px solid #343434;
                border-radius: 14px;
            }
        """)
        setting_layout = QVBoxLayout(setting_card)
        setting_layout.setContentsMargins(12, 11, 12, 11)
        setting_layout.setSpacing(7)

        setting_row = QHBoxLayout()
        setting_row.setSpacing(8)
        setting_title = QLabel("会話を記憶する")
        setting_title.setStyleSheet("font-size: 13px; font-weight: 600; border: none;")
        setting_row.addWidget(setting_title, 1)

        # 独自描画のスイッチなので、つまみが実際に左右へ移動する
        self.memory_toggle = MemoryToggle(self.parent_window.memory_enabled)
        self.memory_toggle.toggled.connect(self.parent_window.set_memory_enabled)
        setting_row.addWidget(self.memory_toggle, 0, Qt.AlignmentFlag.AlignVCenter)
        setting_layout.addLayout(setting_row)

        explanation = QLabel("この設定をオンにするとあなたの会話がAIに記憶されて学習されます")
        explanation.setWordWrap(True)
        explanation.setStyleSheet("color: #999999; font-size: 10px; font-weight: 300; border: none;")
        setting_layout.addWidget(explanation)

        # 記憶削除ボタン（現時点では見た目のみ。機能は未接続）
        delete_memory_button = QPushButton("記憶を削除")
        delete_memory_button.setCursor(Qt.CursorShape.PointingHandCursor)
        delete_memory_button.setFixedHeight(32)
        delete_memory_button.setStyleSheet("""
            QPushButton {
                color: #ff5c64;
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 8px;
                padding: 5px 8px;
                text-align: left;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                color: #ff777d;
                background-color: #302020;
            }
            QPushButton:pressed {
                background-color: #3a2224;
            }
        """)
        # 確認画面を開く。削除処理は確認コードが正しい場合だけ実行する。
        delete_memory_button.clicked.connect(
            self.parent_window.show_memory_delete_confirmation
        )
        setting_layout.addWidget(delete_memory_button, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(setting_card)

        footer = QHBoxLayout()
        footer.addStretch(1)
        done_button = QPushButton("閉じる")
        done_button.setObjectName("memoryClose")
        done_button.setStyleSheet("QPushButton#memoryClose { padding: 6px 12px; font-size: 12px; }")
        done_button.setCursor(Qt.CursorShape.PointingHandCursor)
        done_button.clicked.connect(self.close_overlay)
        footer.addWidget(done_button)
        layout.addLayout(footer)

    def showEvent(self, event):
        super().showEvent(event)
        self._center_dialog()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._center_dialog()

    def _center_dialog(self):
        # 中央ではなく、サイドバーの設定ボタンの右側に小さく表示する
        if not hasattr(self, "dialog"):
            return
        self.dialog.adjustSize()
        button = getattr(self.parent_window, "settings_button", None)
        if button is not None:
            button_pos = button.mapTo(self, QPoint(0, 0))
            x = button_pos.x() + button.width() + 12
            y = button_pos.y() - self.dialog.height() + button.height()
        else:
            x = 20
            y = 20
        # ウィンドウ内に収まるように位置を調整
        x = max(8, min(x, self.width() - self.dialog.width() - 8))
        y = max(8, min(y, self.height() - self.dialog.height() - 8))
        self.dialog.move(x, y)

    def mousePressEvent(self, event):
        if not self.dialog.geometry().contains(event.pos()):
            self.close_overlay()
        else:
            super().mousePressEvent(event)

    def close_overlay(self):
        self.hide()
        self.deleteLater()
        if getattr(self.parent_window, "memory_overlay", None) is self:
            self.parent_window.memory_overlay = None


# =========================================================
# メインウィンドウ
# =========================================================

class AngelWhiteWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(
            "AngelWhite"
        )

        self.resize(
            1100,
            700
        )

        self.setMinimumSize(
            800,
            500
        )

        # =================================================
        # チャットデータ
        # =================================================

        self.storage = ChatStorage()
        self.ai_engine = JapaneseAI(DATA_DIR)
        self._answer_threads = []
        self._answer_pending = False
        self._search_status_widget = None
        self._search_status_chat_id = None

        self.chats = []

        self.current_chat_id = None

        self.chat_counter = 0

        self.sidebar_visible = True

        self.delete_overlay = None
        self.bulk_delete_overlay = None
        self.memory_overlay = None
        self.memory_delete_overlay = None
        self.memory_enabled = self.load_memory_enabled()
        self.ai_engine.set_memory_enabled(self.memory_enabled)
        self.plus_popup = None
        self.selected_files = []

        # =================================================
        # メイン背景
        # =================================================

        self.setStyleSheet("""
            QMainWindow {
                background-color: #0d0d0d;
            }

            QWidget {
                font-family: "Segoe UI";
            }
        """)

        # =================================================
        # 全体
        # =================================================

        self.central = QWidget()

        self.setCentralWidget(
            self.central
        )

        main_layout = QHBoxLayout(
            self.central
        )

        main_layout.setContentsMargins(
            0,
            0,
            0,
            0
        )

        main_layout.setSpacing(0)

        # =================================================
        # サイドバー
        # =================================================

        self.sidebar = QFrame()

        self.sidebar.setFixedWidth(
            250
        )

        self.sidebar.setStyleSheet("""
            QFrame {
                background-color: #171717;
            }
        """)

        sidebar_layout = QVBoxLayout(
            self.sidebar
        )

        sidebar_layout.setContentsMargins(
            12,
            12,
            12,
            12
        )

        sidebar_layout.setSpacing(10)

        # =================================================
        # AngelWhiteタイトル
        # =================================================

        top_layout = QHBoxLayout()

        title = QLabel(
            "AngelWhite"
        )

        title.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 17px;
                font-weight: bold;
                padding-left: 4px;
            }
        """)

        top_layout.addWidget(
            title
        )

        top_layout.addStretch()

        # =================================================
        # サイドバー閉じるボタン
        # =================================================

        collapse_button = QPushButton(
            "‹"
        )

        collapse_button.setFixedSize(
            34,
            34
        )

        collapse_button.setStyleSheet("""
            QPushButton {
                color: #aaaaaa;
                background-color: transparent;
                border: none;
                border-radius: 7px;
                font-size: 25px;
            }

            QPushButton:hover {
                color: white;
                background-color: #242424;
            }
        """)

        collapse_button.clicked.connect(
            self.hide_sidebar
        )

        top_layout.addWidget(
            collapse_button
        )

        sidebar_layout.addLayout(
            top_layout
        )

        # =================================================
        # 新しいチャット
        # =================================================

        new_chat_button = QPushButton(
            "+  新しいチャット"
        )

        new_chat_button.setFixedHeight(
            42
        )

        new_chat_button.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding-left: 13px;
                color: #eeeeee;
                background-color: #242424;
                border: none;
                border-radius: 8px;
                font-size: 13px;
            }

            QPushButton:hover {
                background-color: #2d2d2d;
            }
        """)

        new_chat_button.clicked.connect(
            self.create_new_chat
        )

        sidebar_layout.addWidget(
            new_chat_button
        )

        # =================================================
        # チャットラベル
        # =================================================

        history_label = QLabel(
            "チャット"
        )

        history_label.setStyleSheet("""
            QLabel {
                color: #777777;
                font-size: 11px;
                padding-left: 5px;
                padding-top: 8px;
            }
        """)

        sidebar_layout.addWidget(
            history_label
        )

        # =================================================
        # 履歴スクロール
        # =================================================

        self.history_scroll = QScrollArea()

        self.history_scroll.setWidgetResizable(
            True
        )

        self.history_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        self.history_scroll.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }

            QScrollBar:vertical {
                background: transparent;
                width: 7px;
            }

            QScrollBar::handle:vertical {
                background: #333333;
                border-radius: 3px;
            }

            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        self.history_widget = QWidget()

        self.history_layout = QVBoxLayout(
            self.history_widget
        )

        self.history_layout.setContentsMargins(
            0,
            0,
            0,
            0
        )

        self.history_layout.setSpacing(
            3
        )

        self.history_layout.addStretch()

        self.history_scroll.setWidget(
            self.history_widget
        )

        sidebar_layout.addWidget(
            self.history_scroll,
            1
        )

        # =================================================
        # サイドバー左下の設定アイコン
        # =================================================

        self.settings_button = SettingsIconButton()
        self.settings_button.clicked.connect(self.toggle_settings_popup)

        sidebar_layout.addWidget(
            self.settings_button,
            0,
            Qt.AlignmentFlag.AlignLeft
        )

        self.settings_popup = None

        main_layout.addWidget(
            self.sidebar
        )

        # =================================================
        # メインコンテンツ
        # =================================================

        self.content_widget = QWidget()

        self.content_widget.setStyleSheet("""
            QWidget {
                background-color: #0d0d0d;
            }
        """)

        content_layout = QVBoxLayout(
            self.content_widget
        )

        content_layout.setContentsMargins(
            0,
            0,
            0,
            0
        )

        content_layout.setSpacing(0)

        # =================================================
        # チャット表示エリア
        # =================================================

        self.chat_scroll = QScrollArea()

        self.chat_scroll.setWidgetResizable(
            True
        )

        self.chat_scroll.setFrameShape(
            QFrame.Shape.NoFrame
        )

        self.chat_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        self.chat_scroll.setStyleSheet("""
            QScrollArea {
                background-color: #0d0d0d;
                border: none;
            }

            QScrollBar:vertical {
                background: transparent;
                width: 8px;
            }

            QScrollBar::handle:vertical {
                background: #333333;
                border-radius: 4px;
            }

            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        self.chat_widget = QWidget()

        self.chat_widget.setStyleSheet("""
            QWidget {
                background-color: #0d0d0d;
            }
        """)

        self.chat_layout = QVBoxLayout(
            self.chat_widget
        )

        self.chat_layout.setContentsMargins(
            20,
            20,
            20,
            20
        )

        self.chat_layout.setSpacing(
            0
        )

        # チャットがないときに中央表示する案内
        self.empty_state_widget = QLabel(
            "チャットはまだありません"
        )
        self.empty_state_widget.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        self.empty_state_widget.setMinimumHeight(120)
        self.empty_state_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        self.empty_state_widget.setStyleSheet("""
            QLabel {
                color: #a0a0a0;
                background: transparent;
                border: none;
                font-family: "Segoe UI";
                font-size: 24px;
                font-weight: 500;
            }
        """)

        self.empty_state_active = False
        self.show_empty_state()

        self.chat_scroll.setWidget(
            self.chat_widget
        )

        content_layout.addWidget(
            self.chat_scroll
        )

        # =================================================
        # 入力エリア（選択ファイル表示 + 入力欄）
        # =================================================

        input_area = QWidget()
        input_area_layout = QVBoxLayout(input_area)
        input_area_layout.setContentsMargins(0, 0, 0, 0)
        input_area_layout.setSpacing(0)

        self.attachment_row = QWidget()
        self.attachment_row_layout = QHBoxLayout(self.attachment_row)
        self.attachment_row_layout.setContentsMargins(20, 4, 20, 0)
        self.attachment_row_layout.setSpacing(8)
        self.attachment_row.hide()
        input_area_layout.addWidget(self.attachment_row)

        input_container = QWidget()

        input_layout = QHBoxLayout(
            input_container
        )

        input_layout.setContentsMargins(
            20,
            12,
            20,
            18
        )

        input_layout.setSpacing(
            8
        )

        # =================================================
        # 入力欄
        # =================================================

        self.input = QLineEdit()

        self.input.setPlaceholderText(
            "メッセージを入力..."
        )

        self.input.setFixedHeight(
            48
        )

        self.input.setStyleSheet("""
            QLineEdit {
                color: white;
                background-color: #191919;
                border: 1px solid #292929;
                border-radius: 14px;
                padding-left: 48px;
                padding-right: 15px;
                font-size: 14px;
            }

            QLineEdit:focus {
                border: 1px solid #3a3a3a;
            }
        """)

        # =================================================
        # ＋ボタン
        # =================================================

        self.plus_button = QPushButton(
            "+"
        )

        self.plus_button.setFixedSize(
            36,
            36
        )

        self.plus_button.setStyleSheet("""
            QPushButton {
                color: #aaaaaa;
                background-color: transparent;
                border: none;
                font-size: 25px;
                padding-bottom: 5px;
            }

            QPushButton:hover {
                color: white;
            }
        """)

        self.plus_button.clicked.connect(
            self.plus_clicked
        )

        input_wrapper = QWidget()

        input_wrapper_layout = QHBoxLayout(
            input_wrapper
        )

        input_wrapper_layout.setContentsMargins(
            0,
            0,
            0,
            0
        )

        input_wrapper_layout.setSpacing(
            0
        )

        input_wrapper_layout.addWidget(
            self.input
        )

        self.plus_button.setParent(
            input_wrapper
        )

        self.plus_button.move(
            8,
            6
        )

        self.plus_button.raise_()

        input_layout.addWidget(
            input_wrapper,
            1
        )

        # =================================================
        # モード選択ボタン（独自の上向きポップアップ）
        # =================================================

        self.response_mode = "一般モード"
        self.mode_popup = None
        self.mode_button = QPushButton("一般モード  ˄")
        self.mode_button.setFixedSize(112, 48)
        self.mode_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.mode_button.setStyleSheet("""
            QPushButton {
                color: #dddddd;
                background-color: #202020;
                border: 1px solid #383838;
                border-radius: 12px;
                padding: 0 8px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #2b2b2b;
                border-color: #505050;
            }
            QPushButton:pressed { background-color: #303030; }
        """)
        self.mode_button.clicked.connect(self.toggle_mode_popup)
        input_layout.addWidget(self.mode_button)

        # =================================================
        # 送信ボタン
        # =================================================

        send_button = QPushButton(
            "送信"
        )

        send_button.setFixedSize(
            70,
            48
        )

        send_button.setStyleSheet("""
            QPushButton {
                color: white;
                background-color: #7C3AED;
                border: none;
                border-radius: 12px;
                font-size: 13px;
            }

            QPushButton:hover {
                background-color: #8B5CF6;
            }

            QPushButton:pressed {
                background-color: #6D28D9;
            }
        """)

        send_button.clicked.connect(
            self.send_message
        )

        input_layout.addWidget(
            send_button
        )

        input_area_layout.addWidget(input_container)
        content_layout.addWidget(input_area)

        main_layout.addWidget(
            self.content_widget
        )

        # =================================================
        # サイドバー再表示ボタン
        # =================================================

        self.open_sidebar_button = QPushButton(
            "›",
            self.content_widget
        )

        self.open_sidebar_button.setFixedSize(
            34,
            34
        )

        self.open_sidebar_button.setStyleSheet("""
            QPushButton {
                color: #aaaaaa;
                background-color: #171717;
                border: none;
                border-radius: 7px;
                font-size: 25px;
            }

            QPushButton:hover {
                color: white;
                background-color: #242424;
            }
        """)

        self.open_sidebar_button.clicked.connect(
            self.show_sidebar
        )

        self.open_sidebar_button.hide()

        # =================================================
        # Enterで送信
        # =================================================

        self.input.returnPressed.connect(
            self.send_message
        )

        # =================================================
        # 保存済みチャット読み込み
        # =================================================

        self.load_saved_chats()

    # =====================================================
    # サイドバーを隠す
    # =====================================================

    def toggle_settings_popup(self):
        """歯車の近くに、独自デザインの角丸ポップアップを表示する。"""
        if self.settings_popup is not None and self.settings_popup.isVisible():
            self.settings_popup.close()
            return

        popup = QFrame(self.sidebar, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        popup.setObjectName("settingsPopup")
        popup.setFixedWidth(218)
        popup.setStyleSheet("""
            QFrame#settingsPopup {
                background-color: #232323;
                border: 1px solid #3a3a3a;
                border-radius: 13px;
            }
            QLabel#popupTitle {
                color: #8f8f8f;
                font-size: 11px;
                font-weight: 600;
                padding: 2px 8px 5px 8px;
                border: none;
            }
            QPushButton#popupItem {
                color: #eeeeee;
                background: transparent;
                border: none;
                border-radius: 8px;
                text-align: left;
                padding: 9px 10px;
                font-size: 13px;
            }
            QPushButton#popupItem:hover { background-color: #353535; }
            QFrame#popupDivider {
                background-color: #3a3a3a;
                border: none;
                max-height: 1px;
            }
        """)

        layout = QVBoxLayout(popup)
        layout.setContentsMargins(8, 9, 8, 8)
        layout.setSpacing(3)

        title = QLabel("AngelWhite")
        title.setObjectName("popupTitle")
        layout.addWidget(title)

        # 記憶機能：専用の独自UIポップアップを開く
        memory_item = QPushButton("記憶機能")
        memory_item.setObjectName("popupItem")
        memory_item.setCursor(Qt.CursorShape.PointingHandCursor)
        memory_item.clicked.connect(popup.close)
        memory_item.clicked.connect(self.show_memory_settings)
        layout.addWidget(memory_item)

        divider = QFrame()
        divider.setObjectName("popupDivider")
        divider.setFixedHeight(1)
        layout.addWidget(divider)

        bulk_delete_item = QPushButton("一括削除")
        bulk_delete_item.setObjectName("popupItem")
        bulk_delete_item.setCursor(Qt.CursorShape.PointingHandCursor)
        bulk_delete_item.setStyleSheet("QPushButton#popupItem { color: #ff8585; }")
        bulk_delete_item.clicked.connect(popup.close)
        bulk_delete_item.clicked.connect(self.show_bulk_delete_confirmation)
        layout.addWidget(bulk_delete_item)

        self.settings_popup = popup
        popup.adjustSize()
        # 歯車の少し右側に配置し、サイドバー下部から自然に開く
        button_pos = self.settings_button.mapToGlobal(QPoint(0, 0))
        popup.move(button_pos.x() + self.settings_button.width() + 8,
                   button_pos.y() - popup.sizeHint().height() + self.settings_button.height())
        popup.show()

    def load_memory_enabled(self):
        """記憶機能のオン・オフ設定をファイルから読み込む。"""
        try:
            if MEMORY_SETTINGS_FILE.exists():
                with MEMORY_SETTINGS_FILE.open("r", encoding="utf-8") as file:
                    settings = json.load(file)
                return bool(settings.get("remember_conversations", False))
        except (OSError, json.JSONDecodeError, TypeError):
            pass
        return False

    def set_memory_enabled(self, enabled):
        """記憶機能の設定をメモリとファイルに保存する。"""
        self.memory_enabled = bool(enabled)
        self.ai_engine.set_memory_enabled(self.memory_enabled)
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            with MEMORY_SETTINGS_FILE.open("w", encoding="utf-8") as file:
                json.dump(
                    {"remember_conversations": self.memory_enabled},
                    file,
                    ensure_ascii=False,
                    indent=2,
                )
        except OSError as exc:
            print("AngelWhite: 記憶設定を保存できませんでした:", repr(exc))

    def show_memory_settings(self):
        """記憶機能の独自UIポップアップを表示する。"""
        if self.memory_overlay is not None:
            self.memory_overlay.close_overlay()
        self.memory_overlay = MemorySettingsOverlay(self)
        self.memory_overlay.setGeometry(self.rect())
        self.memory_overlay.show()
        self.memory_overlay.raise_()

    def show_memory_delete_confirmation(self):
        """記憶削除の確認画面を開く。"""
        if self.memory_delete_overlay is not None:
            self.memory_delete_overlay.close_overlay()
        self.memory_delete_overlay = MemoryDeleteConfirmOverlay(self)
        self.memory_delete_overlay.setGeometry(self.rect())
        self.memory_delete_overlay.show()
        self.memory_delete_overlay.raise_()

    def show_memory_delete_code_dialog(self):
        """記憶削除の確認コード入力画面を開く。"""
        if self.memory_delete_overlay is not None:
            self.memory_delete_overlay.close_overlay()
        self.memory_delete_overlay = MemoryDeleteCodeOverlay(self)
        self.memory_delete_overlay.setGeometry(self.rect())
        self.memory_delete_overlay.show()
        self.memory_delete_overlay.raise_()

    def delete_ai_memory(self):
        """会話メモリ（turns/facts）のみ消去し、チャット履歴や知識DBは残す。"""
        memory = self.ai_engine.context_memory
        memory["turns"] = []
        memory["facts"] = []
        self.ai_engine._save_memory()

    def show_bulk_delete_confirmation(self):
        if self.bulk_delete_overlay is not None:
            self.bulk_delete_overlay.close_overlay()
        self.bulk_delete_overlay = BulkDeleteConfirmOverlay(self)
        self.bulk_delete_overlay.setGeometry(self.rect())
        self.bulk_delete_overlay.show()
        self.bulk_delete_overlay.raise_()

    def show_bulk_delete_code_dialog(self):
        if self.bulk_delete_overlay is not None:
            self.bulk_delete_overlay.close_overlay()
        self.bulk_delete_overlay = BulkDeleteCodeOverlay(self)
        self.bulk_delete_overlay.setGeometry(self.rect())
        self.bulk_delete_overlay.show()
        self.bulk_delete_overlay.raise_()

    def delete_all_chats(self):
        # 履歴ボタンを取り除き、チャットデータを空にする
        for chat in self.chats:
            button = chat.get("button")
            if button is not None:
                button.deleteLater()

        self.chats = []
        self.current_chat_id = None
        self.selected_files = []
        self.refresh_attachment_chips()
        self.clear_chat_area()
        self.save_chats()

    def hide_sidebar(self):
        self.sidebar.hide()

        self.sidebar_visible = False

        self.open_sidebar_button.move(
            12,
            12
        )

        self.open_sidebar_button.raise_()

        self.open_sidebar_button.show()

    # =====================================================
    # サイドバーを表示
    # =====================================================

    def show_sidebar(self):
        self.sidebar.show()

        self.sidebar_visible = True

        self.open_sidebar_button.hide()

    # =====================================================
    # モード選択ポップアップ
    # =====================================================

    def toggle_mode_popup(self):
        if self.mode_popup is not None and self.mode_popup.isVisible():
            self.mode_popup.close()
            self.mode_popup = None
            self.update_mode_button_arrow(False)
            return

        popup = QFrame(
            self,
            Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint
        )
        popup.setObjectName("modePopup")
        # ポップアップが外側クリックなどで閉じられた場合も矢印を上向きに戻す
        popup.installEventFilter(self)
        popup.setFixedWidth(210)
        popup.setStyleSheet("""
            QFrame#modePopup {
                background-color: #242424;
                border: 1px solid #3a3a3a;
                border-radius: 15px;
            }
            QLabel#modePopupTitle {
                color: #888888;
                font-size: 11px;
                padding: 2px 8px 5px 8px;
            }
            QPushButton#modeChoice {
                color: #eeeeee;
                background-color: transparent;
                border: none;
                border-radius: 9px;
                text-align: left;
                padding: 10px 11px;
                font-size: 13px;
            }
            QPushButton#modeChoice:hover { background-color: #353535; }
            QPushButton#modeChoice:checked {
                background-color: #34303e;
                color: #d8c6ff;
            }
        """)
        layout = QVBoxLayout(popup)
        layout.setContentsMargins(9, 10, 9, 9)
        layout.setSpacing(4)

        title = QLabel("応答モード")
        title.setObjectName("modePopupTitle")
        layout.addWidget(title)

        for mode, description in (
                ("一般モード", "通常の会話や質問に"),
                ("思考モード", "複雑な問題をじっくり考える"),
        ):
            choice = QPushButton(mode + "\n" + description)
            choice.setObjectName("modeChoice")
            choice.setCheckable(True)
            choice.setChecked(self.response_mode == mode)
            choice.setMinimumHeight(56)
            choice.clicked.connect(lambda checked=False, selected=mode: self.select_response_mode(selected))
            layout.addWidget(choice)

        self.mode_popup = popup
        popup.adjustSize()
        # ボタンの上側に開く。画面上端に近い場合は利用可能な範囲で表示する。
        global_pos = self.mode_button.mapToGlobal(QPoint(0, 0))
        popup.move(global_pos.x() + self.mode_button.width() - popup.width(),
                   global_pos.y() - popup.height() - 8)
        popup.show()
        self.update_mode_button_arrow(True)

    def eventFilter(self, watched, event):
        if (self.mode_popup is not None
                and watched is self.mode_popup
                and event.type() == QEvent.Type.Hide):
            self.update_mode_button_arrow(False)
            self.mode_popup = None
        return super().eventFilter(watched, event)

    def update_mode_button_arrow(self, is_open):
        short_label = "一般モード" if self.response_mode == "一般モード" else "思考モード"
        arrow = "˅" if is_open else "˄"
        self.mode_button.setText(f"{short_label}  {arrow}")

    def select_response_mode(self, mode):
        self.response_mode = mode
        if self.mode_popup is not None:
            self.mode_popup.close()
            self.mode_popup = None
        self.update_mode_button_arrow(False)

    # =====================================================
    # ＋ボタン
    # =====================================================

    def plus_clicked(self):
        # すでに表示中なら閉じる
        if self.plus_popup is not None and self.plus_popup.isVisible():
            self.plus_popup.close()
            self.plus_popup = None
            return

        # ChatGPT風の、角丸でダークな独自ポップアップ
        popup = QFrame(
            self,
            Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint
        )
        popup.setObjectName("plusPopup")
        popup.setFixedWidth(300)
        popup.setStyleSheet("""
            QFrame#plusPopup {
                background-color: #242424;
                border: 1px solid #383838;
                border-radius: 18px;
            }
        """)

        popup_layout = QVBoxLayout(popup)
        popup_layout.setContentsMargins(6, 6, 6, 6)
        popup_layout.setSpacing(2)

        # 項目全体がクリックできる、余白のある角丸ボタン
        add_file_button = QPushButton()
        add_file_button.setObjectName("addFileButton")
        add_file_button.setCursor(Qt.CursorShape.PointingHandCursor)
        add_file_button.setMinimumHeight(50)
        add_file_button.setStyleSheet("""
            QPushButton#addFileButton {
                color: #eeeeee;
                background-color: transparent;
                border: none;
                border-radius: 12px;
                text-align: left;
            }
            QPushButton#addFileButton:hover {
                background-color: #383838;
            }
            QPushButton#addFileButton:pressed {
                background-color: #424242;
            }
        """)

        item_layout = QHBoxLayout(add_file_button)
        item_layout.setContentsMargins(12, 6, 12, 6)
        item_layout.setSpacing(9)

        paperclip_icon = PaperclipIcon(add_file_button)
        item_layout.addWidget(paperclip_icon, 0, Qt.AlignmentFlag.AlignVCenter)

        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(3)

        title_label = QLabel("写真やファイルを追加")
        title_label.setStyleSheet("""
            QLabel {
                color: #f2f2f2;
                background: transparent;
                border: none;
                font-size: 13px;
                font-weight: 600;
            }
        """)

        subtitle_label = QLabel("コンピューターからアップロード")
        subtitle_label.setStyleSheet("""
            QLabel {
                color: #a5a5a5;
                background: transparent;
                border: none;
                font-size: 11px;
            }
        """)

        text_layout.addWidget(title_label)
        text_layout.addWidget(subtitle_label)
        item_layout.addLayout(text_layout, 1)
        popup_layout.addWidget(add_file_button)

        add_file_button.clicked.connect(self.add_files)

        shadow = QGraphicsDropShadowEffect(popup)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 5)
        shadow.setColor(QColor(0, 0, 0, 190))
        popup.setGraphicsEffect(shadow)

        self.plus_popup = popup
        popup.adjustSize()
        plus_top_left = self.plus_button.mapToGlobal(QPoint(0, 0))
        popup.move(
            plus_top_left.x(),
            plus_top_left.y() - popup.height() - 10
        )
        popup.show()
        popup.raise_()

    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "ファイルを追加",
            "",
            "すべてのファイル (*)"
        )

        if not files:
            return

        for file_path in files:
            if file_path not in self.selected_files:
                self.selected_files.append(file_path)

        self.refresh_attachment_chips()

        if self.plus_popup is not None:
            self.plus_popup.close()
            self.plus_popup = None

    def refresh_attachment_chips(self):
        # 既存のファイル表示をすべて作り直す
        while self.attachment_row_layout.count():
            item = self.attachment_row_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        for file_path in self.selected_files:
            chip = QFrame()
            chip.setObjectName("attachmentChip")
            chip.setStyleSheet("""
                QFrame#attachmentChip {
                    background-color: #242424;
                    border: 1px solid #383838;
                    border-radius: 10px;
                }
            """)
            chip_layout = QHBoxLayout(chip)
            chip_layout.setContentsMargins(10, 5, 5, 5)
            chip_layout.setSpacing(8)

            name_label = QLabel(Path(file_path).name)
            name_label.setToolTip(file_path)
            name_label.setMaximumWidth(220)
            name_label.setStyleSheet("""
                QLabel {
                    color: #eeeeee;
                    background: transparent;
                    border: none;
                    font-size: 12px;
                }
            """)
            chip_layout.addWidget(name_label)

            remove_button = QPushButton("×")
            remove_button.setFixedSize(22, 22)
            remove_button.setCursor(Qt.CursorShape.PointingHandCursor)
            remove_button.setStyleSheet("""
                QPushButton {
                    color: #aaaaaa;
                    background: transparent;
                    border: none;
                    border-radius: 11px;
                    font-size: 18px;
                    padding: 0;
                }
                QPushButton:hover {
                    color: white;
                    background-color: #454545;
                }
            """)
            remove_button.clicked.connect(
                lambda checked=False, path=file_path: self.remove_selected_file(path)
            )
            chip_layout.addWidget(remove_button)
            self.attachment_row_layout.addWidget(chip)

        self.attachment_row_layout.addStretch(1)
        self.attachment_row.setVisible(bool(self.selected_files))

        if self.selected_files:
            self.input.setPlaceholderText(
                "メッセージを入力..."
            )
        else:
            self.input.setPlaceholderText("メッセージを入力...")

    def remove_selected_file(self, file_path):
        if file_path in self.selected_files:
            self.selected_files.remove(file_path)
        self.refresh_attachment_chips()

    # =====================================================
    # 保存済みチャット読み込み
    # =====================================================

    def load_saved_chats(self):
        saved_chats = self.storage.load()

        self.chats = []

        numeric_ids = []

        for chat in saved_chats:
            if not isinstance(
                    chat,
                    dict
            ):
                continue

            chat_id = chat.get(
                "id"
            )

            title = chat.get(
                "title",
                "新しいチャット"
            )

            messages = chat.get(
                "messages",
                []
            )

            if not isinstance(
                    messages,
                    list
            ):
                messages = []

            if not isinstance(
                    chat_id,
                    int
            ):
                continue

            self.chats.append({
                "id": chat_id,
                "title": title,
                "messages": messages,
                "button": None
            })

            numeric_ids.append(
                chat_id
            )

        if numeric_ids:
            self.chat_counter = max(
                numeric_ids
            )
        else:
            self.chat_counter = 0

        for chat in self.chats:
            self.add_history_button(
                chat
            )

        if self.chats:
            self.current_chat_id = (
                self.chats[-1]["id"]
            )

            self.load_chat(
                self.current_chat_id
            )

        else:
            self.current_chat_id = None
            self.clear_chat_area()

    # =====================================================
    # チャット保存
    # =====================================================

    def save_chats(self):
        save_data = []

        for chat in self.chats:
            save_data.append({
                "id": chat["id"],
                "title": chat["title"],
                "messages": chat["messages"]
            })

        self.storage.save(
            save_data
        )

    # =====================================================
    # 新しいチャット
    # =====================================================

    def create_new_chat(self):
        self.chat_counter += 1

        chat_id = self.chat_counter

        chat = {
            "id": chat_id,
            "title": f"新しいチャット {chat_id}",
            "messages": [],
            "button": None
        }

        self.chats.append(
            chat
        )

        self.current_chat_id = chat_id

        self.add_history_button(
            chat
        )

        self.clear_chat_area()

        self.save_chats()

    # =====================================================
    # 履歴ボタン追加
    # =====================================================

    def add_history_button(self, chat):
        button = ChatHistoryButton(
            chat["title"],
            chat["id"],
            self
        )

        chat["button"] = button

        self.history_layout.insertWidget(
            self.history_layout.count() - 1,
            button
        )

    # =====================================================
    # チャット取得
    # =====================================================

    def get_chat(self, chat_id):
        # 保存データ由来のIDが数値、スレッド由来のIDが文字列でも
        # 同じチャットとして取得できるように比較形式を統一する。
        if chat_id is None:
            return None
        wanted_id = str(chat_id)
        for chat in self.chats:
            if chat.get("id") is not None and str(chat.get("id")) == wanted_id:
                return chat
        return None

    # =====================================================
    # チャット読み込み
    # =====================================================

    def load_chat(self, chat_id):
        chat = self.get_chat(
            chat_id
        )

        if chat is None:
            return

        self.current_chat_id = chat_id

        if self.delete_overlay is not None:
            self.delete_overlay.close_overlay()

        self.clear_chat_area()

        if chat["messages"]:
            self.prepare_message_area()
            for message in chat["messages"]:
                if isinstance(message, dict):
                    widget = ChatMessage(
                        message.get("text", ""),
                        message.get("attachments", []),
                        role=message.get("role", "user")
                    )
                else:
                    widget = ChatMessage(message)
                self.chat_layout.insertWidget(
                    self.chat_layout.count() - 1,
                    widget
                )
        else:
            self.show_empty_state()

        self.scroll_to_bottom()

    # =====================================================
    # チャット表示クリア・空状態
    # =====================================================

    def clear_chat_area(self):
        self._remove_search_status_widget()
        while self.chat_layout.count():
            item = self.chat_layout.takeAt(0)
            widget = item.widget()
            if widget is not None and widget is not self.empty_state_widget:
                widget.deleteLater()

        self.empty_state_active = False
        self.show_empty_state()

    def show_empty_state(self):
        # 既存のレイアウト項目を空状態用に組み直す
        while self.chat_layout.count():
            item = self.chat_layout.takeAt(0)
            widget = item.widget()
            if widget is not None and widget is not self.empty_state_widget:
                widget.deleteLater()

        self.empty_state_widget.show()
        self.chat_layout.addStretch(1)
        self.chat_layout.addWidget(self.empty_state_widget, 1)
        self.chat_layout.addStretch(1)
        self.empty_state_active = True

    def prepare_message_area(self):
        # 空状態のラベルや伸縮スペースを外し、末尾の伸縮スペースだけ残す
        while self.chat_layout.count():
            item = self.chat_layout.takeAt(0)
            widget = item.widget()
            if widget is self.empty_state_widget:
                self.empty_state_widget.hide()
            elif widget is not None:
                widget.deleteLater()

        self.empty_state_active = False
        self.chat_layout.addStretch(1)

    # =====================================================
    # メッセージ送信
    # =====================================================

    def send_message(self):
        if self._answer_pending:
            return
        text = self.input.text().strip()
        selected_paths = list(self.selected_files)

        # 文字もファイルもない場合だけ送信しない
        if not text and not selected_paths:
            return

        chat = self.get_chat(self.current_chat_id)
        if chat is None:
            self.create_new_chat()
            chat = self.get_chat(self.current_chat_id)
        if chat is None:
            return

        if not chat["messages"] and self.empty_state_active:
            self.prepare_message_area()

        if len(chat["messages"]) == 0:
            title_source = text or Path(selected_paths[0]).name
            chat["title"] = title_source[:18] + ("..." if len(title_source) > 18 else "")
            if chat["button"] is not None:
                chat["button"].setText(chat["title"])

        attachments = []
        for file_path in selected_paths:
            file_name = Path(file_path).name
            mime_type = mimetypes.guess_type(file_name)[0] or "application/octet-stream"
            attachments.append({
                "name": file_name,
                "path": file_path,
                "type": mime_type,
            })

        # ユーザーメッセージを保存・表示
        saved_message = {"text": text, "attachments": attachments} if attachments else text
        chat["messages"].append(saved_message)
        message_widget = ChatMessage(text, attachments)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, message_widget)

        # AI検索・回答はバックグラウンドで実行。既存のチャットUIを維持する。
        if text:
            prior_messages = []
            for item in chat["messages"][:-1]:
                if isinstance(item, dict):
                    prior_messages.append(item.get("text", ""))
                else:
                    prior_messages.append(str(item))
            self._answer_pending = True
            self.input.setEnabled(False)
            self.input.setPlaceholderText("AngelWhiteが検索・回答を作成中...")
            worker = AIResponseThread(self.ai_engine, chat.get("id", self.current_chat_id), text,
                                      self.response_mode, prior_messages)
            worker.response_ready.connect(self._on_ai_response)
            worker.search_progress.connect(self._on_ai_search_progress)
            worker.finished.connect(lambda w=worker: self._cleanup_answer_thread(w))
            self._answer_threads.append(worker)
            worker.start()

        self.input.clear()
        self.selected_files.clear()
        self.refresh_attachment_chips()
        self.scroll_to_bottom()
        self.save_chats()

    def _on_ai_search_progress(self, chat_id, status_text):
        """AIスレッドから届いた検索進捗を、現在開いている同じチャットに表示する。"""
        if str(self.current_chat_id) != str(chat_id):
            return

        if (self._search_status_widget is None
                or self._search_status_chat_id != str(chat_id)):
            self._remove_search_status_widget()
            self._search_status_widget = SearchStatusWidget(status_text)
            self._search_status_chat_id = str(chat_id)
            insert_at = max(0, self.chat_layout.count() - 1)
            self.chat_layout.insertWidget(insert_at, self._search_status_widget)
        else:
            self._search_status_widget.set_status_text(status_text)

        self.scroll_to_bottom()

    def _remove_search_status_widget(self, chat_id=None):
        """検索進捗表示をレイアウトから取り外す。"""
        if chat_id is not None and self._search_status_chat_id != str(chat_id):
            return
        widget = self._search_status_widget
        self._search_status_widget = None
        self._search_status_chat_id = None
        if widget is not None:
            self.chat_layout.removeWidget(widget)
            widget.hide()
            widget.deleteLater()

    def _on_ai_response(self, chat_id, question, answer):
        self._remove_search_status_widget(chat_id)
        self._answer_pending = False
        self.input.setEnabled(True)
        self.input.setPlaceholderText("メッセージを送信")
        target = self.get_chat(chat_id)
        if target is not None:
            target["messages"].append({"role": "assistant", "text": answer})
            if str(self.current_chat_id) == str(chat_id):
                assistant_widget = ChatMessage(answer, role="assistant")
                self.chat_layout.insertWidget(self.chat_layout.count() - 1, assistant_widget)
                self.scroll_to_bottom()
            self.save_chats()

    def _cleanup_answer_thread(self, worker):
        if worker in self._answer_threads:
            self._answer_threads.remove(worker)
        # 応答シグナルが何らかの理由で届かなかった場合にも、入力欄を復帰させる。
        if self._answer_pending:
            self._remove_search_status_widget(getattr(worker, "chat_id", None))
            self._answer_pending = False
            self.input.setEnabled(True)
            self.input.setPlaceholderText("メッセージを送信")
        worker.deleteLater()

    # =====================================================
    # 最下部へスクロール
    # =====================================================

    def scroll_to_bottom(self):
        scrollbar = (
            self.chat_scroll.verticalScrollBar()
        )

        scrollbar.setValue(
            scrollbar.maximum()
        )

    # =====================================================
    # 削除確認画面
    # =====================================================

    def show_delete_confirmation(
            self,
            chat_id
    ):
        chat = self.get_chat(
            chat_id
        )

        if chat is None:
            return

        if self.delete_overlay is not None:
            self.delete_overlay.close_overlay()

        self.delete_overlay = DeleteConfirmationOverlay(
            self,
            chat["title"],
            lambda: self.perform_delete_chat(
                chat_id
            )
        )

        self.delete_overlay.setGeometry(
            self.rect()
        )

        self.delete_overlay.show()

        self.delete_overlay.raise_()

    # =====================================================
    # チャット完全削除
    # =====================================================

    def perform_delete_chat(
            self,
            chat_id
    ):
        chat = self.get_chat(
            chat_id
        )

        if chat is None:
            return

        # =================================================
        # データから完全削除
        # =================================================

        self.chats = [
            chat_data
            for chat_data in self.chats
            if chat_data["id"] != chat_id
        ]

        # =================================================
        # サイドバーから完全削除
        # =================================================

        button = chat.get(
            "button"
        )

        if button is not None:
            button.deleteLater()

        # =================================================
        # 現在のチャットだった場合
        # =================================================

        if self.current_chat_id == chat_id:
            self.current_chat_id = None

            self.clear_chat_area()

            # 別のチャットが残っている場合
            if self.chats:
                self.current_chat_id = (
                    self.chats[-1]["id"]
                )

                self.load_chat(
                    self.current_chat_id
                )

            # 何もなくなった場合は空状態を表示する
            else:
                self.clear_chat_area()

        # =================================================
        # 保存
        # =================================================

        self.save_chats()

    # =====================================================
    # ウィンドウサイズ変更
    # =====================================================

    def resizeEvent(self, event):
        super().resizeEvent(event)

        if self.delete_overlay is not None:
            self.delete_overlay.setGeometry(
                self.rect()
            )

        if not self.sidebar_visible:
            self.open_sidebar_button.move(
                12,
                12
            )

    # =====================================================
    # 終了時保存
    # =====================================================

    def closeEvent(self, event):
        self.save_chats()

        event.accept()


# =========================================================
# メイン
# =========================================================

def main():
    app = QApplication(sys.argv)

    app.setApplicationName(
        "AngelWhite"
    )

    app.setFont(
        QFont(
            "Segoe UI",
            10
        )
    )

    window = AngelWhiteWindow()

    window.show()

    sys.exit(
        app.exec()
    )


if __name__ == "__main__":
    main()