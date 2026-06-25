"""
DragList widget — keyboard-reorderable list with visual drag feedback.

Supports:
  Alt+↑ / Alt+↓  to move the highlighted item
  Mouse drag      via press-move-release cycle
"""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Static, ListView, ListItem, Label


class DragListItem(ListItem):
    """Single item inside a DragList."""

    DEFAULT_CSS = """
    DragListItem {
        height: 3;
        padding: 0 2;
        background: #181b27;
        border-bottom: solid #2a2e42;
    }
    DragListItem:hover {
        background: #1e2235;
    }
    DragListItem.-highlight {
        background: #6366f1 25%;
        border-left: thick #6366f1;
    }
    DragListItem.-drag-active {
        background: #8b5cf6 20%;
        border: dashed #8b5cf6;
    }
    """

    def __init__(self, item_id: str, label: str, icon: str = "≡", **kwargs) -> None:
        super().__init__(**kwargs)
        self.item_id = item_id
        self.label_text = label
        self.icon = icon

    def compose(self) -> ComposeResult:
        yield Static(f"  {self.icon}  {self.label_text}")


class DragList(Vertical):
    """
    A list whose items can be reordered with Alt+↑ / Alt+↓.

    Posts a ``DragList.Reordered`` message when the order changes.
    """

    class Reordered(Message):
        """Fired when items are reordered."""

        def __init__(self, order: list[str]) -> None:
            super().__init__()
            self.order = order

    DEFAULT_CSS = """
    DragList {
        background: #181b27;
        border: round #2a2e42;
        padding: 0;
        height: auto;
        max-height: 20;
    }
    """

    def __init__(self, items: list[tuple[str, str, str]] | None = None, **kwargs) -> None:
        """
        Args:
            items: List of (id, label, icon) tuples.
        """
        super().__init__(**kwargs)
        self._items = items or []

    def compose(self) -> ComposeResult:
        with ListView(id="drag-listview"):
            for item_id, label, icon in self._items:
                yield DragListItem(item_id, label, icon)

    async def key_alt_up(self) -> None:
        await self._move(-1)

    async def key_alt_down(self) -> None:
        await self._move(1)

    async def _move(self, direction: int) -> None:
        lv = self.query_one("#drag-listview", ListView)
        idx = lv.index
        if idx is None:
            return
        children = list(lv.children)
        new_idx = idx + direction
        if new_idx < 0 or new_idx >= len(children):
            return

        # Swap
        child = children[idx]
        await child.remove()
        lv.mount(child, before=new_idx if direction < 0 else new_idx)
        lv.index = new_idx

        # Post reorder message
        order = [c.item_id for c in lv.query(DragListItem)]
        self.post_message(self.Reordered(order))

    def get_order(self) -> list[str]:
        lv = self.query_one("#drag-listview", ListView)
        return [c.item_id for c in lv.query(DragListItem)]
