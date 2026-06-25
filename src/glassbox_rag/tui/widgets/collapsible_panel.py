"""
CollapsiblePanel — a reusable toggle-able section widget.

Click the title bar to expand/collapse the content area.
Uses ▶/▼ indicators and the project's dark-theme palette.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static
from textual.reactive import reactive


class CollapsiblePanel(Vertical):
    """
    A panel with a clickable title bar that toggles content visibility.

    Usage::

        with CollapsiblePanel(title="Details", collapsed=False):
            yield Static("Some content")
            yield DataTable(...)
    """

    collapsed = reactive(False)

    DEFAULT_CSS = """
    CollapsiblePanel {
        height: auto;
        background: #181b27;
        border: round #2a2e42;
        margin-bottom: 1;
    }
    CollapsiblePanel:hover {
        border: round #6366f1;
    }
    CollapsiblePanel .cp-title-bar {
        height: 1;
        padding: 0 1;
        color: #c4b5fd;
        text-style: bold;
    }
    CollapsiblePanel .cp-title-bar:hover {
        background: #22263a;
    }
    CollapsiblePanel .cp-body {
        padding: 1 2;
    }
    CollapsiblePanel .cp-body.-hidden {
        display: none;
    }
    """

    def __init__(
        self,
        title: str = "Section",
        collapsed: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._title = title
        self.collapsed = collapsed

    def compose(self) -> ComposeResult:
        indicator = "▶" if self.collapsed else "▼"
        yield Static(
            f"{indicator}  {self._title}",
            classes="cp-title-bar",
            id=f"cp-toggle-{id(self)}",
        )
        body = Vertical(classes="cp-body")
        if self.collapsed:
            body.add_class("-hidden")
        yield body

    def on_click(self, event) -> None:
        """Toggle on click anywhere in the title bar area."""
        # Check if click is on the title bar (first line)
        title_bar = self.query_one(".cp-title-bar", Static)
        if event.y <= 0 or event.widget is title_bar:
            self.collapsed = not self.collapsed

    def watch_collapsed(self, value: bool) -> None:
        """React to collapsed state changes."""
        try:
            title_bar = self.query_one(".cp-title-bar", Static)
            body = self.query_one(".cp-body", Vertical)
            indicator = "▶" if value else "▼"
            title_bar.update(f"{indicator}  {self._title}")
            if value:
                body.add_class("-hidden")
            else:
                body.remove_class("-hidden")
        except Exception:
            pass

    def add_content(self, *widgets) -> None:
        """Programmatically add widgets to the body."""
        body = self.query_one(".cp-body", Vertical)
        for w in widgets:
            body.mount(w)
