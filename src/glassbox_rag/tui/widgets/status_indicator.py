"""
Reusable StatusIndicator widget — shows component health with Rich markup.
"""

from __future__ import annotations

from textual.widgets import Static


_STATUS_MAP = {
    "operational": ("🟢", "#34d399"),
    "enabled": ("🟢", "#34d399"),
    "healthy": ("🟢", "#34d399"),
    "degraded": ("🟡", "#fbbf24"),
    "unhealthy": ("🔴", "#f43f5e"),
    "error": ("🔴", "#f43f5e"),
    "no_encoders": ("🟡", "#fbbf24"),
    "not_configured": ("⚪", "#5c5f73"),
    "disabled": ("⚪", "#5c5f73"),
    "initializing": ("⏳", "#8b5cf6"),
}


class StatusIndicator(Static):
    """Single component health line with aligned columns and Rich color."""

    DEFAULT_CSS = """
    StatusIndicator {
        height: 1;
        padding: 0 1;
    }
    StatusIndicator.-blink-off {
        opacity: 0.3;
    }
    """

    def __init__(self, name: str = "", status: str = "unknown", **kwargs) -> None:
        dot, color = _STATUS_MAP.get(status, ("⚪", "#5c5f73"))
        label = name.replace("_", " ").title()
        markup = f"{dot}  {label:<24} [{color}]{status}[/]"
        super().__init__(markup, markup=True, **kwargs)
        self._name = name
        self._status = status
        self._blink_timer = None

    def on_mount(self) -> None:
        if self._status == "initializing":
            self._blink_timer = self.set_interval(0.8, self._toggle_blink)

    def _toggle_blink(self) -> None:
        self.toggle_class("-blink-off")

    def update_status(self, status: str) -> None:
        dot, color = _STATUS_MAP.get(status, ("⚪", "#5c5f73"))
        label = self._name.replace("_", " ").title()
        self._status = status
        self.update(f"{dot}  {label:<24} [{color}]{status}[/]")
        # Start or stop blinking
        if status == "initializing" and self._blink_timer is None:
            self._blink_timer = self.set_interval(0.8, self._toggle_blink)
        elif status != "initializing" and self._blink_timer is not None:
            self._blink_timer.stop()
            self._blink_timer = None
            self.remove_class("-blink-off")
