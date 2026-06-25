"""
Reusable MetricCard widget — displays a KPI with sparkline and trend.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static, Sparkline


class MetricCard(Static):
    """
    A compact metric card showing label, value, trend, and sparkline.

    Usage:
        card = MetricCard(label="Requests", value="1,234", trend="+12%",
                          trend_up=True, spark_data=[1,3,2,5,4],
                          spark_color="#6366f1")
    """

    DEFAULT_CSS = """
    MetricCard {
        width: 1fr;
        height: auto;
        min-height: 6;
        background: #181b27;
        border: round #2a2e42;
        padding: 1 2;
        margin: 0 1;
    }
    MetricCard:hover {
        border: round #6366f1;
    }
    MetricCard .mc-label {
        color: #8b8fa3;
        margin-bottom: 0;
    }
    MetricCard .mc-value {
        color: #e8eaf0;
        text-style: bold;
    }
    MetricCard .mc-trend-up {
        color: #34d399;
    }
    MetricCard .mc-trend-down {
        color: #f43f5e;
    }
    MetricCard .mc-trend-neutral {
        color: #8b8fa3;
    }
    MetricCard Sparkline {
        min-height: 1;
        max-height: 1;
        margin-top: 1;
    }
    """

    def __init__(
        self,
        label: str = "",
        value: str = "0",
        trend: str = "+0%",
        trend_up: bool = True,
        spark_data: list[float] | None = None,
        icon: str = "●",
        spark_color: str = "#6366f1",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._label = label
        self._value = value
        self._trend = trend
        self._trend_up = trend_up
        self._spark_data = spark_data or [0.0]
        self._icon = icon
        self._spark_color = spark_color

    def compose(self) -> ComposeResult:
        trend_cls = "mc-trend-up" if self._trend_up else "mc-trend-down"
        if self._trend == "+0%" or self._trend == "0%":
            trend_cls = "mc-trend-neutral"
        yield Static(f"{self._icon}  {self._label}", classes="mc-label")
        yield Static(self._value, classes="mc-value")
        yield Static(f"  {self._trend}", classes=trend_cls)
        sparkline = Sparkline(self._spark_data, summary_function=max)
        sparkline.styles.color = self._spark_color
        yield sparkline

    def update_data(
        self,
        value: str,
        trend: str = "+0%",
        trend_up: bool = True,
        spark_data: list[float] | None = None,
    ) -> None:
        """Update the card's displayed data."""
        self._value = value
        self._trend = trend
        self._trend_up = trend_up
        if spark_data is not None:
            self._spark_data = spark_data
        # Re-compose isn't cheap, so just update the Static children
        children = list(self.query("Static"))
        if len(children) >= 3:
            children[1].update(value)
            trend_cls = "mc-trend-up" if trend_up else "mc-trend-down"
            if trend == "+0%" or trend == "0%":
                trend_cls = "mc-trend-neutral"
            children[2].update(f"  {trend}")
            children[2].set_classes(trend_cls)
        sparks = list(self.query("Sparkline"))
        if sparks and spark_data:
            sparks[0].data = spark_data
