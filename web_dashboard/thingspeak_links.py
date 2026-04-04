"""
Build read-only ThingSpeak **web UI** links for the local dashboard.

Uses the public MathWorks ThingSpeak site (not api.thingspeak.com).
Channel pages include field charts when the channel is shared or you are logged in.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ThingSpeakChannelLinks:
    channel_id: str
    label: str
    channel_url: str


def _web_base() -> str:
    return os.getenv("AGROEDGE_THINGSPEAK_WEB_BASE", "https://thingspeak.com").rstrip("/")


def _channel_url(channel_id: str) -> str:
    cid = str(channel_id).strip()
    return f"{_web_base()}/channels/{cid}"


def collect_configured_links() -> list[ThingSpeakChannelLinks]:
    """Return links for any channel IDs present in the environment."""
    out: list[ThingSpeakChannelLinks] = []
    env_id = os.getenv("THINGSPEAK_ENV_CHANNEL_ID", "").strip()
    if env_id:
        out.append(
            ThingSpeakChannelLinks(
                channel_id=env_id,
                label="Environmental sensors (rice_env)",
                channel_url=_channel_url(env_id),
            )
        )
    irr_id = os.getenv("THINGSPEAK_IRRIGATION_CHANNEL_ID", "").strip()
    if irr_id:
        out.append(
            ThingSpeakChannelLinks(
                channel_id=irr_id,
                label="Irrigation log (rice_irrigation)",
                channel_url=_channel_url(irr_id),
            )
        )
    return out


def links_section_html() -> str:
    """HTML fragment for dashboard; empty instruction if no IDs configured."""
    links = collect_configured_links()
    if not links:
        return (
            "<section><h2>ThingSpeak</h2>"
            "<p>Set <code>THINGSPEAK_ENV_CHANNEL_ID</code> and/or "
            "<code>THINGSPEAK_IRRIGATION_CHANNEL_ID</code> in the environment "
            "to show channel links (read-only).</p></section>"
        )
    items = "".join(
        f"<li><a href='{link.channel_url}' rel='noopener noreferrer' target='_blank'>"
        f"{link.label}</a> <span class='muted'>(channel {link.channel_id})</span></li>"
        for link in links
    )
    return (
        "<section><h2>ThingSpeak</h2>"
        "<ul style='margin-top:0.5rem'>"
        f"{items}"
        "</ul>"
        "<p class='muted' style='font-size:90%'>Opens the MathWorks ThingSpeak channel page "
        "(field graphs require a shared channel or an active login).</p></section>"
    )
