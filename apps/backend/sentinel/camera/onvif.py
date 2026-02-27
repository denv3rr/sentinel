from __future__ import annotations

import re
import socket
import uuid
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass
class OnvifEndpoint:
    id: str
    xaddr: str
    host: str


def discover_onvif(timeout_seconds: float = 2.0) -> list[OnvifEndpoint]:
    """Best-effort ONVIF Profile S discovery via WS-Discovery UDP probe."""
    probe = f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<e:Envelope xmlns:e=\"http://www.w3.org/2003/05/soap-envelope\"
            xmlns:w=\"http://schemas.xmlsoap.org/ws/2004/08/addressing\"
            xmlns:d=\"http://schemas.xmlsoap.org/ws/2005/04/discovery\"
            xmlns:dn=\"http://www.onvif.org/ver10/network/wsdl\">
  <e:Header>
    <w:MessageID>uuid:{uuid.uuid4()}</w:MessageID>
    <w:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>
    <w:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>
  </e:Header>
  <e:Body>
    <d:Probe>
      <d:Types>dn:NetworkVideoTransmitter</d:Types>
    </d:Probe>
  </e:Body>
</e:Envelope>"""

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
    sock.settimeout(timeout_seconds)

    results: dict[str, OnvifEndpoint] = {}
    try:
        sock.sendto(probe.encode("utf-8"), ("239.255.255.250", 3702))
        while True:
            try:
                data, _ = sock.recvfrom(65535)
            except socket.timeout:
                break
            text = data.decode("utf-8", errors="ignore")
            xaddrs = re.findall(r"<XAddrs>(.*?)</XAddrs>", text, flags=re.IGNORECASE | re.DOTALL)
            for block in xaddrs:
                for xaddr in block.split():
                    parsed = urlparse(xaddr)
                    host = parsed.hostname or "unknown"
                    key = f"{host}-{xaddr}"
                    if key not in results:
                        results[key] = OnvifEndpoint(
                            id=f"onvif-{len(results) + 1}",
                            xaddr=xaddr,
                            host=host,
                        )
    finally:
        sock.close()

    return list(results.values())


def guess_rtsp_candidates(host: str, username: str | None = None, password: str | None = None) -> list[str]:
    cred = ""
    if username and password:
        cred = f"{username}:{password}@"
    elif username:
        cred = f"{username}@"

    base = f"rtsp://{cred}{host}"
    return [
        f"{base}/Streaming/Channels/101",
        f"{base}/cam/realmonitor?channel=1&subtype=0",
        f"{base}/h264/ch1/main/av_stream",
        f"{base}/live",
    ]