"""
空地通信：无人机端发布栅格地图与空中发现的人员位置，小车端订阅并回传搜救状态

基于 MQTT（mosquitto），与立项阶段的 “yolo/test” 通信测试一致。消息为 JSON，
栅格地图按位压缩（packbits + zlib + base64），200x200 地图约几百字节，适合无线链路。

话题:
    rescue/map      无人机 -> 小车   栅格地图 + 地图参考信息 + 人员位置
    rescue/status   小车   -> 无人机 小车位姿、路径进度、地面确认的人员位置
"""

from __future__ import annotations

import base64
import json
import time
import zlib

import numpy as np

from .geo_mapping import GridMap

TOPIC_MAP = "rescue/map"
TOPIC_STATUS = "rescue/status"


def encode_map(gm: GridMap, persons_xy=(), extra: dict | None = None) -> bytes:
    bits = np.packbits(gm.grid.astype(bool), axis=None)
    msg = {
        "t": time.time(),
        "map": gm.to_dict(),
        "grid": base64.b64encode(zlib.compress(bits.tobytes(), 9)).decode("ascii"),
        "persons": [list(map(float, p)) for p in persons_xy],
    }
    if extra:
        msg.update(extra)
    return json.dumps(msg, separators=(",", ":")).encode("utf-8")


def decode_map(payload: bytes):
    msg = json.loads(payload)
    meta = msg["map"]
    rows, cols = meta["shape"]
    bits = np.frombuffer(zlib.decompress(base64.b64decode(msg["grid"])), np.uint8)
    grid = np.unpackbits(bits)[:rows * cols].reshape(rows, cols).astype(np.uint8)
    gm = GridMap(grid, meta["resolution"], meta["x_min"], meta["y_max"])
    return gm, [tuple(p) for p in msg.get("persons", [])], msg


class LoopbackLink:
    """进程内消息总线：仿真与单元测试时代替 MQTT"""

    def __init__(self):
        self._subs = {}

    def subscribe(self, topic, callback):
        self._subs.setdefault(topic, []).append(callback)

    def publish(self, topic, payload: bytes):
        for cb in self._subs.get(topic, []):
            cb(topic, payload)

    def close(self):
        pass


class MqttLink:
    """paho-mqtt 封装（兼容 paho-mqtt 1.x 与 2.x）"""

    def __init__(self, host="localhost", port=1883, client_id=None, keepalive=30):
        import paho.mqtt.client as mqtt
        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id or "")
        except AttributeError:
            self.client = mqtt.Client(client_id=client_id or "")
        self._subs = {}
        self.client.on_message = self._on_message
        self.client.on_connect = self._on_connect
        self.client.connect(host, port, keepalive)
        self.client.loop_start()

    def _on_connect(self, client, userdata, *args):
        for topic in self._subs:
            client.subscribe(topic, qos=1)

    def _on_message(self, client, userdata, msg):
        for cb in self._subs.get(msg.topic, []):
            cb(msg.topic, msg.payload)

    def subscribe(self, topic, callback):
        self._subs.setdefault(topic, []).append(callback)
        self.client.subscribe(topic, qos=1)

    def publish(self, topic, payload: bytes):
        self.client.publish(topic, payload, qos=1)

    def close(self):
        self.client.loop_stop()
        self.client.disconnect()
